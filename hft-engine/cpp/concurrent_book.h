#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#include "order_book.h"

namespace hft {

/**
 * ConcurrentOrderBook — single-writer L1/L2 book with lock-free snapshot reads.
 *
 * WHY THIS SHAPE:
 *   A single writer owning the book is simpler and faster than concurrent
 *   mutation, and it matches the real pipeline: the feed thread applies
 *   messages in sequence order. Readers (dashboards, strategy threads,
 *   monitoring) must never block the writer, so snapshots use a seqlock:
 *   the writer bumps an even/odd sequence around a mutation; readers retry if
 *   the sequence moved. No mutex, no reader lock, no allocation.
 *
 * WHY ATOMIC FIELDS, NOT PLAIN DOUBLES:
 *   A seqlock over plain fields is a data race under the C++ memory model even
 *   when readers retry — the race is UB, not just a torn value. Every shared
 *   field is therefore an atomic (relaxed inside the seqlock, the sequence
 *   carries the ordering). ThreadSanitizer in CI enforces this.
 *
 * HONEST SCOPE:
 *   Readers are lock-free (they retry, they never block the writer). The
 *   writer is single-threaded by contract; this is not a multi-writer lock-free
 *   book, and it is not claimed to be.
 */
class ConcurrentOrderBook {
public:
    static constexpr std::size_t MaxLevels = 64;

    static_assert(std::atomic<std::uint64_t>::is_always_lock_free,
                  "seqlock requires lock-free 64-bit atomics");
    static_assert(std::atomic<std::uint32_t>::is_always_lock_free,
                  "counters must be lock-free");

    struct AtomicLevel {
        std::atomic<std::uint64_t> price_bits{0};
        std::atomic<std::uint64_t> size_bits{0};
        std::atomic<std::int64_t> count{0};

        void store(const PriceLevel& level) noexcept {
            price_bits.store(encode(level.price), std::memory_order_relaxed);
            size_bits.store(encode(level.size), std::memory_order_relaxed);
            count.store(level.order_count, std::memory_order_relaxed);
        }

        PriceLevel load() const noexcept {
            return PriceLevel(decode(price_bits.load(std::memory_order_relaxed)),
                              decode(size_bits.load(std::memory_order_relaxed)),
                              count.load(std::memory_order_relaxed));
        }

    private:
        static std::uint64_t encode(double value) noexcept {
            std::uint64_t bits = 0;
            static_assert(sizeof(bits) == sizeof(value), "double must be 64-bit");
            std::memcpy(&bits, &value, sizeof(bits));
            return bits;
        }

        static double decode(std::uint64_t bits) noexcept {
            double value = 0.0;
            std::memcpy(&value, &bits, sizeof(value));
            return value;
        }
    };

    ConcurrentOrderBook() = default;

    // Single writer only.
    void apply_l1(double bid, double ask, double bid_size = 0.0,
                  double ask_size = 0.0) noexcept {
        const std::uint64_t seq = seq_.load(std::memory_order_relaxed);
        seq_.store(seq + 1, std::memory_order_release);  // odd: writing

        if (bid > 0.0) {
            bids_[0].store(PriceLevel(bid, bid_size, 0));
            bid_count_.store(1, std::memory_order_relaxed);
        } else {
            bid_count_.store(0, std::memory_order_relaxed);
        }
        if (ask > 0.0) {
            asks_[0].store(PriceLevel(ask, ask_size, 0));
            ask_count_.store(1, std::memory_order_relaxed);
        } else {
            ask_count_.store(0, std::memory_order_relaxed);
        }

        seq_.store(seq + 2, std::memory_order_release);  // even: stable
        updates_.fetch_add(1, std::memory_order_relaxed);
    }

    void apply_diff(const std::string& side, double price, double size,
                    std::int64_t order_count = 0) noexcept {
        const bool is_bid = (side == "bid" || side == "BID");
        std::uint64_t seq = seq_.load(std::memory_order_relaxed);
        seq_.store(seq + 1, std::memory_order_release);

        auto& levels = is_bid ? bids_ : asks_;
        std::atomic<std::uint32_t>& count = is_bid ? bid_count_ : ask_count_;
        const std::uint32_t n = count.load(std::memory_order_relaxed);

        int found = -1;
        for (std::uint32_t i = 0; i < n; ++i) {
            const double level_price = levels[i].load().price;
            if (level_price == price) {
                found = static_cast<int>(i);
                break;
            }
            const bool beyond = is_bid ? (price > level_price) : (price < level_price);
            if (beyond) {
                found = -2 - static_cast<int>(i);  // insertion point marker
                break;
            }
        }
        if (found == -1) {
            found = -2 - static_cast<int>(n);
        }

        if (size <= 0.0) {
            if (found >= 0) {
                for (std::uint32_t i = static_cast<std::uint32_t>(found);
                     i + 1 < n; ++i) {
                    levels[i].store(levels[i + 1].load());
                }
                if (n > 0) count.store(n - 1, std::memory_order_relaxed);
            }
        } else if (found >= 0) {
            levels[found].store(PriceLevel(price, size, order_count));
        } else {
            const int insert_at = -2 - found;
            if (n < MaxLevels) {
                for (int i = static_cast<int>(n); i > insert_at; --i) {
                    levels[i].store(levels[i - 1].load());
                }
                levels[insert_at].store(PriceLevel(price, size, order_count));
                count.store(n + 1, std::memory_order_relaxed);
            }
        }

        seq_.store(seq + 2, std::memory_order_release);
        updates_.fetch_add(1, std::memory_order_relaxed);
    }

    // Lock-free reader. Returns false if the book has no bid or no ask, or if
    // the writer was too busy to yield a consistent snapshot within the budget.
    bool snapshot_l1(L1Snapshot& out) const noexcept {
        for (int attempt = 0; attempt < kMaxRetries; ++attempt) {
            const std::uint64_t before = seq_.load(std::memory_order_acquire);
            if (before & 1U) {
                cpu_relax();
                continue;
            }
            const std::uint32_t n_bid = bid_count_.load(std::memory_order_relaxed);
            const std::uint32_t n_ask = ask_count_.load(std::memory_order_relaxed);
            const PriceLevel bid = n_bid ? bids_[0].load() : PriceLevel(0, 0, 0);
            const PriceLevel ask = n_ask ? asks_[0].load() : PriceLevel(0, 0, 0);

            std::atomic_thread_fence(std::memory_order_acquire);
            const std::uint64_t after = seq_.load(std::memory_order_relaxed);
            if (before == after && !(after & 1U)) {
                if (n_bid == 0 || n_ask == 0) return false;
                out = L1Snapshot(bid.price, bid.size, ask.price, ask.size);
                snapshots_.fetch_add(1, std::memory_order_relaxed);
                return true;
            }
            retries_.fetch_add(1, std::memory_order_relaxed);
        }
        return false;
    }

    bool snapshot_l2(std::int64_t max_levels, L2Snapshot& out) const noexcept {
        for (int attempt = 0; attempt < kMaxRetries; ++attempt) {
            const std::uint64_t before = seq_.load(std::memory_order_acquire);
            if (before & 1U) {
                cpu_relax();
                continue;
            }
            const std::uint32_t n_bid = bid_count_.load(std::memory_order_relaxed);
            const std::uint32_t n_ask = ask_count_.load(std::memory_order_relaxed);
            const std::uint32_t limit = static_cast<std::uint32_t>(
                max_levels <= 0 ? MaxLevels : max_levels);
            const std::uint32_t take_bid = n_bid < limit ? n_bid : limit;
            const std::uint32_t take_ask = n_ask < limit ? n_ask : limit;

            std::vector<PriceLevel> bids;
            std::vector<PriceLevel> asks;
            bids.reserve(take_bid);
            asks.reserve(take_ask);
            for (std::uint32_t i = 0; i < take_bid; ++i) bids.push_back(bids_[i].load());
            for (std::uint32_t i = 0; i < take_ask; ++i) asks.push_back(asks_[i].load());

            std::atomic_thread_fence(std::memory_order_acquire);
            const std::uint64_t after = seq_.load(std::memory_order_relaxed);
            if (before == after && !(after & 1U)) {
                out = L2Snapshot(std::move(bids), std::move(asks), 0.0);
                snapshots_.fetch_add(1, std::memory_order_relaxed);
                return true;
            }
            retries_.fetch_add(1, std::memory_order_relaxed);
        }
        return false;
    }

    std::uint64_t updates() const noexcept {
        return updates_.load(std::memory_order_relaxed);
    }
    std::uint64_t snapshots() const noexcept {
        return snapshots_.load(std::memory_order_relaxed);
    }
    std::uint64_t retries() const noexcept {
        return retries_.load(std::memory_order_relaxed);
    }

private:
    static void cpu_relax() noexcept {
#if defined(__x86_64__) || defined(_M_X64)
        _mm_pause();
#else
        std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
    }

    static constexpr int kMaxRetries = 4096;

    std::array<AtomicLevel, MaxLevels> bids_{};
    std::array<AtomicLevel, MaxLevels> asks_{};
    std::atomic<std::uint32_t> bid_count_{0};
    std::atomic<std::uint32_t> ask_count_{0};
    alignas(64) std::atomic<std::uint64_t> seq_{0};
    std::atomic<std::uint64_t> updates_{0};
    // Mutable: snapshot readers count themselves; this is instrumentation, not
    // observable book state.
    mutable std::atomic<std::uint64_t> snapshots_{0};
    mutable std::atomic<std::uint64_t> retries_{0};
};

}  // namespace hft
