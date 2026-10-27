#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>

namespace hft {

/**
 * SpscRing — bounded single-producer / single-consumer queue.
 *
 * WHY LOCK-FREE AND WHY SPSC:
 *   The hot path has exactly one feed thread producing parsed messages and one
 *   book thread consuming them. An SPSC ring needs only two indices and
 *   acquire/release ordering; no mutex, no CAS loop, no contention on the data.
 *   A multi-producer queue would need CAS and would be slower for no benefit.
 *
 * MEMORY ORDERING:
 *   The producer writes the slot, then publishes `tail_` with release. The
 *   consumer reads `tail_` with acquire before reading the slot, so the payload
 *   is visible. The reverse holds for `head_`: the consumer finishes with a slot
 *   before publishing `head_`, and the producer acquires it before overwriting.
 *
 * CAPACITY:
 *   Must be a power of two. One slot is always empty, so usable capacity is
 *   Capacity - 1. This is the standard way to distinguish full from empty
 *   without a third counter.
 *
 * WHAT IS NOT LOCK-FREE:
 *   Copying a `T` that allocates (e.g. std::string) calls the allocator, which
 *   may take a lock. The queue's synchronisation is lock-free; keep `T`
 *   trivially copyable on the hot path and parse strings before enqueueing.
 */
template <typename T, std::size_t Capacity>
class SpscRing {
    static_assert(Capacity >= 2, "capacity must be at least 2");
    static_assert((Capacity & (Capacity - 1)) == 0,
                  "capacity must be a power of two");

public:
    static constexpr std::size_t capacity() noexcept { return Capacity - 1; }

    bool try_push(const T& value) noexcept {
        const std::size_t tail = tail_.load(std::memory_order_relaxed);
        const std::size_t next = (tail + 1) & (Capacity - 1);
        if (next == head_.load(std::memory_order_acquire)) {
            return false;  // full
        }
        buffer_[tail] = value;
        tail_.store(next, std::memory_order_release);
        return true;
    }

    bool try_pop(T& out) noexcept {
        const std::size_t head = head_.load(std::memory_order_relaxed);
        if (head == tail_.load(std::memory_order_acquire)) {
            return false;  // empty
        }
        out = buffer_[head];
        head_.store((head + 1) & (Capacity - 1), std::memory_order_release);
        return true;
    }

    bool empty() const noexcept {
        return head_.load(std::memory_order_acquire) ==
               tail_.load(std::memory_order_acquire);
    }

    std::size_t size() const noexcept {
        const std::size_t head = head_.load(std::memory_order_acquire);
        const std::size_t tail = tail_.load(std::memory_order_acquire);
        return (tail - head) & (Capacity - 1);
    }

private:
    std::array<T, Capacity> buffer_{};
    // Separate cache lines: a producer updating tail_ must not invalidate the
    // line holding head_ for the consumer.
    alignas(64) std::atomic<std::size_t> head_{0};
    alignas(64) std::atomic<std::size_t> tail_{0};
};

}  // namespace hft
