// Microbenchmark for the lock-free pieces: SPSC handoff and snapshot reads.
//
// Numbers are for this machine and compiler; the README quotes the output with
// the platform attached. They are not a claim about production latency.

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <thread>

#include "concurrent_book.h"
#include "spsc_ring.h"

namespace {

struct Item {
    std::uint64_t seq{0};
    double bid{0.0};
    double ask{0.0};
};

void bench_spsc() {
    constexpr std::uint64_t kMessages = 5'000'000;
    hft::SpscRing<Item, 4096> ring;
    std::atomic<bool> done{false};

    const auto start = std::chrono::steady_clock::now();

    std::thread consumer([&] {
        Item item{};
        std::uint64_t seen = 0;
        while (seen < kMessages) {
            if (ring.try_pop(item)) {
                ++seen;
            } else {
                std::this_thread::yield();
            }
        }
        done.store(true, std::memory_order_release);
    });

    std::thread producer([&] {
        for (std::uint64_t i = 0; i < kMessages; ++i) {
            const Item item{i, 1.0, 2.0};
            while (!ring.try_push(item)) std::this_thread::yield();
        }
    });

    producer.join();
    consumer.join();
    const auto elapsed = std::chrono::steady_clock::now() - start;
    const double ns = std::chrono::duration<double, std::nano>(elapsed).count();
    std::printf("spsc: messages=%llu total_ms=%.1f ns_per_message=%.1f\n",
                static_cast<unsigned long long>(kMessages), ns / 1e6, ns / kMessages);
}

void bench_book() {
    constexpr std::uint64_t kUpdates = 5'000'000;
    hft::ConcurrentOrderBook book;
    std::atomic<bool> done{false};
    std::atomic<std::uint64_t> reads{0};

    std::thread reader([&] {
        hft::L1Snapshot snap(0.0, 0.0, 0.0, 0.0);
        for (;;) {
            if (book.snapshot_l1(snap)) {
                reads.fetch_add(1, std::memory_order_relaxed);
            }
            if (done.load(std::memory_order_acquire)) break;
        }
    });

    const auto start = std::chrono::steady_clock::now();
    for (std::uint64_t i = 0; i < kUpdates; ++i) {
        const double price = 100.0 + 0.01 * static_cast<double>(i % 1000);
        book.apply_l1(price - 0.5, price + 0.5, 10.0, 10.0);
    }
    done.store(true, std::memory_order_release);
    reader.join();
    const auto elapsed = std::chrono::steady_clock::now() - start;
    const double ns = std::chrono::duration<double, std::nano>(elapsed).count();

    std::printf("book: updates=%llu total_ms=%.1f ns_per_update=%.1f "
                "snapshots=%llu retries=%llu\n",
                static_cast<unsigned long long>(kUpdates), ns / 1e6,
                ns / kUpdates,
                static_cast<unsigned long long>(reads.load()),
                static_cast<unsigned long long>(book.retries()));
}

}  // namespace

int main() {
    bench_spsc();
    bench_book();
    return 0;
}
