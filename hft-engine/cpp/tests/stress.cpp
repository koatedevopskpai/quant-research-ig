// Concurrency stress test for the lock-free path.
//
// What it proves:
//   1. SPSC ring: no loss, no duplication, strict order under a producer and a
//      consumer running flat out.
//   2. ConcurrentOrderBook: readers never see a torn snapshot (bid < ask), and
//      the writer's update count matches exactly.
//
// Run under ThreadSanitizer in CI:
//   cmake -DHFT_BUILD_STRESS=ON -DCMAKE_CXX_FLAGS="-fsanitize=thread -g -O1" ...
//   ./hft_stress
//
// A failure exits non-zero so CI fails.

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <thread>
#include <vector>

#include "concurrent_book.h"
#include "spsc_ring.h"

namespace {

struct Item {
    std::uint64_t seq{0};
};

int spsc_stress() {
    constexpr std::uint64_t kMessages = 1'000'000;
    hft::SpscRing<Item, 1024> ring;

    std::atomic<bool> producer_done{false};
    std::uint64_t consumed = 0;
    bool ordered = true;

    std::thread consumer([&] {
        Item item{};
        std::uint64_t expected = 0;
        for (;;) {
            if (ring.try_pop(item)) {
                if (item.seq != expected) ordered = false;
                ++expected;
                ++consumed;
                continue;
            }
            if (producer_done.load(std::memory_order_acquire)) {
                while (ring.try_pop(item)) {
                    if (item.seq != expected) ordered = false;
                    ++expected;
                    ++consumed;
                }
                break;
            }
            std::this_thread::yield();
        }
    });

    std::thread producer([&] {
        for (std::uint64_t i = 0; i < kMessages; ++i) {
            const Item item{i};
            while (!ring.try_push(item)) {
                std::this_thread::yield();
            }
        }
        producer_done.store(true, std::memory_order_release);
    });

    producer.join();
    consumer.join();

    const bool ok = ordered && consumed == kMessages;
    std::printf("spsc: messages=%llu consumed=%llu ordered=%s\n",
                static_cast<unsigned long long>(kMessages),
                static_cast<unsigned long long>(consumed),
                ordered ? "yes" : "NO");
    return ok ? 0 : 1;
}

int book_stress() {
    constexpr std::uint64_t kUpdates = 500'000;
    hft::ConcurrentOrderBook book;
    std::atomic<bool> writer_done{false};
    std::atomic<std::uint64_t> valid{0};
    std::atomic<std::uint64_t> torn{0};

    auto reader = [&] {
        for (;;) {
            hft::L1Snapshot snap(0.0, 0.0, 0.0, 0.0);
            if (book.snapshot_l1(snap)) {
                if (snap.bid_price < snap.ask_price) {
                    valid.fetch_add(1, std::memory_order_relaxed);
                } else {
                    torn.fetch_add(1, std::memory_order_relaxed);
                }
            }
            if (writer_done.load(std::memory_order_acquire)) break;
        }
    };

    std::vector<std::thread> readers;
    for (int i = 0; i < 3; ++i) readers.emplace_back(reader);

    for (std::uint64_t i = 0; i < kUpdates; ++i) {
        const double price = 100.0 + 0.01 * static_cast<double>(i % 1000);
        book.apply_l1(price - 0.5, price + 0.5, 10.0, 10.0);
    }
    writer_done.store(true, std::memory_order_release);
    for (auto& thread : readers) thread.join();

    const bool ok = torn.load() == 0
                    && book.updates() == kUpdates
                    && valid.load() > 0;
    std::printf("book: updates=%llu valid_snapshots=%llu torn=%llu retries=%llu\n",
                static_cast<unsigned long long>(book.updates()),
                static_cast<unsigned long long>(valid.load()),
                static_cast<unsigned long long>(torn.load()),
                static_cast<unsigned long long>(book.retries()));
    return ok ? 0 : 1;
}

}  // namespace

int main() {
    const int spsc = spsc_stress();
    const int book = book_stress();
    if (spsc != 0 || book != 0) {
        std::printf("STRESS FAILED\n");
        return 1;
    }
    std::printf("STRESS OK\n");
    return 0;
}
