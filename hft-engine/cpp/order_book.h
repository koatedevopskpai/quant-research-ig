#pragma once

#include <vector>
#include <string>
#include <algorithm>
#include <stdexcept>
#include <cstdint>

namespace hft {

struct PriceLevel {
    double price;
    double size;
    int64_t order_count;

    PriceLevel(double p, double s, int64_t oc)
        : price(p), size(s), order_count(oc) {}
};

struct L1Snapshot {
    double bid_price;
    double bid_size;
    double ask_price;
    double ask_size;
    double spread;
    double mid_price;

    L1Snapshot(double bp, double bs, double ap, double as)
        : bid_price(bp), bid_size(bs), ask_price(ap), ask_size(as),
          spread(ap - bp), mid_price((bp + ap) / 2.0) {}
};

struct L2Snapshot {
    std::vector<PriceLevel> bids;
    std::vector<PriceLevel> asks;
    double timestamp;

    L2Snapshot(std::vector<PriceLevel> b, std::vector<PriceLevel> a, double ts)
        : bids(std::move(b)), asks(std::move(a)), timestamp(ts) {}
};

struct BookDepth {
    double bid_volume;
    double ask_volume;
    double imbalance;
    int64_t num_bid_levels;
    int64_t num_ask_levels;

    BookDepth(double bv, double av, int64_t nbid, int64_t nask)
        : bid_volume(bv), ask_volume(av), num_bid_levels(nbid), num_ask_levels(nask) {
        double total = bid_volume + ask_volume;
        imbalance = (total > 0.0) ? (bid_volume - ask_volume) / total : 0.0;
    }
};

class OrderBook {
public:
    OrderBook() = default;

    void apply_diff(const std::string& side, double price, double size, int64_t order_count);

    void apply_snapshot(const std::vector<PriceLevel>& bids,
                        const std::vector<PriceLevel>& asks);

    L1Snapshot l1_snapshot() const;

    L2Snapshot l2_snapshot(int64_t max_levels) const;

    BookDepth depth_summary(int64_t max_levels = 100) const;

    int64_t bid_count() const { return static_cast<int64_t>(bids_.size()); }
    int64_t ask_count() const { return static_cast<int64_t>(asks_.size()); }

private:
    std::vector<PriceLevel> bids_;  // sorted descending
    std::vector<PriceLevel> asks_;  // sorted ascending

    static bool bids_descending(const PriceLevel& a, const PriceLevel& b) {
        return a.price > b.price;
    }

    static bool asks_ascending(const PriceLevel& a, const PriceLevel& b) {
        return a.price < b.price;
    }

    int64_t find_price_index(const std::vector<PriceLevel>& levels,
                             double price, bool ascending) const;
};

} // namespace hft
