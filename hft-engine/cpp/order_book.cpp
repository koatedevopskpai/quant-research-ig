#include "order_book.h"
#include "feed_handler.h"
#include <algorithm>
#include <cmath>

namespace hft {

int64_t OrderBook::find_price_index(const std::vector<PriceLevel>& levels,
                                     double price, bool ascending) const {
    int64_t lo = 0;
    int64_t hi = static_cast<int64_t>(levels.size()) - 1;

    while (lo <= hi) {
        int64_t mid = lo + (hi - lo) / 2;
        double mid_price = levels[static_cast<size_t>(mid)].price;

        if (std::abs(mid_price - price) < 1e-10) {
            return mid;
        }
        if (ascending) {
            if (mid_price < price) lo = mid + 1;
            else hi = mid - 1;
        } else {
            if (mid_price > price) lo = mid + 1;
            else hi = mid - 1;
        }
    }
    return -1;
}

void OrderBook::apply_diff(const std::string& side, double price,
                            double size, int64_t order_count) {
    bool is_bid = (side == "bid");
    auto& levels = is_bid ? bids_ : asks_;
    bool ascending = !is_bid;

    int64_t idx = find_price_index(levels, price, ascending);

    if (size == 0.0) {
        if (idx >= 0) {
            levels.erase(levels.begin() + static_cast<ptrdiff_t>(idx));
        }
        return;
    }

    if (idx >= 0) {
        levels[static_cast<size_t>(idx)] = PriceLevel(price, size, order_count);
    } else {
        PriceLevel new_level(price, size, order_count);
        typename std::vector<PriceLevel>::iterator pos;
        if (ascending) {
            pos = std::upper_bound(levels.begin(), levels.end(), new_level,
                [](const PriceLevel& a, const PriceLevel& b) { return a.price < b.price; });
        } else {
            pos = std::upper_bound(levels.begin(), levels.end(), new_level,
                [](const PriceLevel& a, const PriceLevel& b) { return a.price > b.price; });
        }
        levels.insert(pos, new_level);
    }
}

void OrderBook::apply_snapshot(const std::vector<PriceLevel>& bids,
                                const std::vector<PriceLevel>& asks) {
    bids_ = bids;
    asks_ = asks;

    std::sort(bids_.begin(), bids_.end(),
              [](const PriceLevel& a, const PriceLevel& b) { return a.price > b.price; });
    std::sort(asks_.begin(), asks_.end(),
              [](const PriceLevel& a, const PriceLevel& b) { return a.price < b.price; });
}

L1Snapshot OrderBook::l1_snapshot() const {
    if (bids_.empty() || asks_.empty()) {
        throw std::runtime_error("Order book is empty on one side");
    }
    const auto& best_bid = bids_.front();
    const auto& best_ask = asks_.front();
    return L1Snapshot(best_bid.price, best_bid.size,
                      best_ask.price, best_ask.size);
}

L2Snapshot OrderBook::l2_snapshot(int64_t max_levels) const {
    int64_t n_bid = std::min(max_levels, static_cast<int64_t>(bids_.size()));
    int64_t n_ask = std::min(max_levels, static_cast<int64_t>(asks_.size()));

    std::vector<PriceLevel> top_bids(
        bids_.begin(), bids_.begin() + static_cast<ptrdiff_t>(n_bid));
    std::vector<PriceLevel> top_asks(
        asks_.begin(), asks_.begin() + static_cast<ptrdiff_t>(n_ask));

    return L2Snapshot(std::move(top_bids), std::move(top_asks),
                      static_cast<double>(now_ns()));
}

BookDepth OrderBook::depth_summary(int64_t max_levels) const {
    double bid_vol = 0.0, ask_vol = 0.0;
    int64_t n_bid = std::min(max_levels, static_cast<int64_t>(bids_.size()));
    int64_t n_ask = std::min(max_levels, static_cast<int64_t>(asks_.size()));

    for (int64_t i = 0; i < n_bid; ++i) {
        bid_vol += bids_[static_cast<size_t>(i)].size;
    }
    for (int64_t i = 0; i < n_ask; ++i) {
        ask_vol += asks_[static_cast<size_t>(i)].size;
    }

    return BookDepth(bid_vol, ask_vol, n_bid, n_ask);
}

} // namespace hft
