#include "microstructure.h"
#include "order_book.h"
#include <cmath>
#include <algorithm>
// std::accumulate is used below; do not rely on microstructure.h including it.
#include <numeric>

namespace hft {

MicrostructureAnalyser::MicrostructureAnalyser(int64_t window_size)
    : window_size_(window_size) {}

void MicrostructureAnalyser::on_tick(const std::string& message_type,
                                     double bid_price, double ask_price,
                                     double bid_size, double ask_size) {
    if (message_type != "L1") return;
    if (bid_price <= 0.0 || ask_price <= 0.0) return;

    double spread = ask_price - bid_price;
    double mid = (bid_price + ask_price) / 2.0;
    double relative_spread = (mid > 0.0) ? spread / mid : 0.0;

    record_spread(spread, relative_spread);
}

void MicrostructureAnalyser::record_spread(double spread, double relative_spread) {
    spreads_.push_back(spread);
    relative_spreads_.push_back(relative_spread);

    while (static_cast<int64_t>(spreads_.size()) > window_size_) {
        spreads_.pop_front();
        relative_spreads_.pop_front();
    }
}

SpreadMetrics MicrostructureAnalyser::spread_stats() const {
    SpreadMetrics metrics;
    metrics.count = static_cast<int64_t>(spreads_.size());
    if (metrics.count == 0) return metrics;

    double sum = std::accumulate(spreads_.begin(), spreads_.end(), 0.0);
    metrics.mean = sum / static_cast<double>(metrics.count);

    metrics.min = *std::min_element(spreads_.begin(), spreads_.end());
    metrics.max = *std::max_element(spreads_.begin(), spreads_.end());

    if (metrics.count > 1) {
        double sq_sum = 0.0;
        for (double s : spreads_) {
            double diff = s - metrics.mean;
            sq_sum += diff * diff;
        }
        metrics.std = std::sqrt(sq_sum / static_cast<double>(metrics.count - 1));
    }

    double rel_sum = std::accumulate(relative_spreads_.begin(), relative_spreads_.end(), 0.0);
    metrics.relative = rel_sum / static_cast<double>(metrics.count);

    return metrics;
}

DepthProfile MicrostructureAnalyser::current_depth(const OrderBook& book, int64_t max_levels) {
    auto snapshot = book.l2_snapshot(max_levels);
    double bid_vol = 0.0, ask_vol = 0.0;
    double weighted_bid_sum = 0.0, weighted_ask_sum = 0.0;

    for (const auto& level : snapshot.bids) {
        bid_vol += level.size;
        weighted_bid_sum += level.price * level.size;
    }
    for (const auto& level : snapshot.asks) {
        ask_vol += level.size;
        weighted_ask_sum += level.price * level.size;
    }

    double total = bid_vol + ask_vol;
    double imbalance = (total > 0.0) ? (bid_vol - ask_vol) / total : 0.0;
    double weighted_mid = (bid_vol > 0.0 && ask_vol > 0.0)
        ? ((weighted_bid_sum / bid_vol) + (weighted_ask_sum / ask_vol)) / 2.0
        : 0.0;

    return DepthProfile(snapshot.bids, snapshot.asks,
                        bid_vol, ask_vol, imbalance, weighted_mid);
}

TradeSignResult MicrostructureAnalyser::classify_trade(
    double trade_price, double trade_size, const OrderBook& book) {
    L1Snapshot l1 = [&]() {
        try { return book.l1_snapshot(); }
        catch (...) { return L1Snapshot(0.0, 0.0, 0.0, 0.0); }
    }();
    if (l1.bid_price <= 0.0 && l1.ask_price <= 0.0) {
        return TradeSignResult("neutral", trade_price, trade_size, 0.0);
    }

    double bid = l1.bid_price;
    double ask = l1.ask_price;
    double mid = l1.mid_price;

    if (trade_price >= ask) {
        return TradeSignResult("buy", trade_price, trade_size, 1.0);
    }
    if (trade_price <= bid) {
        return TradeSignResult("sell", trade_price, trade_size, 1.0);
    }

    // Trade at the ask -> buy
    if (std::abs(trade_price - ask) < 1e-10) {
        return TradeSignResult("buy", trade_price, trade_size, 1.0);
    }
    // Trade at the bid -> sell
    if (std::abs(trade_price - bid) < 1e-10) {
        return TradeSignResult("sell", trade_price, trade_size, 1.0);
    }

    // Inside spread — classify by mid-price
    if (trade_price > mid) {
        return TradeSignResult("buy", trade_price, trade_size, 0.5);
    } else {
        return TradeSignResult("sell", trade_price, trade_size, 0.5);
    }
}

} // namespace hft
