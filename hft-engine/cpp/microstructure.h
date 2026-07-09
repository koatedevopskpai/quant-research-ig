#pragma once

#include <vector>
#include <string>
#include <deque>
#include <cstdint>
#include <numeric>
#include <cmath>

namespace hft {

// Forward declarations from order_book.h
struct PriceLevel;
struct L1Snapshot;
class OrderBook;

struct SpreadMetrics {
    double mean;
    double std;
    double min;
    double max;
    double relative;
    int64_t count;

    SpreadMetrics() : mean(0.0), std(0.0), min(0.0), max(0.0),
                      relative(0.0), count(0) {}
};

struct DepthProfile {
    std::vector<PriceLevel> bid_levels;
    std::vector<PriceLevel> ask_levels;
    double total_bid_volume;
    double total_ask_volume;
    double imbalance;
    double weighted_mid;

    DepthProfile(std::vector<PriceLevel> bids, std::vector<PriceLevel> asks,
                 double tbv, double tav, double imb, double wm)
        : bid_levels(std::move(bids)), ask_levels(std::move(asks)),
          total_bid_volume(tbv), total_ask_volume(tav),
          imbalance(imb), weighted_mid(wm) {}
};

struct TradeSignResult {
    std::string side;  // "buy", "sell", "neutral"
    double price;
    double size;
    double confidence;

    TradeSignResult(std::string s, double p, double sz, double conf)
        : side(std::move(s)), price(p), size(sz), confidence(conf) {}
};

class MicrostructureAnalyser {
public:
    explicit MicrostructureAnalyser(int64_t window_size = 100);

    void on_tick(const std::string& message_type,
                 double bid_price, double ask_price,
                 double bid_size, double ask_size);

    SpreadMetrics spread_stats() const;

    static DepthProfile current_depth(const OrderBook& book, int64_t max_levels = 10);

    static TradeSignResult classify_trade(double trade_price, double trade_size,
                                           const OrderBook& book);

    int64_t window_size() const { return window_size_; }
    int64_t spread_count() const { return static_cast<int64_t>(spreads_.size()); }

private:
    int64_t window_size_;
    std::deque<double> spreads_;
    std::deque<double> relative_spreads_;

    void record_spread(double spread, double relative_spread);
};

} // namespace hft
