#pragma once

#include <string>
#include <unordered_map>
#include <functional>
#include <vector>
#include <chrono>
#include <cstdint>

namespace hft {

struct MarketDataMessage {
    std::string epic;
    std::string message_type;  // "L1", "L2", "TRADE"
    double bid_price{0.0};
    double ask_price{0.0};
    double bid_size{0.0};
    double ask_size{0.0};
    double level_price{0.0};
    double last_price{0.0};
    double last_size{0.0};
    int64_t timestamp_ns{0};
    std::string raw;

    MarketDataMessage() = default;

    MarketDataMessage(std::string ep, std::string mt)
        : epic(std::move(ep)), message_type(std::move(mt)) {}
};

using TickCallback = std::function<void(const MarketDataMessage&)>;

inline double safe_float(const std::string& value, double default_val = 0.0) {
    if (value.empty()) return default_val;
    try {
        return std::stod(value);
    } catch (...) {
        return default_val;
    }
}

inline int64_t safe_int(const std::string& value, int64_t default_val = 0) {
    if (value.empty()) return default_val;
    try {
        return static_cast<int64_t>(std::stoll(value));
    } catch (...) {
        return default_val;
    }
}

inline int64_t now_ns() {
    auto now = std::chrono::system_clock::now();
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        now.time_since_epoch()).count();
    return ns;
}

MarketDataMessage parse_l1_update(const std::string& epic,
                                    const std::unordered_map<std::string, std::string>& fields);

MarketDataMessage parse_l2_update(const std::string& epic,
                                    const std::unordered_map<std::string, std::string>& fields);

MarketDataMessage parse_trade_update(const std::string& epic,
                                       const std::unordered_map<std::string, std::string>& fields);

class TickConsumer {
public:
    TickConsumer() = default;

    void subscribe(const std::string& epic, TickCallback callback);

    void unsubscribe(const std::string& epic, TickCallback callback);

    void dispatch(const MarketDataMessage& msg);

    void connect(uint64_t api_key, const std::string& account_id) { connected_ = true; }
    void disconnect() { connected_ = false; }
    bool is_connected() const { return connected_; }

private:
    std::unordered_map<std::string, std::vector<TickCallback>> callbacks_;
    bool connected_{false};
};

} // namespace hft
