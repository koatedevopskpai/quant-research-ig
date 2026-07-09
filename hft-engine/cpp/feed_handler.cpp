#include "feed_handler.h"

namespace hft {

MarketDataMessage parse_l1_update(
    const std::string& epic,
    const std::unordered_map<std::string, std::string>& fields) {
    MarketDataMessage msg(epic, "L1");
    msg.timestamp_ns = now_ns();

    auto it = fields.find("BID");
    if (it != fields.end()) msg.bid_price = safe_float(it->second);
    it = fields.find("OFFER");
    if (it != fields.end()) msg.ask_price = safe_float(it->second);
    it = fields.find("BID_SIZE");
    if (it != fields.end()) msg.bid_size = safe_float(it->second);
    it = fields.find("OFFER_SIZE");
    if (it != fields.end()) msg.ask_size = safe_float(it->second);

    return msg;
}

MarketDataMessage parse_l2_update(
    const std::string& epic,
    const std::unordered_map<std::string, std::string>& fields) {
    MarketDataMessage msg(epic, "L2");
    msg.timestamp_ns = now_ns();

    auto it = fields.find("PRICE");
    if (it != fields.end()) msg.level_price = safe_float(it->second);
    it = fields.find("BID");
    if (it != fields.end()) msg.bid_price = safe_float(it->second);
    it = fields.find("OFFER");
    if (it != fields.end()) msg.ask_price = safe_float(it->second);
    it = fields.find("BID_SIZE");
    if (it != fields.end()) msg.bid_size = safe_float(it->second);
    it = fields.find("OFFER_SIZE");
    if (it != fields.end()) msg.ask_size = safe_float(it->second);

    return msg;
}

MarketDataMessage parse_trade_update(
    const std::string& epic,
    const std::unordered_map<std::string, std::string>& fields) {
    MarketDataMessage msg(epic, "TRADE");
    msg.timestamp_ns = now_ns();

    auto it = fields.find("PRICE");
    if (it != fields.end()) msg.last_price = safe_float(it->second);
    it = fields.find("SIZE");
    if (it != fields.end()) msg.last_size = safe_float(it->second);
    it = fields.find("BID");
    if (it != fields.end()) msg.bid_price = safe_float(it->second);
    it = fields.find("OFFER");
    if (it != fields.end()) msg.ask_price = safe_float(it->second);

    return msg;
}

void TickConsumer::subscribe(const std::string& epic, TickCallback callback) {
    callbacks_[epic].push_back(std::move(callback));
}

void TickConsumer::unsubscribe(const std::string& epic, TickCallback callback) {
    auto it = callbacks_.find(epic);
    if (it == callbacks_.end()) return;

    auto& cbs = it->second;
    cbs.erase(std::remove_if(cbs.begin(), cbs.end(),
        [&](const TickCallback& cb) {
            return cb.target_type() == callback.target_type();
        }), cbs.end());

    if (cbs.empty()) {
        callbacks_.erase(it);
    }
}

void TickConsumer::dispatch(const MarketDataMessage& msg) {
    auto it = callbacks_.find(msg.epic);
    if (it == callbacks_.end()) return;

    for (const auto& cb : it->second) {
        cb(msg);
    }
}

} // namespace hft
