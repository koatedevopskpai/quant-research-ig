#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "order_book.h"
#include "feed_handler.h"
#include "microstructure.h"
#include "spsc_ring.h"
#include "concurrent_book.h"

namespace py = pybind11;
namespace hft {

// A fixed-size, trivially copyable message for the lock-free handoff. Strings
// and parsing stay outside the queue: allocation inside a "lock-free" path
// would undermine the claim.
struct RingTick {
    std::int64_t ts_ns{0};
    double bid{0.0};
    double ask{0.0};
    double size{0.0};
};

using TickRing = SpscRing<RingTick, 4096>;

PYBIND11_MODULE(hft_cpp, m) {
    m.doc() = "HFT Engine C++ hot-path accelerators";

    // ── PriceLevel ───────────────────────────────────────────────────────
    py::class_<PriceLevel>(m, "PriceLevel")
        .def(py::init<double, double, int64_t>(),
             py::arg("price"), py::arg("size"), py::arg("order_count"))
        .def_readonly("price", &PriceLevel::price)
        .def_readonly("size", &PriceLevel::size)
        .def_readonly("order_count", &PriceLevel::order_count);

    // ── L1Snapshot ───────────────────────────────────────────────────────
    py::class_<L1Snapshot>(m, "L1Snapshot")
        .def(py::init<double, double, double, double>(),
             py::arg("bid_price"), py::arg("bid_size"),
             py::arg("ask_price"), py::arg("ask_size"))
        .def_readonly("bid_price", &L1Snapshot::bid_price)
        .def_readonly("bid_size", &L1Snapshot::bid_size)
        .def_readonly("ask_price", &L1Snapshot::ask_price)
        .def_readonly("ask_size", &L1Snapshot::ask_size)
        .def_readonly("spread", &L1Snapshot::spread)
        .def_readonly("mid_price", &L1Snapshot::mid_price);

    // ── L2Snapshot ───────────────────────────────────────────────────────
    py::class_<L2Snapshot>(m, "L2Snapshot")
        .def(py::init<std::vector<PriceLevel>, std::vector<PriceLevel>, double>(),
             py::arg("bids"), py::arg("asks"), py::arg("timestamp"))
        .def_readonly("bids", &L2Snapshot::bids)
        .def_readonly("asks", &L2Snapshot::asks)
        .def_readonly("timestamp", &L2Snapshot::timestamp);

    // ── BookDepth ────────────────────────────────────────────────────────
    py::class_<BookDepth>(m, "BookDepth")
        .def(py::init<double, double, int64_t, int64_t>(),
             py::arg("bid_volume"), py::arg("ask_volume"),
             py::arg("num_bid_levels"), py::arg("num_ask_levels"))
        .def_readonly("bid_volume", &BookDepth::bid_volume)
        .def_readonly("ask_volume", &BookDepth::ask_volume)
        .def_readonly("imbalance", &BookDepth::imbalance)
        .def_readonly("num_bid_levels", &BookDepth::num_bid_levels)
        .def_readonly("num_ask_levels", &BookDepth::num_ask_levels);

    // ── OrderBook ────────────────────────────────────────────────────────
    py::class_<OrderBook>(m, "OrderBook")
        .def(py::init<>())
        .def("apply_diff", &OrderBook::apply_diff,
             py::arg("side"), py::arg("price"),
             py::arg("size"), py::arg("order_count"))
        .def("apply_snapshot", &OrderBook::apply_snapshot,
             py::arg("bids"), py::arg("asks"))
        .def("l1_snapshot", &OrderBook::l1_snapshot)
        .def("l2_snapshot", &OrderBook::l2_snapshot, py::arg("max_levels"))
        .def("depth_summary", &OrderBook::depth_summary,
             py::arg("max_levels") = 100)
        .def("bid_count", &OrderBook::bid_count)
        .def("ask_count", &OrderBook::ask_count);

    // ── SpreadMetrics ────────────────────────────────────────────────────
    py::class_<SpreadMetrics>(m, "SpreadMetrics")
        .def(py::init<>())
        .def_readonly("mean", &SpreadMetrics::mean)
        .def_readonly("std", &SpreadMetrics::std)
        .def_readonly("min", &SpreadMetrics::min)
        .def_readonly("max", &SpreadMetrics::max)
        .def_readonly("relative", &SpreadMetrics::relative)
        .def_readonly("count", &SpreadMetrics::count);

    // ── DepthProfile ─────────────────────────────────────────────────────
    py::class_<DepthProfile>(m, "DepthProfile")
        .def(py::init<std::vector<PriceLevel>, std::vector<PriceLevel>,
                      double, double, double, double>(),
             py::arg("bid_levels"), py::arg("ask_levels"),
             py::arg("total_bid_volume"), py::arg("total_ask_volume"),
             py::arg("imbalance"), py::arg("weighted_mid"))
        .def_readonly("bid_levels", &DepthProfile::bid_levels)
        .def_readonly("ask_levels", &DepthProfile::ask_levels)
        .def_readonly("total_bid_volume", &DepthProfile::total_bid_volume)
        .def_readonly("total_ask_volume", &DepthProfile::total_ask_volume)
        .def_readonly("imbalance", &DepthProfile::imbalance)
        .def_readonly("weighted_mid", &DepthProfile::weighted_mid);

    // ── TradeSignResult ──────────────────────────────────────────────────
    py::class_<TradeSignResult>(m, "TradeSignResult")
        .def(py::init<std::string, double, double, double>(),
             py::arg("side"), py::arg("price"),
             py::arg("size"), py::arg("confidence"))
        .def_readonly("side", &TradeSignResult::side)
        .def_readonly("price", &TradeSignResult::price)
        .def_readonly("size", &TradeSignResult::size)
        .def_readonly("confidence", &TradeSignResult::confidence);

    // ── MicrostructureAnalyser ───────────────────────────────────────────
    py::class_<MicrostructureAnalyser>(m, "MicrostructureAnalyser")
        .def(py::init<int64_t>(), py::arg("window_size") = 100)
        .def("on_tick", &MicrostructureAnalyser::on_tick,
             py::arg("message_type"), py::arg("bid_price"),
             py::arg("ask_price"), py::arg("bid_size"), py::arg("ask_size"))
        .def("spread_stats", &MicrostructureAnalyser::spread_stats)
        .def_static("current_depth", &MicrostructureAnalyser::current_depth,
                     py::arg("book"), py::arg("max_levels") = 10)
        .def_static("classify_trade", &MicrostructureAnalyser::classify_trade,
                     py::arg("trade_price"), py::arg("trade_size"),
                     py::arg("book"))
        .def("window_size", &MicrostructureAnalyser::window_size)
        .def("spread_count", &MicrostructureAnalyser::spread_count);

    // ── Free functions (feed handler) ────────────────────────────────────
    m.def("parse_l1_update", &parse_l1_update,
          py::arg("epic"), py::arg("fields"));
    m.def("parse_l2_update", &parse_l2_update,
          py::arg("epic"), py::arg("fields"));
    m.def("parse_trade_update", &parse_trade_update,
          py::arg("epic"), py::arg("fields"));
    m.def("safe_float", &safe_float,
          py::arg("value"), py::arg("default_val") = 0.0);
    m.def("safe_int", &safe_int,
          py::arg("value"), py::arg("default_val") = 0);
    m.def("now_ns", &now_ns);

    // ── MarketDataMessage ────────────────────────────────────────────────
    py::class_<MarketDataMessage>(m, "MarketDataMessage")
        .def(py::init<>())
        .def(py::init<std::string, std::string>(),
             py::arg("epic"), py::arg("message_type"))
        .def_readwrite("epic", &MarketDataMessage::epic)
        .def_readwrite("message_type", &MarketDataMessage::message_type)
        .def_readwrite("bid_price", &MarketDataMessage::bid_price)
        .def_readwrite("ask_price", &MarketDataMessage::ask_price)
        .def_readwrite("bid_size", &MarketDataMessage::bid_size)
        .def_readwrite("ask_size", &MarketDataMessage::ask_size)
        .def_readwrite("level_price", &MarketDataMessage::level_price)
        .def_readwrite("last_price", &MarketDataMessage::last_price)
        .def_readwrite("last_size", &MarketDataMessage::last_size)
        .def_readwrite("timestamp_ns", &MarketDataMessage::timestamp_ns)
        .def_readwrite("raw", &MarketDataMessage::raw);

    // ── Lock-free handoff and concurrent snapshots ─────────────────────────
    py::class_<ConcurrentOrderBook>(m, "ConcurrentOrderBook")
        .def(py::init<>())
        .def("apply_l1", &ConcurrentOrderBook::apply_l1,
             py::arg("bid"), py::arg("ask"),
             py::arg("bid_size") = 0.0, py::arg("ask_size") = 0.0,
             py::call_guard<py::gil_scoped_release>())
        .def("apply_diff", &ConcurrentOrderBook::apply_diff,
             py::arg("side"), py::arg("price"), py::arg("size"),
             py::arg("order_count") = 0,
             py::call_guard<py::gil_scoped_release>())
        .def("snapshot_l1", [](const ConcurrentOrderBook& book) {
            L1Snapshot snapshot(0.0, 0.0, 0.0, 0.0);
            if (!book.snapshot_l1(snapshot)) {
                throw std::runtime_error("no consistent L1 snapshot available");
            }
            return snapshot;
        }, py::call_guard<py::gil_scoped_release>())
        .def("snapshot_l2", [](const ConcurrentOrderBook& book,
                               std::int64_t max_levels) {
            L2Snapshot snapshot(std::vector<PriceLevel>{},
                                std::vector<PriceLevel>{}, 0.0);
            if (!book.snapshot_l2(max_levels, snapshot)) {
                throw std::runtime_error("no consistent L2 snapshot available");
            }
            return snapshot;
        }, py::arg("max_levels") = 0, py::call_guard<py::gil_scoped_release>())
        .def("updates", &ConcurrentOrderBook::updates)
        .def("snapshots", &ConcurrentOrderBook::snapshots)
        .def("retries", &ConcurrentOrderBook::retries);

    py::class_<TickRing>(m, "TickRing")
        .def(py::init<>())
        .def("try_push", [](TickRing& ring, std::int64_t ts_ns, double bid,
                            double ask, double size) {
            return ring.try_push(RingTick{ts_ns, bid, ask, size});
        }, py::arg("ts_ns"), py::arg("bid"), py::arg("ask"), py::arg("size"),
           py::call_guard<py::gil_scoped_release>())
        .def("try_pop", [](TickRing& ring) -> py::object {
            RingTick tick;
            if (!ring.try_pop(tick)) {
                return py::none();
            }
            return py::make_tuple(tick.ts_ns, tick.bid, tick.ask, tick.size);
        })
        .def("size", &TickRing::size)
        .def_static("capacity", []() { return TickRing::capacity(); });
}

} // namespace hft
