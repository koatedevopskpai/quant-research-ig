"""
test_tick_consumer.py — Unit tests for the Lightstreamer tick consumer.
"""

from math import isclose

import pytest

from hft_engine.feeds.tick_consumer import TickConsumer, MarketDataMessage


class TestTickConsumer:
    """Test feed parsing and callback dispatch."""

    # ------------------------------------------------------------------
    # L1 parsing
    # ------------------------------------------------------------------

    def test_parse_l1_basic(self) -> None:
        consumer = TickConsumer()
        fields = {
            "Bid": "100.50",
            "Offer": "100.55",
            "BidSize": "5000",
            "OfferSize": "3000",
            "Last": "100.52",
        }
        msg = consumer.parse_l1_update("IX.D.FTSE.IFP.IP", fields)
        assert msg.epic == "IX.D.FTSE.IFP.IP"
        assert msg.message_type == "L1"
        assert isclose(msg.bid_price, 100.50)
        assert isclose(msg.ask_price, 100.55)
        assert isclose(msg.bid_size, 5000)
        assert isclose(msg.ask_size, 3000)
        assert isclose(msg.last_price, 100.52)
        assert msg.timestamp_ns > 0

    def test_parse_l1_missing_fields(self) -> None:
        """Missing or null fields should default to 0.0."""
        consumer = TickConsumer()
        fields = {"Bid": "null", "Offer": "100.55"}
        msg = consumer.parse_l1_update("IX.D.FTSE.IFP.IP", fields)
        assert msg.bid_price == 0.0
        assert isclose(msg.ask_price, 100.55)
        assert msg.bid_size == 0.0
        assert msg.ask_size == 0.0

    # ------------------------------------------------------------------
    # L2 parsing
    # ------------------------------------------------------------------

    def test_parse_l2_bid_level(self) -> None:
        consumer = TickConsumer()
        fields = {
            "MarketLevel": "3",
            "Bid": "100.48",
            "Offer": "",
            "BidSize": "10000",
            "OfferSize": "",
        }
        msg = consumer.parse_l2_update("IX.D.FTSE.IFP.IP", fields)
        assert msg.message_type == "L2"
        assert isclose(msg.level_price, 100.48)
        assert isclose(msg.bid_price, 100.48)
        assert isclose(msg.bid_size, 10000)

    def test_parse_l2_ask_level(self) -> None:
        consumer = TickConsumer()
        fields = {
            "MarketLevel": "2",
            "Bid": "",
            "Offer": "100.53",
            "BidSize": "",
            "OfferSize": "8000",
        }
        msg = consumer.parse_l2_update("IX.D.FTSE.IFP.IP", fields)
        assert msg.message_type == "L2"
        assert isclose(msg.level_price, 100.53)
        assert isclose(msg.ask_price, 100.53)
        assert isclose(msg.ask_size, 8000)

    # ------------------------------------------------------------------
    # Trade parsing
    # ------------------------------------------------------------------

    def test_parse_trade(self) -> None:
        consumer = TickConsumer()
        fields = {"Last": "100.52", "LastSize": "1000"}
        msg = consumer.parse_trade_update("IX.D.FTSE.IFP.IP", fields)
        assert msg.message_type == "TRADE"
        assert isclose(msg.last_price, 100.52)
        assert isclose(msg.last_size, 1000)

    # ------------------------------------------------------------------
    # Callback dispatch
    # ------------------------------------------------------------------

    def test_dispatch(self) -> None:
        received: list[MarketDataMessage] = []
        consumer = TickConsumer()

        def callback(msg: MarketDataMessage) -> None:
            received.append(msg)

        consumer.subscribe("IX.D.FTSE.IFP.IP", callback)
        msg = MarketDataMessage(
            epic="IX.D.FTSE.IFP.IP",
            message_type="L1",
            bid_price=100.50,
            ask_price=100.55,
        )
        consumer.dispatch(msg)
        assert len(received) == 1
        assert received[0].bid_price == 100.50

    def test_dispatch_multiple_callbacks(self) -> None:
        results: list[int] = []
        consumer = TickConsumer()

        def cb1(msg: MarketDataMessage) -> None:
            results.append(1)

        def cb2(msg: MarketDataMessage) -> None:
            results.append(2)

        consumer.subscribe("EPIC", cb1)
        consumer.subscribe("EPIC", cb2)
        msg = MarketDataMessage(epic="EPIC", message_type="L1")
        consumer.dispatch(msg)
        assert results == [1, 2]

    def test_dispatch_wrong_epic(self) -> None:
        received: list[MarketDataMessage] = []
        consumer = TickConsumer()

        def callback(msg: MarketDataMessage) -> None:
            received.append(msg)

        consumer.subscribe("EPIC_A", callback)
        msg = MarketDataMessage(epic="EPIC_B", message_type="L1")
        consumer.dispatch(msg)
        assert len(received) == 0

    def test_unsubscribe_all(self) -> None:
        received: list[MarketDataMessage] = []
        consumer = TickConsumer()

        def callback(msg: MarketDataMessage) -> None:
            received.append(msg)

        consumer.subscribe("EPIC", callback)
        consumer.unsubscribe("EPIC")
        msg = MarketDataMessage(epic="EPIC", message_type="L1")
        consumer.dispatch(msg)
        assert len(received) == 0

    # ------------------------------------------------------------------
    # Safe conversions
    # ------------------------------------------------------------------

    def test_safe_float_invalid(self) -> None:
        assert TickConsumer._safe_float("null") == 0.0
        assert TickConsumer._safe_float("") == 0.0
        assert TickConsumer._safe_float("abc") == 0.0

    def test_safe_float_valid(self) -> None:
        assert TickConsumer._safe_float("100.50") == 100.50
        assert TickConsumer._safe_float("0") == 0.0

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def test_connect_sets_connected(self, monkeypatch) -> None:
        """connect() should set _connected even when IGStreamer unavailable."""
        from types import ModuleType
        import sys

        stub = ModuleType("ig_market_data.stream")
        def unavailable():
            raise RuntimeError("offline fixture")
        stub.IGStreamer = unavailable
        monkeypatch.setitem(sys.modules, "ig_market_data.stream", stub)
        consumer = TickConsumer()
        consumer.subscribe("IX.D.FTSE.IFP.IP", lambda m: None)
        consumer.connect()
        assert consumer._connected is True

    def test_disconnect_cleans_up(self, monkeypatch) -> None:
        from types import ModuleType
        import sys

        stub = ModuleType("ig_market_data.stream")
        def unavailable():
            raise RuntimeError("offline fixture")
        stub.IGStreamer = unavailable
        monkeypatch.setitem(sys.modules, "ig_market_data.stream", stub)
        consumer = TickConsumer()
        consumer.subscribe("EPIC", lambda m: None)
        consumer.connect()
        consumer.disconnect()
        assert consumer._connected is False

    def test_connect_market_simulator(self) -> None:
        consumer = TickConsumer()
        consumer.subscribe("EPIC", lambda m: None)
        consumer.connect_market_simulator("sim")
        assert consumer._connected is True
        assert consumer._simulator == "sim"
        consumer.disconnect()
        assert consumer._connected is False

    # ------------------------------------------------------------------
    # IGStreamer bridge callbacks
    # ------------------------------------------------------------------

    def test_on_l1_update_parses_and_dispatches(self) -> None:
        received: list[MarketDataMessage] = []
        consumer = TickConsumer()

        def callback(msg: MarketDataMessage) -> None:
            received.append(msg)

        consumer.subscribe("IX.D.FTSE.IFP.IP", callback)
        fields = {
            "Bid": "100.50",
            "Offer": "100.55",
            "BidSize": "5000",
            "OfferSize": "3000",
            "Last": "100.52",
        }
        consumer._on_l1_update("MARKET:IX.D.FTSE.IFP.IP", fields)
        assert len(received) == 1
        msg = received[0]
        assert msg.epic == "IX.D.FTSE.IFP.IP"
        assert msg.message_type == "L1"
        assert isclose(msg.bid_price, 100.50)
        assert isclose(msg.ask_price, 100.55)

    def test_on_l1_update_handles_uppercase_fields(self) -> None:
        """Should normalise BID/OFFER -> Bid/Offer for parse_l1_update."""
        received: list[MarketDataMessage] = []
        consumer = TickConsumer()

        def callback(msg: MarketDataMessage) -> None:
            received.append(msg)

        consumer.subscribe("EPIC", callback)
        fields = {"BID": "99.00", "OFFER": "99.05"}
        consumer._on_l1_update("MARKET:EPIC", fields)
        assert len(received) == 1
        assert isclose(received[0].bid_price, 99.00)
        assert isclose(received[0].ask_price, 99.05)

    def test_on_tick_update_parses_and_dispatches(self) -> None:
        received: list[MarketDataMessage] = []
        consumer = TickConsumer()

        def callback(msg: MarketDataMessage) -> None:
            received.append(msg)

        consumer.subscribe("CS.D.EURUSD.CFD.IP", callback)
        fields = {
            "Ltp": "1.1234",
            "Ltv": "1000000",
            "Bid": "1.1233",
            "Ofr": "1.1235",
        }
        consumer._on_tick_update(
            "CHART:CS.D.EURUSD.CFD.IP:TICK", fields,
        )
        assert len(received) == 1
        msg = received[0]
        assert msg.epic == "CS.D.EURUSD.CFD.IP"
        assert msg.message_type == "TRADE"
        assert isclose(msg.last_price, 1.1234)
        assert isclose(msg.last_size, 1000000)

    def test_on_tick_update_empty_fields(self) -> None:
        """Missing tick fields should default to 0.0."""
        received: list[MarketDataMessage] = []
        consumer = TickConsumer()

        def callback(msg: MarketDataMessage) -> None:
            received.append(msg)

        consumer.subscribe("EPIC", callback)
        consumer._on_tick_update("CHART:EPIC:TICK", {})
        assert len(received) == 1
        assert received[0].last_price == 0.0
        assert received[0].last_size == 0.0

    def test_on_l1_update_unknown_epic_dispatches_nothing(self) -> None:
        received: list[MarketDataMessage] = []
        consumer = TickConsumer()

        def callback(msg: MarketDataMessage) -> None:
            received.append(msg)

        consumer.subscribe("EPIC_A", callback)
        consumer._on_l1_update("MARKET:EPIC_B", {"Bid": "100"})
        assert len(received) == 0
