"""
Tests for the provider layer. All offline: no sockets, no accounts, no vendor data.

What is pinned here:
  * replay ordering, per-instrument isolation and JSONL round-trip
  * the IG field-mapping defects found in the 2026-10 review and confirmed live
  * Dukascopy bi5 decoding (LZMA + big-endian records + point scaling)
  * Binance L2 book maintenance: stale/gap handling and level removal
  * the tick-consumer casing bug: {"LTP": ...} must not parse to zero
"""

import lzma
import struct
from datetime import datetime, timezone

import pytest

from ig_market_data.providers.base import Capabilities, MarketEvent
from ig_market_data.providers.binance import BinanceProvider, _Book
from ig_market_data.providers.dukascopy import DukascopyTicks
from ig_market_data.providers.ig import IGProvider
from ig_market_data.providers.replay import ReplayProvider

TS = 1_761_486_000_000_000_000  # a fixed nanosecond timestamp


def _ev(**kw):
    base = dict(epic="EURUSD", kind="trade", provider_ts_ns=TS, receive_ts_ns=TS, last=1.0)
    base.update(kw)
    return MarketEvent(**base)


# ===========================================================================
# Event contract
# ===========================================================================

class TestEventContract:
    def test_l2_requires_level_information(self):
        with pytest.raises(ValueError, match="carry prices"):
            _ev(kind="l2", book_action="diff", last=None)

    def test_trade_rejects_book_action(self):
        with pytest.raises(ValueError, match="must not set book_action"):
            _ev(book_action="diff")

    def test_trade_requires_price(self):
        with pytest.raises(ValueError, match="require last price"):
            _ev(last=None)

    def test_timestamps_must_be_positive(self):
        with pytest.raises(ValueError, match="positive"):
            _ev(provider_ts_ns=0)

    def test_non_finite_prices_rejected(self):
        with pytest.raises(ValueError, match="finite"):
            _ev(last=float("nan"))

    def test_l1_requires_replace_or_diff(self):
        with pytest.raises(ValueError, match="book_action"):
            _ev(kind="l1", last=None, bid=1.0)


# ===========================================================================
# Replay
# ===========================================================================

class TestReplay:
    def test_orders_by_provider_time_regardless_of_input_order(self):
        late = _ev(provider_ts_ns=TS + 2, receive_ts_ns=TS + 2)
        early = _ev(provider_ts_ns=TS, receive_ts_ns=TS)
        provider = ReplayProvider([late, early])
        assert [e.provider_ts_ns for e in provider.events()] == [TS, TS + 2]

    def test_instruments_are_isolated(self):
        a = _ev(epic="EURUSD")
        b = _ev(epic="US500")
        provider = ReplayProvider([a, b])
        seen = []
        provider.connect()
        provider.subscribe("EURUSD", "trade", seen.append)
        provider.replay()
        assert [e.epic for e in seen] == ["EURUSD"]

    def test_kind_filter_is_respected(self):
        provider = ReplayProvider([_ev(kind="trade"), _ev(kind="l1", bid=1.0, book_action="replace", last=None)])
        seen = []
        provider.connect()
        provider.subscribe("EURUSD", "l1", seen.append)
        provider.replay()
        assert len(seen) == 1 and seen[0].kind == "l1"

    def test_jsonl_round_trip(self, tmp_path):
        events = [_ev(epic="EURUSD"), _ev(epic="US500", last=7644.5)]
        path = ReplayProvider(events).to_jsonl(tmp_path / "fixture.jsonl")
        reloaded = ReplayProvider.from_jsonl(path)
        assert [e.epic for e in reloaded.events()] == ["EURUSD", "US500"]

    def test_replay_requires_connect(self):
        with pytest.raises(RuntimeError, match="connect"):
            ReplayProvider([_ev()]).replay()

    def test_capabilities_are_explicit(self):
        caps = ReplayProvider([]).capabilities
        assert isinstance(caps, Capabilities)
        assert caps.live is False and caps.historical is True


# ===========================================================================
# IG mapping — the defects confirmed live on 2026-10-26
# ===========================================================================

class _FakeStreamer:
    def __init__(self):
        self.l1 = {}
        self.ticks = {}

    def connect(self):
        pass

    def disconnect(self):
        pass

    def subscribe_l1(self, epic, cb):
        self.l1[epic] = cb

    def subscribe_ticks(self, epic, cb):
        self.ticks[epic] = cb


class TestIGMapping:
    def _provider(self):
        streamer = _FakeStreamer()
        provider = IGProvider(streamer=streamer)
        provider.connect()
        return provider, streamer

    def test_l1_is_a_replacement_not_a_diff(self):
        provider, streamer = self._provider()
        seen = []
        provider.subscribe("CS.D.EURUSD.CFD.IP", "l1", seen.append)
        streamer.l1["CS.D.EURUSD.CFD.IP"]("MARKET:CS.D.EURUSD.CFD.IP",
                                          {"BID": "1.11937", "OFR": "1.11943"})
        assert seen[0].book_action == "replace"
        assert seen[0].bid == pytest.approx(1.11937)
        assert seen[0].ask == pytest.approx(1.11943)

    def test_l1_without_provider_timestamp_says_so(self):
        provider, streamer = self._provider()
        seen = []
        provider.subscribe("EURUSD", "l1", seen.append)
        streamer.l1["EURUSD"]("MARKET:EURUSD", {"BID": "1.0", "OFR": "1.1"})
        assert seen[0].raw["provider_ts_source"] == "receive"

    def test_uppercase_tick_fields_do_not_parse_to_zero(self):
        """The live repro: {'LTP': '105', 'LTV': '12'} previously gave 0 and 0."""
        provider, streamer = self._provider()
        seen = []
        provider.subscribe("US500", "trade", seen.append)
        streamer.ticks["US500"]("CHART:US500:TICK", {"LTP": "105", "LTV": "12", "UTM": "1761486000123"})
        assert seen[0].last == pytest.approx(105.0)
        assert seen[0].size == pytest.approx(12.0)

    def test_provider_timestamp_comes_from_utm_milliseconds(self):
        provider, streamer = self._provider()
        seen = []
        provider.subscribe("US500", "trade", seen.append)
        streamer.ticks["US500"]("CHART:US500:TICK", {"LTP": "1", "UTM": "1761486000123"})
        assert seen[0].provider_ts_ns == 1_761_486_000_123_000_000

    def test_l2_is_refused_rather_than_faked(self):
        provider, _ = self._provider()
        with pytest.raises(ValueError, match="no L2"):
            provider.subscribe("EURUSD", "l2", lambda e: None)


# ===========================================================================
# Dukascopy bi5 decoding
# ===========================================================================

class TestDukascopy:
    HOUR = datetime(2026, 10, 20, 10, tzinfo=timezone.utc)

    def _payload(self, records):
        raw = b"".join(struct.pack(">IIIII", *r) for r in records)
        return lzma.compress(raw, format=lzma.FORMAT_ALONE)

    def test_decodes_prices_and_timestamps(self):
        payload = self._payload([
            (1_000, 111_943, 111_937, 5, 7),
            (2_500, 111_944, 111_938, 3, 4),
        ])
        events = DukascopyTicks().decode_hour("EURUSD", self.HOUR, payload)
        assert len(events) == 2
        assert events[0].bid == pytest.approx(1.11937)
        assert events[0].ask == pytest.approx(1.11943)
        base_ns = int(self.HOUR.timestamp()) * 1_000_000_000
        assert events[0].provider_ts_ns == base_ns + 1_000_000_000
        assert events[1].provider_ts_ns == base_ns + 2_500_000_000

    def test_jpy_uses_three_decimals(self):
        payload = self._payload([(0, 1_234, 1_233, 1, 1)])
        events = DukascopyTicks().decode_hour("USDJPY", self.HOUR, payload)
        assert events[0].bid == pytest.approx(1.233)

    def test_empty_hour_is_empty_not_an_error(self):
        assert DukascopyTicks().decode_hour("EURUSD", self.HOUR, b"") == []

    def test_month_in_url_is_zero_indexed(self):
        url = DukascopyTicks().hour_url("EURUSD", datetime(2026, 1, 5, 7, tzinfo=timezone.utc))
        assert "/2026/00/05/07h_ticks.bi5" in url

    def test_malformed_payload_is_rejected(self):
        truncated = lzma.compress(b"\x00" * 19, format=lzma.FORMAT_ALONE)
        with pytest.raises(ValueError, match="multiple"):
            DukascopyTicks().decode_hour("EURUSD", self.HOUR, truncated)


# ===========================================================================
# Binance L2 book maintenance
# ===========================================================================

class TestBinanceBook:
    def _provider(self):
        provider = BinanceProvider(client=object())
        provider._books["BTCUSDT"] = _Book(symbol="BTCUSDT", last_update_id=100, synced=True)
        provider._books["BTCUSDT"].bids = {99.0: 1.0}
        provider._books["BTCUSDT"].asks = {101.0: 1.0}
        return provider

    def test_stale_event_is_dropped(self):
        provider = self._provider()
        changes = provider._apply_depth({"s": "BTCUSDT", "U": 90, "u": 100, "b": [["98", "5"]], "a": []})
        assert changes == []

    def test_gap_marks_book_unsynced_and_applies_nothing(self):
        provider = self._provider()
        changes = provider._apply_depth({"s": "BTCUSDT", "U": 105, "u": 106, "b": [["98", "5"]], "a": []})
        assert changes == []
        assert provider._books["BTCUSDT"].synced is False

    def test_valid_diff_applies_and_updates_best(self):
        provider = self._provider()
        changes = provider._apply_depth({
            "s": "BTCUSDT", "U": 101, "u": 102,
            "b": [["100", "5"]], "a": [["101", "0"]],
        })
        assert ("bid", 100.0, 5.0) in changes
        assert provider.best("BTCUSDT") == (100.0, None)

    def test_zero_quantity_removes_level(self):
        provider = self._provider()
        provider._apply_depth({"s": "BTCUSDT", "U": 101, "u": 102, "b": [["99", "0"]], "a": []})
        assert 99.0 not in provider._books["BTCUSDT"].bids

    def test_trade_event_is_emitted_with_provider_time(self):
        provider = self._provider()
        seen = []
        provider._subs.append(("BTCUSDT", "trade", seen.append))
        provider._handle({"data": {"e": "trade", "s": "BTCUSDT", "T": 1761486000123,
                                   "p": "50000.5", "q": "0.010"}})
        assert seen[0].last == pytest.approx(50000.5)
        assert seen[0].provider_ts_ns == 1_761_486_000_123_000_000

    def test_l1_is_refused(self):
        provider = BinanceProvider(client=object())
        with pytest.raises(ValueError, match="L2"):
            provider.subscribe("BTCUSDT", "l1", lambda e: None)
