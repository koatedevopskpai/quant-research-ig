"""
provider_smoke.py — live, read-only checks for the market-data providers.

    python scripts/provider_smoke.py binance  --symbol BTCUSDT --seconds 20
    python scripts/provider_smoke.py dukascopy --instrument EURUSD --date 2026-10-20 --hour 10
    python scripts/provider_smoke.py ig       --epic-l1 CS.D.EURUSD.CFD.IP --seconds 15

Not part of CI: CI must never depend on a network or a vendor. These exist so a
human can confirm a provider is alive after a dependency or endpoint change, and
so the IG deferment in the README has a concrete exit test.
"""

from __future__ import annotations

import argparse
import statistics
from datetime import datetime, timezone

from ig_market_data.providers.binance import BinanceProvider
from ig_market_data.providers.dukascopy import DukascopyTicks


def binance(symbol: str, seconds: float) -> None:
    provider = BinanceProvider()
    seen = {"l2": 0, "trade": 0, "last_trade": None, "sample_levels": 0}

    def on_l2(event) -> None:
        seen["l2"] += 1
        seen["sample_levels"] += 1

    def on_trade(event) -> None:
        seen["trade"] += 1
        seen["last_trade"] = (event.last, event.size, event.provider_ts_ns)

    provider.connect()
    provider.subscribe(symbol, "l2", on_l2)
    provider.subscribe(symbol, "trade", on_trade)
    counts = provider.run_for(seconds)
    bid, ask = provider.best(symbol)
    provider.close()

    print(f"symbol={symbol} seconds={seconds}")
    print(f"  l2 events:    {counts['l2']}")
    print(f"  trade events: {counts['trade']}")
    print(f"  best bid/ask: {bid} / {ask}")
    if seen["last_trade"]:
        print(f"  last trade:   price={seen['last_trade'][0]} size={seen['last_trade'][1]}")


def dukascopy(instrument: str, date: str, hour: int) -> None:
    day = datetime.strptime(date, "%Y-%m-%d").replace(hour=hour, tzinfo=timezone.utc)
    ticks = DukascopyTicks()
    events = ticks.fetch_hour(instrument, day)
    ticks.close()

    print(f"instrument={instrument} hour={day.isoformat()} records={len(events)}")
    if not events:
        print("  no records (market closed or no data for this hour)")
        return
    spreads = [e.ask - e.bid for e in events if e.bid is not None and e.ask is not None]
    print(f"  first: bid={events[0].bid} ask={events[0].ask} ts={events[0].provider_ts_ns}")
    print(f"  last:  bid={events[-1].bid} ask={events[-1].ask} ts={events[-1].provider_ts_ns}")
    print(f"  spread: median={statistics.median(spreads):.5f} max={max(spreads):.5f}")


def ig(epic_l1: str, epic_ticks: str, seconds: float) -> None:
    """IG demo check: auth, L1 replacement, tick freshness, then reconnect."""
    import time

    from ig_market_data.providers.ig import IGProvider

    counts = {
        "l1": 0,
        "trade": 0,
        "after_reconnect": 0,
        "l1_sample": None,
        "trade_sample": None,
        "utm_age_ms": [],
    }

    def on_l1(event) -> None:
        counts["l1"] += 1
        if counts["l1_sample"] is None:
            counts["l1_sample"] = (event.bid, event.ask, event.book_action)

    def on_trade(event) -> None:
        counts["trade"] += 1
        counts["utm_age_ms"].append((time.time_ns() - event.provider_ts_ns) / 1e6)
        if counts["trade_sample"] is None:
            counts["trade_sample"] = (event.last, event.size)

    def on_trade_after(event) -> None:
        counts["after_reconnect"] += 1

    provider = IGProvider()
    provider.connect()
    provider.subscribe(epic_l1, "l1", on_l1)
    provider.subscribe(epic_ticks, "trade", on_trade)
    time.sleep(seconds)

    provider.disconnect()
    time.sleep(3)
    provider.connect()
    time.sleep(3)
    provider.subscribe(epic_ticks, "trade", on_trade_after)
    time.sleep(max(seconds / 2, 5))

    provider.disconnect()

    print(f"l1={epic_l1} ticks={epic_ticks} seconds={seconds}")
    print(f"  L1 events:        {counts['l1']}")
    print(f"  trade events:     {counts['trade']}")
    print(f"  after reconnect:  {counts['after_reconnect']}")
    if counts["l1_sample"]:
        print(f"  L1 sample:        bid/ask/action={counts['l1_sample']}")
    if counts["trade_sample"]:
        print(f"  trade sample:     last/size={counts['trade_sample']}")
    if counts["utm_age_ms"]:
        ages = sorted(counts["utm_age_ms"])
        print(
            f"  provider UTM age: median={ages[len(ages)//2]:.0f} ms "
            f"min={ages[0]:.0f} max={ages[-1]:.0f}"
        )
    if counts["l1"] == 0:
        print("  NOTE: no L1 events. This is the open question in")
        print("        docs/phase-3-market-data.md (MARKET: items delivered nothing).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="source", required=True)

    b = sub.add_parser("binance")
    b.add_argument("--symbol", default="BTCUSDT")
    b.add_argument("--seconds", type=float, default=20.0)

    d = sub.add_parser("dukascopy")
    d.add_argument("--instrument", default="EURUSD")
    d.add_argument("--date", default="2026-10-20")
    d.add_argument("--hour", type=int, default=10)

    i = sub.add_parser("ig")
    i.add_argument("--epic-l1", default="CS.D.EURUSD.CFD.IP")
    i.add_argument("--epic-ticks", default="IX.D.SPTRD.IFD.IP")
    i.add_argument("--seconds", type=float, default=15.0)

    args = parser.parse_args()
    if args.source == "binance":
        binance(args.symbol, args.seconds)
    elif args.source == "dukascopy":
        dukascopy(args.instrument, args.date, args.hour)
    else:
        ig(args.epic_l1, args.epic_ticks, args.seconds)


if __name__ == "__main__":
    main()
