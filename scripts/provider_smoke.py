"""
provider_smoke.py — live, read-only checks for the non-IG providers.

    python scripts/provider_smoke.py binance  --symbol BTCUSDT --seconds 20
    python scripts/provider_smoke.py dukascopy --instrument EURUSD --date 2026-10-20 --hour 10

Not part of CI: CI must never depend on a network or a vendor. These exist so a
human can confirm a provider is alive after a dependency or endpoint change.
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

    args = parser.parse_args()
    if args.source == "binance":
        binance(args.symbol, args.seconds)
    else:
        dukascopy(args.instrument, args.date, args.hour)


if __name__ == "__main__":
    main()
