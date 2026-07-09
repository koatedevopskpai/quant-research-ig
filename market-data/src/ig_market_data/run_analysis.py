"""
run_analysis.py — Real-time market data streaming demo.

Usage:
    python -m ig_market_data.run_analysis

Subscribes to L1 market data and tick-by-tick for US 500 and EUR/USD.
Press Ctrl+C to stop.
"""

import logging
import signal
import sys
from typing import NoReturn

from ig_market_data.stream import IGStreamer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
logger = logging.getLogger("run_analysis")


def on_l1_update(item: str, fields: dict) -> None:
    bid = fields.get("BID", "?")
    ask = fields.get("OFFER", "?")
    state = fields.get("MARKET_STATE", "?")
    print(f"[L1] {item:40s}  bid={bid:>10s}  ask={ask:>10s}  state={state}")


def on_tick_update(item: str, fields: dict) -> None:
    bid = fields.get("BID", "?")
    ofr = fields.get("OFR", "?")
    ltp = fields.get("LTP", "?")
    print(f"[TICK] {item:40s}  bid={bid:>10s}  ofr={ofr:>10s}  ltp={ltp:>10s}")


def main() -> NoReturn:
    streamer = IGStreamer()
    streamer.connect()

    streamer.subscribe_l1("IX.D.SPTRD.IFD.IP", callback=on_l1_update)
    streamer.subscribe_ticks("CS.D.EURUSD.CFD.IP", callback=on_tick_update)

    def shutdown(sig, frame) -> None:
        print("\nShutting down...")
        streamer.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Keep main thread alive — Lightstreamer runs on its own threads
    signal.pause()


if __name__ == "__main__":
    main()