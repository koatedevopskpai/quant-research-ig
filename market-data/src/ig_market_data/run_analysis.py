"""
run_analysis.py — Real-time market data streaming demo.

WHY THIS EXISTS:
    After credentials are verified (verify_ig.py), this script demonstrates
    the real-time streaming capability. It subscribes to:
    1. Level 1 market data for US 500 (S&P 500 CFD) — best bid/offer,
       daily high/low, market state.
    2. Tick-by-tick data for EUR/USD — every individual trade as it happens.

    This is the foundation for Bryan Downing's HFT architecture — the feed
    handler that will feed order book reconstruction (Phase 2b) and
    microstructure analytics.

WHAT IT DEMONSTRATES:
    - IGStreamer connection and dual auth (v2 for LS, v3 for REST).
    - Multiple subscription types (L1 MERGE + ticks DISTINCT).
    - Callback-based data processing.
    - Graceful shutdown on Ctrl+C.

EXPECTED OUTPUT:
    2025-01-02 10:00:01 [L1] IX.D.SPTRD.IFD.IP  bid=   4728.5  ask=   4729.0  state=TRADEABLE
    2025-01-02 10:00:01 [TICK] CS.D.EURUSD.CFD.IP  bid=   1.1234  ofr=   1.1235  ltp=   1.1234

METHODOLOGY NOTE (Bryan Downing's HFT Philosophy):
    Downing's Ether HFT bot architecture starts with the feed handler —
    the component that ingests raw market data ticks. Every subsequent
    component (order book, regime detector, toxicity detector, strategy)
    depends on the speed and correctness of this layer. Running this
    script confirms the feed handler is operational before building
    the rest of the HFT pipeline.
"""

import logging
import signal
import sys
from typing import NoReturn

from ig_market_data.stream import IGStreamer

# WHY INFO level logging:
#   For a demo script, INFO provides useful lifecycle messages (connected,
#   subscribed, disconnected) without the noise of DEBUG-level detail.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
logger = logging.getLogger("run_analysis")


def on_l1_update(item: str, fields: dict) -> None:
    """
    Callback for Level 1 market data updates.

    WHAT IT PRINTS:
        Item name, best bid, best ask, and market state.

    WHY these specific fields:
        BID and OFFER are the most commonly watched L1 fields — they
        represent the current tradable prices. MARKET_STATE indicates
        whether the instrument is tradeable, closed, or in auction.

    FUTURE USE (Phase 2b):
        In production, this callback would feed:
        - Order book reconstruction (L1 is the top of the book)
        - Spread monitoring and analysis
        - Trade signal generation
    """
    bid = fields.get("BID", "?")
    ask = fields.get("OFFER", "?")
    state = fields.get("MARKET_STATE", "?")
    print(f"[L1] {item:40s}  bid={bid:>10s}  ask={ask:>10s}  state={state}")


def on_tick_update(item: str, fields: dict) -> None:
    """
    Callback for tick-by-tick trade updates.

    WHAT IT PRINTS:
        Item name, best bid, best offer, and last traded price.

    WHY TICK data matters (Bryan Downing):
        Ticks are the atomic unit of market data. They reveal:
        - Trade direction: was the trade at the bid (sell) or ask (buy)?
        - Order flow: are more trades buying or selling?
        - Trade clustering: do trades arrive in bursts? (Hawkes process)
        - These are the inputs to the Order Flow Toxicity Detector.

    FUTURE USE (Phase 3b):
        This callback will feed:
        - Order book reconstruction (L2 depth)
        - Microstructure analytics (spread, depth, trade signs)
        - Order Flow Toxicity Detector (Bayesian + Hawkes)
        - HMM regime detection (price return distribution)
    """
    bid = fields.get("BID", "?")
    ofr = fields.get("OFR", "?")
    ltp = fields.get("LTP", "?")
    print(f"[TICK] {item:40s}  bid={bid:>10s}  ofr={ofr:>10s}  ltp={ltp:>10s}")


def main() -> NoReturn:
    """
    Main entry point: connect to Lightstreamer and stream live data.

    DESIGN DECISIONS:
    1. Creating IGStreamer inside main() (not module-level):
       Module-level instantiation would connect on import, which is
       undesirable for testing and import-only scenarios.

    2. signal.signal for shutdown:
       Lightstreamer runs on its own threads. signal.pause() keeps the
       main thread alive while data streams on background threads.
       Ctrl+C triggers the handler which calls streamer.stop().

    3. No event loop:
       Lightstreamer handles threading internally. We do not need
       asyncio or a manual event loop.
    """
    streamer = IGStreamer()
    streamer.connect()

    # WHY these specific epics:
    #   IX.D.SPTRD.IFD.IP = US 500 (S&P 500 CFD) — the most commonly
    #   traded index CFD on IG.
    #   CS.D.EURUSD.CFD.IP = EUR/USD — the most liquid forex pair.
    #   Together they demonstrate both index and forex data streaming.
    streamer.subscribe_l1("IX.D.SPTRD.IFD.IP", callback=on_l1_update)
    streamer.subscribe_ticks("CS.D.EURUSD.CFD.IP", callback=on_tick_update)

    def shutdown(sig, frame) -> None:
        """
        Graceful shutdown on Ctrl+C or SIGTERM.

        WHY graceful shutdown matters:
            Lightstreamer performs a clean disconnect handshake. Simply
            killing the process can leave stale sessions on IG's server,
            potentially affecting rate limits or connection quotas.
        """
        print("\nShutting down...")
        streamer.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # WHY signal.pause():
    #   The main thread waits for signals while Lightstreamer data streams
    #   on background threads. This is simpler and more efficient than a
    #   busy-wait loop.
    signal.pause()


if __name__ == "__main__":
    main()
