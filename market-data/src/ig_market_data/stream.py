"""
stream.py — IGStreamer: Lightstreamer real-time market data client.

Connects to IG's Lightstreamer endpoint, subscribes to MARKET, CHART:TICK,
and CHART:candle feeds, and dispatches updates via callbacks.

Lightstreamer auth requires the v2 CST/XST tokens (pipe-separated), NOT the
v3 OAuth Bearer token.  The class authenticates twice: one v2 session for LS,
one v3 session for REST calls.

Usage:
    from ig_market_data.stream import IGStreamer

    def on_tick(item: str, fields: dict):
        print(item, fields)

    streamer = IGStreamer()
    streamer.subscribe_l1("CS.D.EURUSD.CFD.IP", callback=on_tick)
    streamer.subscribe_ticks("CS.D.EURUSD.CFD.IP", callback=on_tick)
    streamer.start()
    # ... run event loop ...
    streamer.stop()

Requires:
    pip install lightstreamer-client-lib
"""

import logging
from typing import Callable, Optional

import httpx

from ig_market_data.config import ig_settings

try:
    from lightstreamer.client import (
        LightstreamerClient,
        Subscription,
        SubscriptionListener,
        ItemUpdate,
    )
except ImportError:
    raise ImportError(
        "lightstreamer-client-lib is required. Install with:\n"
        "  pip install lightstreamer-client-lib"
    )

logger = logging.getLogger(__name__)

FieldCallback = Callable[[str, dict[str, str]], None]


class _SubscriptionBridge(SubscriptionListener):
    """
    Internal listener that bridges Lightstreamer events to a Python callback.
    """

    def __init__(self, item_names: list[str], callback: FieldCallback) -> None:
        self._item_names = item_names
        self._callback = callback

    def on_item_update(self, update: ItemUpdate) -> None:
        # Build a flat dict of all subscribed fields
        fields: dict[str, str] = {}
        for name in dir(update.getValue):
            val = update.getValue(name)
            if val is not None:
                fields[name] = str(val)
        self._callback(update.getItemName(), fields)

    def on_subscription(self) -> None:
        logger.info("Subscribed to %s", self._item_names)

    def on_subscription_error(self, code: int, message: str) -> None:
        logger.error("Subscription error [%s] %s", code, message)

    def on_unsubscription(self) -> None:
        logger.info("Unsubscribed from %s", self._item_names)

    def on_real_max_frequency(self, frequency: str) -> None:
        pass


class IGStreamer:
    def __init__(self) -> None:
        self.settings = ig_settings
        self._ls_client: Optional[LightstreamerClient] = None
        self._subscriptions: list[Subscription] = []
        self._bridges: list[_SubscriptionBridge] = []

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------
    def connect(self) -> None:
        cst, xst = self._fetch_v2_tokens()
        password = f"CST-{cst}|XST-{xst}"

        self._ls_client = LightstreamerClient(
            self.settings.lightstreamer_url,
            None,  # adapter set — None = "DEFAULT"
        )
        self._ls_client.connectionDetails.setUser(self.settings.username)
        self._ls_client.connectionDetails.setPassword(password)
        self._ls_client.connect()
        logger.info("Lightstreamer connecting to %s", self.settings.lightstreamer_url)

    def disconnect(self) -> None:
        if self._ls_client is not None:
            self._ls_client.disconnect()
            logger.info("Lightstreamer disconnected")

    # ------------------------------------------------------------------
    # Subscription helpers
    # ------------------------------------------------------------------
    def subscribe_l1(self, epic: str, callback: FieldCallback) -> None:
        """
        Level 1 market data: bid, ask, daily high/low, market state.
        Fields: BID, OFFER, HIGH, LOW, MID_OPEN, CHANGE, CHANGE_PCT,
                UPDATE_TIME, MARKET_STATE.
        """
        item = f"MARKET:{epic}"
        sub = Subscription(
            mode="MERGE",
            items=[item],
            fields=[
                "BID", "OFFER", "HIGH", "LOW",
                "MID_OPEN", "CHANGE", "CHANGE_PCT",
                "UPDATE_TIME", "MARKET_STATE",
            ],
        )
        bridge = _SubscriptionBridge([item], callback)
        sub.addListener(bridge)
        self._subscriptions.append(sub)
        self._bridges.append(bridge)
        self._ls_client.subscribe(sub)

    def subscribe_ticks(self, epic: str, callback: FieldCallback) -> None:
        """
        Tick-by-tick chart data (DISTINCT mode — every tick generates
        a new event).
        Fields: BID, OFR, LTP, LTV, UTM, DAY_OPEN_MID, DAY_HIGH, DAY_LOW.
        """
        item = f"CHART:{epic}:TICK"
        sub = Subscription(
            mode="DISTINCT",
            items=[item],
            fields=[
                "BID", "OFR", "LTP", "LTV", "UTM",
                "DAY_OPEN_MID", "DAY_HIGH", "DAY_LOW",
            ],
        )
        bridge = _SubscriptionBridge([item], callback)
        sub.addListener(bridge)
        self._subscriptions.append(sub)
        self._bridges.append(bridge)
        self._ls_client.subscribe(sub)

    def subscribe_candles(
        self, epic: str, scale: str = "1MINUTE", callback: FieldCallback = None
    ) -> None:
        """
        OHLC candle data at any chart scale.
        Fields: OFR_OPEN/.._HIGH/.._LOW/.._CLOSE
                BID_OPEN/.._HIGH/.._LOW/.._CLOSE
                LTP_OPEN/.._HIGH/.._LOW/.._CLOSE
                CONS_END, CONS_TICK_COUNT, UTM.

        Available scales: TICK, SECOND, 1MINUTE, 5MINUTE, 15MINUTE,
                          30MINUTE, 1HOUR, 2HOURS, 3HOURS, 4HOURS,
                          DAY, WEEK, MONTH.
        """
        item = f"CHART:{epic}:{scale}"
        sub = Subscription(
            mode="MERGE",
            items=[item],
            fields=[
                "OFR_OPEN", "OFR_HIGH", "OFR_LOW", "OFR_CLOSE",
                "BID_OPEN", "BID_HIGH", "BID_LOW", "BID_CLOSE",
                "LTP_OPEN", "LTP_HIGH", "LTP_LOW", "LTP_CLOSE",
                "UTM", "CONS_END", "CONS_TICK_COUNT",
            ],
        )
        bridge = _SubscriptionBridge([item], callback or self._default_callback)
        sub.addListener(bridge)
        self._subscriptions.append(sub)
        self._bridges.append(bridge)
        self._ls_client.subscribe(sub)

    def subscribe_account(self, callback: FieldCallback = None) -> None:
        """
        Account-level P&L and margin data.
        Fields: PNL, FUNDS, MARGIN, AVAILABLE_TO_DEAL, EQUITY.
        """
        item = f"ACCOUNT:{self.settings.username}"
        sub = Subscription(
            mode="MERGE",
            items=[item],
            fields=[
                "PNL", "FUNDS", "MARGIN",
                "AVAILABLE_TO_DEAL", "EQUITY", "EQUITY_USED",
            ],
        )
        bridge = _SubscriptionBridge([item], callback or self._default_callback)
        sub.addListener(bridge)
        self._subscriptions.append(sub)
        self._bridges.append(bridge)
        self._ls_client.subscribe(sub)

    # ------------------------------------------------------------------
    # Run / stop convenience
    # ------------------------------------------------------------------
    def run(self) -> None:
        self.connect()

    def stop(self) -> None:
        self.disconnect()

    @staticmethod
    def _default_callback(item: str, fields: dict) -> None:
        print(f"[{item}] {fields}")

    # ------------------------------------------------------------------
    # V2 auth (CST / XST are needed for Lightstreamer passwords)
    # ------------------------------------------------------------------
    def _fetch_v2_tokens(self) -> tuple[str, str]:
        headers = {
            "X-IG-API-KEY": self.settings.api_key,
            "Content-Type": "application/json",
            "VERSION": "2",
        }
        body = {"identifier": self.settings.username, "password": self.settings.password}
        resp = httpx.post(
            f"{self.settings.base_url}/session",
            headers=headers,
            json=body,
            timeout=15,
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"v2 auth failed: HTTP {resp.status_code} {resp.text[:200]}"
            )
        cst = resp.headers.get("CST", "")
        xst = resp.headers.get("X-SECURITY-TOKEN", "")
        if not cst or not xst:
            raise RuntimeError(
                "Missing CST or X-SECURITY-TOKEN in v2 auth response"
            )
        return cst, xst