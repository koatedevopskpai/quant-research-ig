"""
stream.py — IGStreamer: Lightstreamer real-time market data client.

WHY THIS EXISTS:
    Real-time market data is the lifeblood of any HFT or systematic trading
    system. While the REST API (client.py) provides historical data for
    backtesting, the streaming client provides live data for order book
    reconstruction, trade execution, and real-time analytics.

    This is the foundation for Bryan Downing's HFT architecture — the feed
    handler that feeds order book reconstruction (Phase 2b).

WHY LIGHTSTREAMER (ADR-004):
    IG uses Lightstreamer, a proprietary push protocol, for real-time data.
    Alternative: REST polling every second generates ~86K requests/day per
    instrument and has 1-second minimum latency. Lightstreamer pushes
    updates within 10-50ms and includes built-in reconnection and
    gap-filling.

WHY DUAL AUTH:
    Lightstreamer does NOT support IG API v3 OAuth 2.0 Bearer tokens.
    It requires the legacy v2 format: "CST-{cst}|XST-{xst}" as the
    Lightstreamer password. The class transparently performs both auth
    flows:
    - v2 POST /session for CST/XST headers (Lightstreamer password)
    - v3 POST /session for Bearer token (REST API calls, if needed)

ARCHITECTURE NOTE:
    The IGStreamer uses a callback-based API, not a polling API. This is
    consistent with Bryan Downing's event-driven HFT architecture. Each
    market data update is delivered to a user-provided callback function
    as soon as it arrives, enabling sub-50ms reaction times.

LIMITATION (Java dependency):
    The official lightstreamer-client-lib Python library wraps the
    Lightstreamer Java client via a JNI bridge. This means:
    1. Java Runtime Environment (JRE) must be installed.
    2. The JNI bridge can be fragile across platforms.
    3. Startup is slower due to JVM initialisation.

    For Phase 2b (hft-engine C++ core), we will implement a native C++
    Lightstreamer client to eliminate this dependency.
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
        "lightstreamer-client-lib is required for real-time streaming.\n"
        "  Install with:  pip install lightstreamer-client-lib\n"
        "  Note: This requires a Java Runtime (JRE) to be installed."
    )

logger = logging.getLogger(__name__)

# Type alias for the callback function that receives market data updates.
# The first argument is the item name (epic or account identifier).
# The second argument is a flat dict of field-name -> string-value.
FieldCallback = Callable[[str, dict[str, str]], None]


# =============================================================================
# Internal: Lightstreamer Event Bridge
# =============================================================================


class _SubscriptionBridge(SubscriptionListener):
    """
    Bridges Lightstreamer events to a Python callback.

    WHY this bridge exists:
        Lightstreamer's Java-based API uses listener objects. This class
        translates Lightstreamer's event model into a simple Python callback
        pattern that is familiar to quant researchers and easy to integrate
        with pandas/numpy analysis pipelines.

    WHAT IT DOES:
        - on_item_update(): Called by Lightstreamer when subscribed fields
          change. Builds a flat dictionary of all field values and passes
          them to the user's callback.
        - on_subscription() / on_subscription_error(): Log subscription
          lifecycle events for debugging.
        - on_unsubscription(): Cleanup logging.
        - on_real_max_frequency(): Lightstreamer notifies the maximum
          frequency the server will push updates. Not used currently.
    """

    def __init__(self, item_names: list[str], callback: FieldCallback) -> None:
        # WHY item_names is stored:
        #   For logging and debugging — we log subscription events with
        #   the item names so users can see which instruments are live.
        self._item_names = item_names
        self._callback = callback

    def on_item_update(self, update: ItemUpdate) -> None:
        """
        Called by Lightstreamer when subscribed fields receive updates.

        WHY iterating dir(update.getValue):
            Lightstreamer's ItemUpdate.getValue() interface does not expose
            a list of available fields directly. We use dir() to enumerate
            all field names that have values in this update.

        WHY converting to string:
            Lightstreamer sends all field values as strings. Consumers are
            responsible for numeric conversion based on field type
            (e.g., float for prices, int for volume).
        """
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
        pass  # Not used — frequency control is configured via IG settings


# =============================================================================
# IGStreamer — Main Streaming Client
# =============================================================================


class IGStreamer:
    """
    Lightstreamer client for IG real-time market data.

    Usage:
        def on_data(item: str, fields: dict):
            print(f"{item}: bid={fields.get('BID')}")

        streamer = IGStreamer()
        streamer.connect()
        streamer.subscribe_l1("CS.D.EURUSD.CFD.IP", callback=on_data)
        # ... event loop ...
        streamer.disconnect()

    SUBSCRIPTION MODES:
        - MERGE: Server sends only changed fields. Client merges with
          previous state. Used for L1 data and candles.
        - DISTINCT: Every update is a complete new event. Used for ticks.

    WHY NOT A CONTEXT MANAGER:
        Lightstreamer connections are long-lived (hours or days). A context
        manager would disconnect when exiting scope, which is not the
        intended usage pattern. Call connect()/disconnect() explicitly.
    """

    def __init__(self) -> None:
        self.settings = ig_settings
        self._ls_client: Optional[LightstreamerClient] = None

        # WHY tracking subscriptions:
        #   If the Lightstreamer connection drops and reconnects, all
        #   active subscriptions must be re-registered. Storing them here
        #   enables automatic re-subscription in a future enhancement.
        self._subscriptions: list[Subscription] = []
        self._bridges: list[_SubscriptionBridge] = []

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """
        Establish a Lightstreamer connection.

        FLOW:
        1. Fetch v2 CST/XST tokens (required for Lightstreamer password).
        2. Format password as "CST-{cst}|XST-{xst}".
        3. Create LightstreamerClient instance with the IG endpoint.
        4. Set username and password.
        5. Call connect() — Lightstreamer handles the WebSocket handshake.

        WHY adapter set to None:
            IG's Lightstreamer instance uses the DEFAULT adapter set.
            Passing None selects the default, which is the correct choice.
        """
        self.settings.validate_credentials()
        cst, xst = self._fetch_v2_tokens()
        # WHY this specific password format:
        #   Lightstreamer's IG adapter expects the v2 credentials in the
        #   format "CST-{value}|XST-{value}". Any deviation causes the
        #   connection to be rejected with an auth error.
        password = f"CST-{cst}|XST-{xst}"

        self._ls_client = LightstreamerClient(
            self.settings.lightstreamer_url,
            None,  # adapter set — None = "DEFAULT"
        )
        self._ls_client.connectionDetails.setUser(self.settings.username)
        self._ls_client.connectionDetails.setPassword(password)
        self._ls_client.connect()
        logger.info(
            "Lightstreamer connecting to %s", self.settings.lightstreamer_url
        )

    def disconnect(self) -> None:
        """Gracefully disconnect from Lightstreamer."""
        if self._ls_client is not None:
            self._ls_client.disconnect()
            logger.info("Lightstreamer disconnected")

    # ------------------------------------------------------------------
    # Subscription helpers
    # ------------------------------------------------------------------

    def subscribe_l1(self, epic: str, callback: FieldCallback) -> None:
        """
        Subscribe to Level 1 market data.

        WHAT IT PROVIDES:
            Best bid/offer, daily high/low, opening price, change since
            open, market state (open/closed/suspended).

        FIELDS:
            BID, OFFER, HIGH, LOW, MID_OPEN, CHANGE, CHANGE_PCT,
            UPDATE_TIME, MARKET_STATE

        WHY MERGE mode:
            L1 data uses MERGE because only changed fields are sent each
            update (e.g., BID changes without OFFER changing). The client
            merges new values with the previous state.

        Args:
            epic: The instrument epic (e.g., "CS.D.EURUSD.CFD.IP").
            callback: Function receiving (item_name, fields_dict).
        """
        item = f"MARKET:{epic}"
        sub = Subscription(
            mode="MERGE",
            items=[item],
            fields=[
                "BID",
                "OFFER",
                "HIGH",
                "LOW",
                "MID_OPEN",
                "CHANGE",
                "CHANGE_PCT",
                "UPDATE_TIME",
                "MARKET_STATE",
            ],
        )
        bridge = _SubscriptionBridge([item], callback)
        sub.addListener(bridge)
        self._subscriptions.append(sub)
        self._bridges.append(bridge)
        self._ls_client.subscribe(sub)

    def subscribe_ticks(self, epic: str, callback: FieldCallback) -> None:
        """
        Subscribe to tick-by-tick trade data.

        WHAT IT PROVIDES:
            Every individual trade as it happens. This is the most granular
            data available from IG and forms the basis for order book
            reconstruction and microstructure analysis (Phase 2b).

        FIELDS:
            BID, OFR, LTP (last traded price), LTV (last traded volume),
            UTM (update time in milliseconds), DAY_OPEN_MID, DAY_HIGH,
            DAY_LOW

        WHY DISTINCT mode:
            Ticks are discrete events — each trade is a complete new data
            point. DISTINCT mode ensures every tick is delivered to the
            callback without merging.

        WHY TICK data is critical for microstructure (Bryan Downing):
            Tick data reveals:
            - Trade direction (aggressive buy vs. sell)
            - Order flow imbalance
            - Spread dynamics
            - Trade clustering (Hawkes process for toxicity detection)

        Args:
            epic: The instrument epic.
            callback: Function receiving (item_name, fields_dict).
        """
        item = f"CHART:{epic}:TICK"
        sub = Subscription(
            mode="DISTINCT",
            items=[item],
            fields=[
                "BID",
                "OFR",
                "LTP",
                "LTV",
                "UTM",
                "DAY_OPEN_MID",
                "DAY_HIGH",
                "DAY_LOW",
            ],
        )
        bridge = _SubscriptionBridge([item], callback)
        sub.addListener(bridge)
        self._subscriptions.append(sub)
        self._bridges.append(bridge)
        self._ls_client.subscribe(sub)

    def subscribe_candles(
        self,
        epic: str,
        scale: str = "1MINUTE",
        callback: FieldCallback = None,
    ) -> None:
        """
        Subscribe to OHLC candle updates at a specified time scale.

        WHAT IT PROVIDES:
            Aggregated price data over time intervals. Candle data is a
            compressed view of tick activity — it shows open/high/low/close
            for both bid and ask sides.

        FIELDS:
            OFR_OPEN/HIGH/LOW/CLOSE, BID_OPEN/HIGH/LOW/CLOSE,
            LTP_OPEN/HIGH/LOW/CLOSE, UTM, CONS_END, CONS_TICK_COUNT

        AVAILABLE SCALES:
            TICK, SECOND, 1MINUTE, 5MINUTE, 15MINUTE, 30MINUTE,
            1HOUR, 2HOURS, 3HOURS, 4HOURS, DAY, WEEK, MONTH

        WHY MERGE mode for candles:
            Candle data is built incrementally (e.g., open → high → new
            high → close). MERGE mode updates only the changing field,
            while the client maintains the full candle state.

        Args:
            epic: The instrument epic.
            scale: The candle time scale (default "1MINUTE").
            callback: Function receiving (item_name, fields_dict).
        """
        item = f"CHART:{epic}:{scale}"
        sub = Subscription(
            mode="MERGE",
            items=[item],
            fields=[
                "OFR_OPEN",
                "OFR_HIGH",
                "OFR_LOW",
                "OFR_CLOSE",
                "BID_OPEN",
                "BID_HIGH",
                "BID_LOW",
                "BID_CLOSE",
                "LTP_OPEN",
                "LTP_HIGH",
                "LTP_LOW",
                "LTP_CLOSE",
                "UTM",
                "CONS_END",
                "CONS_TICK_COUNT",
            ],
        )
        bridge = _SubscriptionBridge(
            [item], callback or self._default_callback
        )
        sub.addListener(bridge)
        self._subscriptions.append(sub)
        self._bridges.append(bridge)
        self._ls_client.subscribe(sub)

    def subscribe_account(self, callback: FieldCallback = None) -> None:
        """
        Subscribe to account-level P&L and margin data.

        WHAT IT PROVIDES:
            Real-time account status — profit/loss, available funds,
            margin usage, equity.

        FIELDS:
            PNL, FUNDS, MARGIN, AVAILABLE_TO_DEAL, EQUITY, EQUITY_USED

        WHY subscribe_account exists:
            For the agentic-finance module (Phase 5), agents need real-time
            visibility into account status for position sizing, risk
            management, and drawdown monitoring.

        Args:
            callback: Function receiving (username, fields_dict).
        """
        item = f"ACCOUNT:{self.settings.username}"
        sub = Subscription(
            mode="MERGE",
            items=[item],
            fields=[
                "PNL",
                "FUNDS",
                "MARGIN",
                "AVAILABLE_TO_DEAL",
                "EQUITY",
                "EQUITY_USED",
            ],
        )
        bridge = _SubscriptionBridge(
            [item], callback or self._default_callback
        )
        sub.addListener(bridge)
        self._subscriptions.append(sub)
        self._bridges.append(bridge)
        self._ls_client.subscribe(sub)

    # ------------------------------------------------------------------
    # Run / stop convenience
    # ------------------------------------------------------------------

    def run(self) -> None:
        """
        Convenience method: connect and start streaming.

        Alternative to calling connect() directly. Exists for consistency
        with the run_analysis.py entry point pattern.
        """
        self.connect()

    def stop(self) -> None:
        """Convenience method: disconnect."""
        self.disconnect()

    @staticmethod
    def _default_callback(item: str, fields: dict) -> None:
        """
        Default callback that prints updates to stdout.

        WHY a default:
            Users can subscribe without providing a callback for quick
            debugging. The default callback prints to stdout so data
            visibility is immediate.
        """
        print(f"[{item}] {fields}")

    # ------------------------------------------------------------------
    # V2 auth (CST / XST are needed for Lightstreamer)
    # ------------------------------------------------------------------

    def _fetch_v2_tokens(self) -> tuple[str, str]:
        """
        Fetch v2 CST and X-SECURITY-TOKEN for Lightstreamer authentication.

        WHY v2 auth is still needed:
            IG's Lightstreamer adapter has not been migrated to v3 OAuth 2.0.
            It requires the legacy CST (Client Session Token) and
            X-SECURITY-TOKEN headers from the v2 auth endpoint.

        WHY a separate HTTP session (not using self._client):
            This method may be called before the v3 client is authenticated
            or if a new v2 session is needed independently. Using a fresh
            httpx.post call keeps the v2 and v3 auth flows independent.

        HOW IT WORKS:
            1. POST /session (v2) with API key, username, password.
            2. IG returns CST in the "CST" response header.
            3. IG returns X-SECURITY-TOKEN in the "X-SECURITY-TOKEN" header.
            4. These are formatted as "CST-{cst}|XST-{xst}" for
               Lightstreamer's password field.

        Raises:
            RuntimeError: If v2 auth fails or headers are missing.
        """
        headers = {
            "X-IG-API-KEY": self.settings.api_key,
            "Content-Type": "application/json",
            "VERSION": "2",  # v2 endpoint returns CST/XST headers
        }
        body = {
            "identifier": self.settings.username,
            "password": self.settings.password,
        }
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
            # WHY checking both headers:
            #   Rarely, IG may return HTTP 200 without the expected headers
            #   (e.g., during partial API outages). Checking explicitly
            #   provides a clear error rather than a cryptic connection
            #   failure later.
            raise RuntimeError(
                "Missing CST or X-SECURITY-TOKEN in v2 auth response"
            )
        return cst, xst
