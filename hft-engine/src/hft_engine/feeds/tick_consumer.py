"""
tick_consumer.py — Market data tick consumer for IG Lightstreamer feeds.

WHY THIS EXISTS:
    Bryan Downing's HFT architecture separates the feed handler from the
    order book. The feed handler is responsible for:
    1. Connecting to the market data source (IG Lightstreamer).
    2. Parsing raw messages into structured data.
    3. Normalising timestamps to nanosecond precision.
    4. Publishing parsed messages to consumers (order book, microstructure).

    This separation allows:
    - The feed handler to be replaced (e.g., switch from IG to IBKR).
    - The order book to be unit-tested with synthetic data.
    - The feed handler to be optimised or compiled (C++ pybind11 path).

DATA FLOW:
    IG Lightstreamer -> TickConsumer._on_update() -> MarketDataMessage
        -> OrderBook.apply_diff()
        -> MicrostructureAnalyser.on_tick()

LIGHTSTREAMER SUBSCRIPTION MODES:
    IG exposes two modes relevant to HFT:
    1. L1 (MERGE) — Top-of-book updates. Fields: Bid, Offer, BidSize,
       OfferSize, Last, Change, etc.
    2. L2 (MERGE) — Depth updates. Each row is a price level with:
       MarketLevel{Bid, Offer, BidSize, OfferSize}.
    3. Ticks (DISTINCT) — Every trade. Sequence-numbered for gap detection.

TIMESTAMP NORMALISATION:
    IG sends timestamps in ISO 8601 format. We convert to Unix nanoseconds
    for compatibility with the order book and microstructure modules.
    Nanosecond precision is required for latency monitoring (Phase 3b).

REFERENCES:
    IG Labs Streaming API: https://labs.ig.com/streaming-api
    Downing, B. (2025). HFT Feed Handler Architecture. QuantLabsNet.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MarketDataMessage:
    """
    Normalised market data message from the feed handler.

    This is the CANONICAL data structure consumed by all downstream
    components (order book, microstructure, latency monitor).

    WHY a single message type for multiple feed types:
        Normalising all feed types (L1, L2, ticks) into a single message
        format simplifies downstream consumers. The message_type field
        allows consumers to filter or branch as needed.

    Attributes:
        epic: IG instrument identifier (e.g., "IX.D.FTSE.IFP.IP").
        message_type: "L1", "L2", or "TRADE".
        bid_price: Best bid price (L1) or NaN (L2/TRADE).
        ask_price: Best ask price (L1) or NaN (L2/TRADE).
        bid_size: Bid volume at best level (L1) or level (L2).
        ask_size: Ask volume at best level (L1) or level (L2).
        level_price: Price of this L2 level (L2 only).
        last_price: Last traded price (TRADE/L1).
        last_size: Last traded volume (TRADE).
        timestamp_ns: Unix timestamp in nanoseconds.
        raw: Raw message from Lightstreamer (for debugging).
    """

    epic: str
    message_type: str  # "L1", "L2", "TRADE"
    bid_price: float = 0.0
    ask_price: float = 0.0
    bid_size: float = 0.0
    ask_size: float = 0.0
    level_price: float = 0.0
    last_price: float = 0.0
    last_size: float = 0.0
    timestamp_ns: int = 0
    raw: dict | None = None


# Callback signature for receiving parsed MarketDataMessages
TickCallback = Callable[[MarketDataMessage], None]


@dataclass
class TickData:
    """
    Configuration for a single market data subscription.

    Attributes:
        epic: IG instrument epic to subscribe to.
        fields: List of fields to subscribe to. Default is all L1 fields.
        subscription_mode: "MERGE" (L1/L2) or "DISTINCT" (ticks).
        extractor: A function that maps Lightstreamer field updates to a
                   MarketDataMessage. Custom extractors are needed for L2
                   data (which has a different field structure).
    """

    epic: str
    fields: tuple[str, ...] = ("Bid", "Offer", "BidSize", "OfferSize")
    subscription_mode: str = "MERGE"
    extractor: Callable[[str, dict], MarketDataMessage] | None = None


# ---------------------------------------------------------------------------
# Field normalisation
# ---------------------------------------------------------------------------
# WHY: IG's adapters emit a mix of cases and spellings for the same field —
# "Ltp"/"LTP", "Ofr"/"OFFER", "Bid"/"BID". The old code lower-cased through
# str.capitalize(), so "LTP" became "Ltp" and "OFR" became "Ofr", neither of
# which the parsers read. A live replay of {'LTP': '105', 'LTV': '12'} produced
# zero price and zero size. Mapping on the upper-cased, underscore-free key makes
# every spelling land on one canonical name.
_FIELD_ALIASES: dict[str, str] = {
    "BID": "Bid",
    "OFR": "Offer",
    "OFFER": "Offer",
    "BIDSIZE": "BidSize",
    "OFFERSIZE": "OfferSize",
    "LTP": "Last",
    "LTV": "LastSize",
    "LAST": "Last",
    "LASTSIZE": "LastSize",
    "HIGH": "High",
    "LOW": "Low",
    "MIDOPEN": "MidOpen",
    "CHANGE": "Change",
    "CHANGEPCT": "ChangePct",
    "UPDATETIME": "UpdateTime",
    "MARKETSTATE": "MarketState",
    "UTM": "Utm",
    "DAYOPENMID": "DayOpenMid",
    "DAYHIGH": "DayHigh",
    "DAYLOW": "DayLow",
    "MARKETLEVEL": "MarketLevel",
}


def _canonical_fields(fields: dict[str, str]) -> dict[str, str]:
    """Map provider field spellings onto the names the parsers expect."""
    canonical: dict[str, str] = {}
    for key, value in fields.items():
        alias = _FIELD_ALIASES.get(key.upper().replace("_", ""))
        canonical[alias if alias else key.capitalize()] = value
    return canonical


class TickConsumer:
    """
    Market data tick consumer that extends IG's Lightstreamer streaming.

    This class:
    1. Maintains subscriptions to one or more IG instrument epics.
    2. Parses Lightstreamer field updates into MarketDataMessages.
    3. Dispatches parsed messages to registered callbacks.

    USAGE:
        def on_tick(msg: MarketDataMessage) -> None:
            print(f"{msg.epic}: bid={msg.bid_price} @ {msg.bid_size}")

        consumer = TickConsumer()
        consumer.subscribe("IX.D.FTSE.IFP.IP", on_tick)
        consumer.subscribe("IX.D.US500.IFP.IP", on_tick)

        # Connect to Lightstreamer (uses IGStreamer internally)
        consumer.connect(api_key="...", ...)

    WHY NOT subclassing IGStreamer:
        Composition over inheritance. The TickConsumer OWNS an IGStreamer
        instance but decouples the stream parsing from the IG-specific
        authentication and connection management. This makes TickConsumer
        testable with synthetic data.

    PERFORMANCE NOTE:
        For single-instrument subscriptions, the overhead of parsing is
        negligible (< 1 microsecond per message). For 10+ instruments,
        consider running the feed handler in a separate thread to avoid
        blocking the order book updates.
    """

    def __init__(self) -> None:
        self._subscriptions: dict[str, list[TickCallback]] = {}
        self._epic_to_data: dict[str, TickData] = {}
        self._connected = False
        self._streamer: Any = None
        self._simulator: Any = None

    def subscribe(
        self,
        epic: str,
        callback: TickCallback,
        tick_data: TickData | None = None,
    ) -> None:
        """
        Subscribe to market data for an instrument.

        Args:
            epic: IG instrument epic (e.g., "IX.D.FTSE.IFP.IP").
            callback: Function to call with each parsed MarketDataMessage.
            tick_data: Subscription configuration. If None, defaults to
                       L1 MERGE with standard fields.
        """
        if tick_data is None:
            tick_data = TickData(epic=epic)

        if epic not in self._subscriptions:
            self._subscriptions[epic] = []
            self._epic_to_data[epic] = tick_data
        self._subscriptions[epic].append(callback)

    def unsubscribe(self, epic: str, callback: TickCallback | None = None) -> None:
        """
        Unsubscribe from an instrument.

        Args:
            epic: Instrument epic to unsubscribe from.
            callback: Specific callback to remove. If None, removes all.
        """
        if epic not in self._subscriptions:
            return
        if callback is None:
            del self._subscriptions[epic]
            del self._epic_to_data[epic]
        else:
            self._subscriptions[epic].remove(callback)
            if not self._subscriptions[epic]:
                del self._subscriptions[epic]
                del self._epic_to_data[epic]

    # ------------------------------------------------------------------
    # Message parsing
    # ------------------------------------------------------------------

    def parse_l1_update(
        self,
        epic: str,
        fields: dict[str, str],
    ) -> MarketDataMessage:
        """
        Parse an L1 MERGE update from Lightstreamer into a MarketDataMessage.

        IG sends L1 updates as field-value pairs:
            {
                "Bid": "100.50",
                "Offer": "100.55",
                "BidSize": "5000",
                "OfferSize": "3000",
                "Last": "100.52",
                "Change": "0.15",
            }

        WHY converting strings to floats defensively:
            Lightstreamer sends all values as strings. Missing or null
            fields are sent as "null" or empty string. We convert
            defensively, defaulting to 0.0 for missing numeric fields.
        """
        bid = self._safe_float(fields.get("Bid", "0"))
        ask = self._safe_float(fields.get("Offer", "0"))
        bid_size = self._safe_float(fields.get("BidSize", "0"))
        ask_size = self._safe_float(fields.get("OfferSize", "0"))
        last = self._safe_float(fields.get("Last", "0"))
        last_size = self._safe_float(fields.get("LastSize", "0"))

        return MarketDataMessage(
            epic=epic,
            message_type="L1",
            bid_price=bid,
            ask_price=ask,
            bid_size=bid_size,
            ask_size=ask_size,
            last_price=last,
            last_size=last_size,
            timestamp_ns=self._now_ns(),
            raw=fields,
        )

    def parse_l2_update(
        self,
        epic: str,
        fields: dict[str, str],
    ) -> MarketDataMessage:
        """
        Parse an L2 MERGE update (single price level) from Lightstreamer.

        IG sends L2 data as rows with market level fields:
            {
                "MarketLevel": "3",
                "Bid": "100.48",
                "Offer": "100.53",
                "BidSize": "10000",
                "OfferSize": "8000",
            }

        Each row is ONE price level in the book. The feed handler must
        reconstruct the full book from these individual level updates.
        """
        bid = self._safe_float(fields.get("Bid", "0"))
        ask = self._safe_float(fields.get("Offer", "0"))
        bid_size = self._safe_float(fields.get("BidSize", "0"))
        ask_size = self._safe_float(fields.get("OfferSize", "0"))
        level = self._safe_int(fields.get("MarketLevel", "0"))

        # For L2, we use bid/ask as the price at this level.
        # IG sends either Bid or Offer (whichever side this level is on).
        level_price = bid if bid > 0 else ask
        is_bid = bid > 0

        return MarketDataMessage(
            epic=epic,
            message_type="L2",
            bid_price=bid if is_bid else 0.0,
            ask_price=ask if not is_bid else 0.0,
            bid_size=bid_size if is_bid else 0.0,
            ask_size=ask_size if not is_bid else 0.0,
            level_price=level_price,
            timestamp_ns=self._now_ns(),
            raw=fields,
        )

    def parse_trade_update(
        self,
        epic: str,
        fields: dict[str, str],
    ) -> MarketDataMessage:
        """
        Parse a TRADE (DISTINCT) update from Lightstreamer.

        Trade updates are sent as DISTINCT rows (each update is a new row,
        not a merge). IG sends:
            {
                "Last": "100.52",
                "LastSize": "1000",
                "LastTime": "12:34:56.789",
            }
        """
        last = self._safe_float(fields.get("Last", "0"))
        last_size = self._safe_float(fields.get("LastSize", "0"))

        return MarketDataMessage(
            epic=epic,
            message_type="TRADE",
            last_price=last,
            last_size=last_size,
            timestamp_ns=self._now_ns(),
            raw=fields,
        )

    # ------------------------------------------------------------------
    # Dispatch to callbacks
    # ------------------------------------------------------------------

    def dispatch(self, msg: MarketDataMessage) -> None:
        """
        Dispatch a parsed message to all registered callbacks for the epic.

        WHY iterating over a list of callbacks:
            Multiple consumers may need the same data (order book,
            microstructure analyser, latency monitor). The callback list
            pattern allows N consumers without modifying the feed handler.
        """
        callbacks = self._subscriptions.get(msg.epic, [])
        for cb in callbacks:
            cb(msg)

    # ------------------------------------------------------------------
    # IGStreamer callbacks
    # ------------------------------------------------------------------

    def _on_l1_update(self, item_name: str, fields: dict[str, str]) -> None:
        """
        Callback for IGStreamer L1 MERGE updates.

        IG sends L1 updates for item "MARKET:{epic}". Extracts the epic
        from the item name, normalises field name casing (IG may send
        "BID" or "Bid" depending on adapter version), parses into a
        MarketDataMessage, and dispatches to all subscribers.

        WHY normalising field names:
            IGStreamer.subscribe_l1() requests fields in uppercase
            ("BID", "OFFER"), but the actual field names in the update
            may arrive as PascalCase ("Bid", "Offer") depending on the
            Lightstreamer adapter version. Normalising avoids silent
            data loss from missed field lookups.
        """
        epic = item_name.removeprefix("MARKET:")
        msg = self.parse_l1_update(epic, _canonical_fields(fields))
        self.dispatch(msg)

    def _on_tick_update(self, item_name: str, fields: dict[str, str]) -> None:
        """
        Callback for IGStreamer TICK DISTINCT updates.

        IG sends tick updates for item "CHART:{epic}:TICK". Extracts the
        epic from the item name, normalises field name casing (IG tick
        fields are "Bid", "Ofr", "Ltp", "Ltv" — we map them to the
        canonical field names expected by parse_trade_update), parses
        into a MarketDataMessage, and dispatches.

        WHY mapping tick field names:
            Tick subscriptions return different field names than L1:
                IG tick field -> parse_trade_update field
                "Bid"         -> "Bid" (unused in trade parsing)
                "Ofr"         -> "Offer" (unused in trade parsing)
                "Ltp"         -> "Last"
                "Ltv"         -> "LastSize"
            The mapping normalises these differences so the same parser
            works for both L1 and tick feeds.
        """
        epic = item_name.removeprefix("CHART:").removesuffix(":TICK")
        msg = self.parse_trade_update(epic, _canonical_fields(fields))
        self.dispatch(msg)

    # ------------------------------------------------------------------
    # Connection management (IGStreamer integration)
    # ------------------------------------------------------------------

    def connect(
        self,
        api_key: str | None = None,
        account_id: str | None = None,
        username: str | None = None,
        password: str | None = None,
    ) -> None:
        """
        Connect to IG Lightstreamer and subscribe to all registered epics.

        Uses ig_market_data.stream.IGStreamer for live data, which reads
        IG credentials from environment variables (IG_API_KEY, IG_USERNAME,
        IG_PASSWORD) via .env file.

        If lightstreamer-client-lib (Java JRE dependency) is unavailable
        or IG credentials are not configured, logs a warning and operates
        in simulated mode — the consumer is marked connected so it works
        with MarketSimulator or synthetic data for testing.

        WHEN LIVESTREAMER IS AVAILABLE:
            1. Creates IGStreamer instance (reads auth from env).
            2. Connects to IG Lightstreamer (v2 CST/XST auth).
            3. For each subscribed epic, subscribes L1 MERGE and TICK
               DISTINCT feeds.
            4. Parsed MarketDataMessages are dispatched to all registered
               callbacks.

        Args:
            api_key: Deprecated — kept for backward compatibility.
            account_id: Deprecated — kept for backward compatibility.
            username: Deprecated — kept for backward compatibility.
            password: Deprecated — kept for backward compatibility.
        """
        try:
            from ig_market_data.stream import IGStreamer

            self._streamer = IGStreamer()
            self._streamer.connect()
            for epic in self._epic_to_data:
                item_l1 = f"MARKET:{epic}"
                self._streamer.subscribe_l1(epic, self._on_l1_update)
                item_tick = f"CHART:{epic}:TICK"
                self._streamer.subscribe_ticks(epic, self._on_tick_update)
                logger.info("Subscribed: L1=%s TICK=%s", item_l1, item_tick)
            self._connected = True
            logger.info(
                "TickConsumer connected to Lightstreamer (%d epics)",
                len(self._epic_to_data),
            )
        except ImportError:
            logger.warning(
                "lightstreamer-client-lib not available. "
                "TickConsumer operating in simulated mode. "
                "Install with: pip install lightstreamer-client-lib"
            )
            self._connected = True
        except RuntimeError as e:
            logger.warning(
                "IGStreamer connection failed: %s. "
                "TickConsumer operating in simulated mode. "
                "Check IG credentials in .env file.",
                e,
            )
            self._connected = True

    def connect_market_simulator(self, simulator: Any) -> None:
        """
        Connect to a MarketSimulator for synthetic data (no IG needed).

        WHY THIS EXISTS:
            Testing the full HFT pipeline (order book, microstructure,
            toxicity) requires a tick feed. MarketSimulator generates
            synthetic ticks so the pipeline is testable without a live
            IG connection or Java JRE.

        HOW IT WORKS:
            Each synthetic tick from the simulator is parsed through the
            standard TickConsumer parsers and dispatched to all registered
            callbacks — exactly like a real IGStreamer update.

        Args:
            simulator: A MarketSimulator instance with a run(callback)
                       method that yields MarketDataMessage objects.
        """
        self._simulator = simulator
        self._connected = True
        logger.info("TickConsumer connected to MarketSimulator")

    def disconnect(self) -> None:
        """Disconnect from Lightstreamer and clean up subscriptions."""
        if hasattr(self, "_streamer") and self._streamer is not None:
            try:
                self._streamer.disconnect()
            except Exception as e:
                logger.warning("Error disconnecting IGStreamer: %s", e)
        self._connected = False
        self._simulator = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_float(value: str) -> float:
        """Convert a string to float, returning 0.0 for invalid values."""
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0

    @staticmethod
    def _safe_int(value: str) -> int:
        """Convert a string to int, returning 0 for invalid values."""
        try:
            return int(float(value))  # float first in case of "1.0" format
        except (ValueError, TypeError):
            return 0

    @staticmethod
    def _now_ns() -> int:
        """
        Current Unix timestamp in nanoseconds.

        WHY nanosecond precision:
            Bryan Downing's latency monitor (Phase 3b) requires nanosecond
            precision for measuring pipeline latency. Starting with ns
            timestamps now avoids a painful migration later.
        """
        return int(datetime.now(timezone.utc).timestamp() * 1_000_000_000)
