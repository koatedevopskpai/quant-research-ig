"""
analyser.py — Market microstructure analytics engine.

WHY MICROSTRUCTURE MATTERS:
    Market microstructure is the study of how trading mechanisms affect
    price formation. For systematic trading, microstructure analytics
    provide:

    1. **Spread analysis** — The bid-ask spread is the implicit cost of
       trading. Wide spreads mean higher transaction costs and lower
       expected returns.

    2. **Depth profiles** — The volume available at each price level
       reveals support/resistance zones and potential order flow toxicity
       (spoofing, layering).

    3. **Trade direction** — Classifying trades as aggressive buys vs.
       sells (Lee-Ready algorithm) reveals order flow imbalance — who
       is driving the market.

    4. **Volume imbalance** — (bid_volume - ask_volume) / total_volume
       is a real-time signal of buying vs. selling pressure.

BRYAN DOWNING'S APPROACH:
    Downing uses microstructure analytics to detect "order flow toxicity" —
    situations where market makers are systematically on the wrong side of
    informed trades. His Hawkes process model for toxicity detection will
    be implemented in Phase 3b (hft-engine Advanced).

QUANT BECKMAN'S APPROACH:
    Beckman uses microstructure features as inputs to regime detection
    models (HMM, clustering) and as validation metrics for backtesting
    — a strategy that works in simulation but fails in live trading is
    often killed by microstructure costs that the backtest ignored.

REFERENCES:
    Lee, C. & Ready, M. (1991). Inferring Trade Direction from Intraday
        Data. Journal of Finance, 46(2).
    Harris, L. (2003). Trading & Exchanges. Oxford University Press.
    Downing, B. (2025). Order Flow Toxicity Detection. QuantLabsNet.
"""

from dataclasses import dataclass, field
from math import sqrt
from typing import Literal

from hft_engine.book.order_book import OrderBook, BookDepth
from hft_engine.feeds.tick_consumer import MarketDataMessage


@dataclass
class SpreadMetrics:
    """
    Bid-ask spread statistics over a window of observations.

    WHY these specific statistics:
        - mean: Average transaction cost per trade.
        - std: Spread volatility — high std means erratic liquidity.
        - min/max: Bounds for setting limit order prices.
        - relative: Spread as a fraction of mid-price. Normalises across
          instruments with different price levels (e.g., US 500 ~5000 vs
          Apple ~200).

    INTERPRETATION:
        A relative spread of 0.0005 (0.05%) means it costs 5bps to
        round-trip. For a strategy targeting 10bps per trade, this is
        50% transaction cost — likely unprofitable.
    """

    mean: float = 0.0
    std: float = 0.0
    min: float = 0.0
    max: float = 0.0
    relative: float = 0.0
    count: int = 0


@dataclass
class DepthProfile:
    """
    Market depth profile at a point in time.

    Depth profile reveals:
        - Liquidity clusters (large bid/ask walls).
        - Support/resistance levels (concentrated volume).
        - Spoofing patterns (large orders that appear and disappear).

    Attributes:
        bid_levels: List of (price, size) tuples for bid side.
        ask_levels: List of (price, size) tuples for ask side.
        total_bid_volume: Cumulative bid volume across all levels.
        total_ask_volume: Cumulative ask volume across all levels.
        imbalance: (bid_vol - ask_vol) / (bid_vol + ask_vol).
        weighted_mid: Volume-weighted mid-price. Deviations from the
                      simple mid-price indicate order flow imbalance.
    """

    bid_levels: list[tuple[float, float]] = field(default_factory=list)
    ask_levels: list[tuple[float, float]] = field(default_factory=list)
    total_bid_volume: float = 0.0
    total_ask_volume: float = 0.0
    imbalance: float = 0.0
    weighted_mid: float = 0.0


@dataclass
class TradeSignResult:
    """
    Result of trade direction classification.

    The Lee-Ready algorithm classifies every trade as:
        - "aggressive_buy": Trade at or above the ask (buyside pressure).
        - "aggressive_sell": Trade at or below the bid (sellside pressure).
        - "neutral": Trade inside the spread (ambiguous).

    WHY this matters:
        Cumulative aggressive buy/sell volume is a leading indicator of
        short-term price direction. A sequence of aggressive buys suggests
        upward momentum; aggressive sells suggest downward momentum.

    Attributes:
        side: "buy", "sell", or "neutral".
        price: Trade price.
        size: Trade volume.
        confidence: How certain the classification is. 1.0 for trades at
                    the best bid/ask, lower for trades inside the spread.
    """

    side: Literal["buy", "sell", "neutral"]
    price: float
    size: float
    confidence: float = 1.0


class MicrostructureAnalyser:
    """
    Market microstructure analytics engine.

    Consumes MarketDataMessages and produces spread, depth, and trade
    direction analytics. This class is the "brain" that turns raw tick
    data into actionable trading signals.

    USAGE:
        book = OrderBook()
        analyser = MicrostructureAnalyser()

        def on_tick(msg: MarketDataMessage) -> None:
            # Update the order book
            if msg.message_type == "L2" and msg.level_price > 0:
                is_bid = msg.bid_price > 0
                side = "bid" if is_bid else "ask"
                size = msg.bid_size if is_bid else msg.ask_size
                book.apply_diff(side, msg.level_price, size)

            # Feed the analyser
            analyser.on_tick(msg, book)

            # Query analytics
            spread = analyser.spread_stats()
            depth = analyser.current_depth(book)

        # Subscribe to live data
        consumer = TickConsumer()
        consumer.subscribe("IX.D.FTSE.IFP.IP", on_tick)

    WINDOW-BASED ANALYTICS:
        Spread statistics are computed over a rolling window of N
        observations (default 100). This allows the analyser to adapt
        to changing market conditions while smoothing out noise.
    """

    def __init__(self, window_size: int = 100) -> None:
        self._window_size = window_size
        # Rolling window of spreads (bid-ask differences)
        self._spreads: list[float] = []
        # Rolling window of relative spreads (spread / mid_price)
        self._relative_spreads: list[float] = []

    # ------------------------------------------------------------------
    # Tick processing
    # ------------------------------------------------------------------

    def on_tick(self, msg: MarketDataMessage, book: OrderBook | None = None) -> None:
        """
        Process an incoming market data message.

        Updates rolling analytics windows based on the message type.

        Args:
            msg: Parsed market data message from the feed handler.
            book: Optional order book for context (needed for quote-based
                  analytics like depth profiles).
        """
        if msg.message_type == "L1":
            self._record_spread(msg)

    # ------------------------------------------------------------------
    # Spread analytics
    # ------------------------------------------------------------------

    def _record_spread(self, msg: MarketDataMessage) -> None:
        """Record the bid-ask spread from an L1 message."""
        if msg.ask_price <= 0 or msg.bid_price <= 0:
            return

        spread = msg.ask_price - msg.bid_price
        mid = (msg.ask_price + msg.bid_price) / 2
        rel_spread = spread / mid if mid > 0 else 0.0

        self._spreads.append(spread)
        self._relative_spreads.append(rel_spread)

        # Trim window
        if len(self._spreads) > self._window_size:
            self._spreads.pop(0)
            self._relative_spreads.pop(0)

    def spread_stats(self) -> SpreadMetrics:
        """
        Compute spread statistics over the current observation window.

        Returns:
            SpreadMetrics with mean, std, min, max, and relative spread.
        """
        if not self._spreads:
            return SpreadMetrics()

        n = len(self._spreads)
        mean = sum(self._spreads) / n
        variance = sum((s - mean) ** 2 for s in self._spreads) / (n - 1) if n > 1 else 0.0
        std = sqrt(variance)
        rel_mean = sum(self._relative_spreads) / n

        return SpreadMetrics(
            mean=round(mean, 4),
            std=round(std, 4),
            min=round(min(self._spreads), 4),
            max=round(max(self._spreads), 4),
            relative=round(rel_mean, 6),
            count=n,
        )

    # ------------------------------------------------------------------
    # Depth profile
    # ------------------------------------------------------------------

    @staticmethod
    def current_depth(
        book: OrderBook,
        max_levels: int = 10,
    ) -> DepthProfile:
        """
        Compute the current market depth profile from the order book.

        The depth profile shows the volume available at each price level
        on both sides of the book.

        WHY the weighted mid-price:
            The simple mid-price (best_bid + best_ask / 2) ignores volume.
            The weighted mid-price accounts for volume at all levels:
                weighted_mid = sum(price_i * size_i) / sum(size_i)

            When weighted_mid is above simple mid, there's more volume on
            the ask side — selling pressure. Below = buying pressure.

        Args:
            book: The current order book state.
            max_levels: Number of price levels to include.

        Returns:
            DepthProfile with volume and imbalance metrics.
        """
        l2 = book.l2_snapshot(max_levels)
        depth = book.depth_summary(max_levels)

        bid_levels = [(lev.price, lev.size) for lev in l2.bids]
        ask_levels = [(lev.price, lev.size) for lev in l2.asks]

        # Weighted mid-price: volume-weighted average across all levels
        total_vol = depth.bid_volume + depth.ask_volume
        weighted_mid = 0.0
        if total_vol > 0:
            weighted_mid = (
                sum(p * s for p, s in bid_levels)
                + sum(p * s for p, s in ask_levels)
            ) / total_vol

        return DepthProfile(
            bid_levels=bid_levels,
            ask_levels=ask_levels,
            total_bid_volume=depth.bid_volume,
            total_ask_volume=depth.ask_volume,
            imbalance=depth.imbalance,
            weighted_mid=round(weighted_mid, 4),
        )

    # ------------------------------------------------------------------
    # Trade direction classification (Lee-Ready)
    # ------------------------------------------------------------------

    @staticmethod
    def classify_trade(
        trade_price: float,
        trade_size: float,
        book: OrderBook,
    ) -> TradeSignResult:
        """
        Classify a trade as aggressive buy, sell, or neutral.

        LEE-READY ALGORITHM (simplified):
            1. If trade price >= best ask: aggressive BUY.
            2. If trade price <= best bid: aggressive SELL.
            3. If trade price is inside the spread: compare to the
               mid-price. Above mid = buy, below mid = sell.

        WHY the simplification:
            The full Lee-Ready algorithm also considers quote updates
            within 5 seconds of the trade. We omit this for simplicity
            and latency. The simplified version is approximately 85%
            accurate for equities (Lee & Ready, 1991).

        Args:
            trade_price: The price at which the trade occurred.
            trade_size: The volume of the trade.
            book: The current order book state.

        Returns:
            TradeSignResult with side, price, size, and confidence.
        """
        try:
            l1 = book.l1_snapshot()
        except ValueError:
            return TradeSignResult(
                side="neutral", price=trade_price, size=trade_size, confidence=0.0
            )

        best_bid = l1.bid_price
        best_ask = l1.ask_price
        mid = (best_bid + best_ask) / 2

        if trade_price >= best_ask:
            confidence = 1.0 if trade_price == best_ask else 0.8
            return TradeSignResult(
                side="buy", price=trade_price, size=trade_size, confidence=confidence
            )
        elif trade_price <= best_bid:
            confidence = 1.0 if trade_price == best_bid else 0.8
            return TradeSignResult(
                side="sell", price=trade_price, size=trade_size, confidence=confidence
            )
        else:
            # Inside the spread — classify by position relative to mid
            if trade_price >= mid:
                return TradeSignResult(
                    side="buy",
                    price=trade_price,
                    size=trade_size,
                    confidence=0.5,
                )
            else:
                return TradeSignResult(
                    side="sell",
                    price=trade_price,
                    size=trade_size,
                    confidence=0.5,
                )
