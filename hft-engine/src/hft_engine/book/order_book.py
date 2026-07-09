"""
order_book.py — Limit order book data structures and diff application.

WHY A SEPARATE ORDER BOOK MODULE:
    Bryan Downing's HFT architecture centres on the limit order book as the
    "source of truth" for market state. Every trading decision — whether to
    enter, exit, or adjust a position — depends on the current state of the
    book. A 1-millisecond lag in book updates costs approximately 0.5-2bps
    in alpha slippage.

    This module implements an L2 order book that:
    1. Maintains sorted bid and ask price levels (bids descending, asks
       ascending).
    2. Applies incremental diffs (insert/update/delete) to the book.
    3. Provides O(1) access to the top of book (L1 snapshot).
    4. Provides O(K) access to the full depth (L2 snapshot, K levels).

CRITICAL INSIGHT (Bryan Downing):
    An order book is NOT a static snapshot — it is a continuous event stream.
    Every tick is a DIFF applied to the previous state. The book exists only
    ephemerally in memory and must be reconstructed from the feed. This is why
    the feed handler (feeds/) and the book (book/) are tightly coupled.

DATA MODEL:
    Each level in the book is a PriceLevel:
        - price: The price of this level (float).
        - size: The available volume at this price (float).
        - order_count: Number of resting orders at this level (int). Optional.

    The book maintains two sides:
        - bids: list[PriceLevel] sorted DESCENDING by price (highest bid first).
        - asks: list[PriceLevel] sorted ASCENDING by price (lowest ask first).

DIFF APPLICATION:
    Diffs arrive as PriceLevel updates from the exchange. Each diff has:
        - side: "bid" or "ask"
        - price: The price level
        - size: New size (0 = delete the level)
        - order_count: New order count (optional)

    The algorithm:
        1. Find the level in the sorted list (binary search by price).
        2. If size == 0: remove the level (delete).
        3. If level exists: update size and order_count.
        4. If level does not exist: insert in sorted position.

REFERENCES:
    Downing, B. (2025). Deconstructing an Advanced Ether High Frequency
        Trading Bot: A C++ Breakdown. QuantLabsNet.
    Harris, L. (2003). Trading & Exchanges: Market Microstructure for
        Practitioners. Oxford University Press.
"""

from dataclasses import dataclass, field
from typing import Literal


Side = Literal["bid", "ask"]


@dataclass(frozen=True, order=True)
class PriceLevel:
    """
    A single price level in the order book.

    WHY frozen=True and order=True:
        PriceLevels must be immutable once created — a level update is a
        replacement, not a mutation. Frozen dataclasses enforce this.
        Ordering by price enables bisect-based insertion.

        PriceLevels sort by price ASCENDING. For bids, we reverse the
        ordering in the sortedcontainers-like approach.

    Attributes:
        price: The price of this level.
        size: The available volume at this price (in contracts/shares).
        order_count: Number of individual orders at this level (if known).
    """

    price: float
    size: float
    order_count: int = 0


@dataclass(frozen=True)
class L1Snapshot:
    """
    Top-of-book (Level 1) snapshot.

    This is what most traders see — the best bid, best ask, and their sizes.
    L1 data is sufficient for many trading decisions but insufficient for
    understanding market depth and order flow toxicity.

    WHY spread and mid_price are computed here:
        These derived fields are the most commonly queried metrics. Computing
        them once at snapshot time avoids redundant calculations.
    """

    bid_price: float
    bid_size: float
    ask_price: float
    ask_size: float
    spread: float = 0.0
    mid_price: float = 0.0

    def __post_init__(self) -> None:
        # Use object.__setattr__ because frozen=True
        object.__setattr__(self, "spread", round(self.ask_price - self.bid_price, 4))
        object.__setattr__(
            self, "mid_price", round((self.bid_price + self.ask_price) / 2, 4)
        )


@dataclass(frozen=True)
class L2Snapshot:
    """
    Full depth (Level 2) snapshot with all price levels.

    L2 data reveals:
        - Support/resistance levels (large bid/ask clusters).
        - Order flow imbalance (volume-weighted bid/ask pressure).
        - Spoofing and layering patterns (Downing's toxicity detection).

    WHY keeping bid_depth and ask_depth as lists:
        The full depth is needed for microstructure analytics (depth profiles,
        volume imbalance). Truncating to N levels would lose information.
    """

    bids: list[PriceLevel] = field(default_factory=list)
    asks: list[PriceLevel] = field(default_factory=list)
    timestamp: float = 0.0  # Unix timestamp in seconds (with sub-second precision)


@dataclass(frozen=True)
class BookDepth:
    """
    Depth summary statistics computed from the current book.

    These metrics feed directly into microstructure analytics.
    """

    bid_volume: float = 0.0
    ask_volume: float = 0.0
    imbalance: float = 0.0  # (bid_vol - ask_vol) / (bid_vol + ask_vol)
    num_bid_levels: int = 0
    num_ask_levels: int = 0

    def __post_init__(self) -> None:
        if self.bid_volume + self.ask_volume > 0:
            object.__setattr__(
                self,
                "imbalance",
                round(
                    (self.bid_volume - self.ask_volume)
                    / (self.bid_volume + self.ask_volume),
                    4,
                ),
            )


class OrderBook:
    """
    L2 limit order book with diff application.

    Maintains bid and ask sides as sorted lists of PriceLevels.

    USAGE:
        book = OrderBook()

        # Apply a snapshot (initial state)
        book.apply_snapshot(bids=[...], asks=[...])

        # Apply incremental diffs as they arrive from the feed
        book.apply_diff("bid", 100.50, 5000)
        book.apply_diff("ask", 101.00, 0)  # delete level

        # Query state
        l1 = book.l1_snapshot()
        l2 = book.l2_snapshot()
        depth = book.depth_summary()

    PERFORMANCE CHARACTERISTICS:
        - apply_diff: O(log K + K) where K is the number of levels on a side.
          Binary search (O(log K)) to find position, then O(K) for list
          insertion/deletion.
        - l1_snapshot: O(1) — access first element of each side.
        - l2_snapshot: O(K) — copy all levels (O(K) memory).
        - depth_summary: O(K) — sum sizes across all levels.

        For an L2 book with ~20 levels per side (typical for IG indices),
        all operations complete in microseconds in pure Python.

    WHY PURE PYTHON (for now):
        The Python implementation serves as the reference and prototype.
        The hot path (diff application) will be ported to C++ via pybind11
        in the cpp/ directory. The Python version is adequate for backtesting
        and paper trading at IG's update frequencies (~100ms per tick).

    NOTE ON IG's DATA FORMAT:
        IG sends L2 data as price level updates via Lightstreamer.
        Each update has fields: Bid, Offer, BidSize, OfferSize.
        IG does NOT expose individual order counts, so order_count defaults
        to 0 and is not used in diff application.
    """

    def __init__(self) -> None:
        # WHY using lists with bisect instead of SortedList from bisect:
        #   Avoiding external dependencies. Python's built-in bisect module
        #   combined with list.insert/delete is O(K) for insertion but
        #   K ≤ 100 for typical L2 books, so the cost is negligible.
        self._bids: list[PriceLevel] = []
        self._asks: list[PriceLevel] = []

    # ------------------------------------------------------------------
    # Diff application
    # ------------------------------------------------------------------

    def apply_diff(
        self,
        side: Side,
        price: float,
        size: float,
        order_count: int = 0,
    ) -> None:
        """
        Apply a single price level update to the book.

        This is the CORE method — every tick from the feed handler calls
        this. Performance here determines the overall engine latency.

        Args:
            side: "bid" or "ask".
            price: The price level to update.
            size: The new size at this level. 0 = delete the level.
            order_count: The new order count (0 if unknown).

        WHY checking size == 0 for deletion:
            In IG's data model, a level with size=0 means "no volume at this
            price". This is the delete signal. Never trust that a feed will
            explicitly say "delete" — interpret zero as deletion.
        """
        level = PriceLevel(price, size, order_count)

        if side == "ask":
            # Asks are sorted ascending by price
            idx = self._find_price_index(self._asks, price, reverse=False)
            if size == 0:
                if idx is not None:
                    del self._asks[idx]
            else:
                if idx is not None:
                    self._asks[idx] = level  # update
                else:
                    # Insert in ascending sorted position
                    insert_idx = 0
                    for i, lev in enumerate(self._asks):
                        if price < lev.price:
                            insert_idx = i
                            break
                        insert_idx = i + 1
                    self._asks.insert(insert_idx, level)
        else:
            # Bids are sorted descending by price
            idx = self._find_price_index(self._bids, price, reverse=True)
            if size == 0:
                if idx is not None:
                    del self._bids[idx]
            else:
                if idx is not None:
                    self._bids[idx] = level  # update
                else:
                    # Insert in descending sorted position
                    insert_idx = 0
                    for i, lev in enumerate(self._bids):
                        if price > lev.price:
                            insert_idx = i
                            break
                        insert_idx = i + 1
                    self._bids.insert(insert_idx, level)

    def apply_snapshot(
        self,
        bids: list[PriceLevel],
        asks: list[PriceLevel],
    ) -> None:
        """
        Replace the entire book with a snapshot.

        Used for initial state or recovery after a connection drop.

        WHY replace vs. incremental:
            After a disconnection, the sequence of incremental diffs may
            have gaps. Requesting a full snapshot and replacing the book
            ensures consistency without reconciliation logic.
        """
        # Sort bids descending, asks ascending
        self._bids = sorted(bids, key=lambda x: x.price, reverse=True)
        self._asks = sorted(asks, key=lambda x: x.price)

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def l1_snapshot(self) -> L1Snapshot:
        """
        Get the top-of-book snapshot.

        Returns L1Snapshot with best bid, best ask, spread, and mid-price.

        Raises:
            ValueError: If the book is empty on either side.
        """
        if not self._bids or not self._asks:
            raise ValueError("Book is empty — no L1 snapshot available")

        best_bid = self._bids[0]
        best_ask = self._asks[0]
        return L1Snapshot(
            bid_price=best_bid.price,
            bid_size=best_bid.size,
            ask_price=best_ask.price,
            ask_size=best_ask.size,
        )

    def l2_snapshot(self, max_levels: int = 0) -> L2Snapshot:
        """
        Get a full-depth snapshot.

        Args:
            max_levels: Maximum levels per side to return. 0 = all levels.

        Returns:
            L2Snapshot with bid and ask price levels.
        """
        if max_levels > 0:
            return L2Snapshot(
                bids=list(self._bids[:max_levels]),
                asks=list(self._asks[:max_levels]),
            )
        return L2Snapshot(bids=list(self._bids), asks=list(self._asks))

    def depth_summary(self, max_levels: int = 0) -> BookDepth:
        """
        Compute depth statistics: total volume and imbalance.

        Volume imbalance is a key microstructure signal:
            imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol)

        Positive imbalance = more buying pressure (bullish).
        Negative imbalance = more selling pressure (bearish).
        Values range from -1 to +1.

        Quant Beckman uses imbalance as a feature in regime detection models.
        Bryan Downing uses it for order flow toxicity detection.
        """
        l2 = self.l2_snapshot(max_levels)
        bid_vol = sum(lev.size for lev in l2.bids)
        ask_vol = sum(lev.size for lev in l2.asks)
        return BookDepth(
            bid_volume=bid_vol,
            ask_volume=ask_vol,
            num_bid_levels=len(l2.bids),
            num_ask_levels=len(l2.asks),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_price_index(
        levels: list[PriceLevel],
        price: float,
        reverse: bool = False,
    ) -> int | None:
        """
        Find the index of a price level by price.

        Args:
            levels: Sorted list of PriceLevels.
            price: Price to search for.
            reverse: True if the list is sorted descending (bids).

        Returns:
            Index of the level, or None if not found.

        WHY linear search instead of binary search:
            L2 books typically have 10-50 levels per side. A linear scan
            over 50 elements is faster than binary search due to Python's
            function call overhead for the comparator. For books with 100+
            levels, this should be replaced with bisect.
        """
        if reverse:
            for i, lev in enumerate(levels):
                if lev.price == price:
                    return i
                if lev.price < price:
                    # Since bids are sorted descending, if we pass the
                    # target price, it won't appear later.
                    return None
        else:
            for i, lev in enumerate(levels):
                if lev.price == price:
                    return i
                if lev.price > price:
                    return None
        return None
