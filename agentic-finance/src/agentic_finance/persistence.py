"""
persistence.py — durable state for orders, fills, risk and the audit trail.

WHY THIS EXISTS
    Phase 5-7 state lived in memory. A restart lost the day's fills, positions,
    realised P&L, the halt flag and the audit trail — which makes "paper trading"
    a demo, not an accounting record. This is the smallest thing that fixes it:
    SQLite from the standard library, one file, append-only rows, no new
    dependency and no server.

WHAT IS STORED
    orders        identity and lifecycle state
    fills         the source of truth for cash and positions
    audit         every recorded decision (JSON, lossless enough to rebuild)
    risk_state    a single row: halted flag, day-start equity, peak, daily P&L
    runs          experiment/replay summaries
    equity_curve  optional marks per run, for VaR and reporting

WHAT IS NOT
    Marks (prices) are transient; positions are rebuilt from fills and valued
    against fresh marks on restart. Order/fill identifiers are preserved, so a
    restored book reconciles to the same numbers as the original.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterable

_SCHEMA = """
CREATE TABLE IF NOT EXISTS orders (
    order_id TEXT PRIMARY KEY,
    epic TEXT NOT NULL,
    side TEXT NOT NULL,
    size REAL NOT NULL,
    order_type TEXT NOT NULL,
    status TEXT NOT NULL,
    filled_size REAL NOT NULL DEFAULT 0,
    avg_fill_price REAL,
    stop_loss REAL,
    take_profit REAL,
    reason TEXT,
    reasoning TEXT,
    created_at TEXT
);
CREATE TABLE IF NOT EXISTS fills (
    fill_id TEXT PRIMARY KEY,
    order_id TEXT,
    epic TEXT NOT NULL,
    side TEXT NOT NULL,
    quantity REAL NOT NULL,
    price REAL NOT NULL,
    reference_price REAL NOT NULL,
    commission REAL NOT NULL,
    timestamp TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS audit (
    decision_id TEXT PRIMARY KEY,
    timestamp TEXT,
    agent TEXT,
    decision_type TEXT,
    status TEXT,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS risk_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    halted INTEGER NOT NULL,
    daily_start REAL NOT NULL,
    peak REAL NOT NULL,
    current_capital REAL NOT NULL,
    daily_pnl REAL NOT NULL,
    last_reset_date TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS equity_curve (
    run_id TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    equity REAL NOT NULL,
    PRIMARY KEY (run_id, sequence)
);
"""


class StateStore:
    """A single SQLite file holding the paper-trading state."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock, self._conn:
            self._conn.executescript(_SCHEMA)

    # ------------------------------------------------------------------
    # Orders and fills
    # ------------------------------------------------------------------

    def save_order(self, order: Any) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """INSERT OR REPLACE INTO orders
                   (order_id, epic, side, size, order_type, status, filled_size,
                    avg_fill_price, stop_loss, take_profit, reason, reasoning, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    order.order_id, order.epic, order.side, order.size,
                    order.order_type, order.status, order.filled_size,
                    order.avg_fill_price, order.stop_loss, order.take_profit,
                    order.reason, order.reasoning,
                    order.timestamp.isoformat() if order.timestamp else None,
                ),
            )

    def load_orders(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM orders ORDER BY rowid"
            ).fetchall()
        return [dict(row) for row in rows]

    def save_fill(self, fill: Any) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """INSERT OR REPLACE INTO fills
                   (fill_id, order_id, epic, side, quantity, price,
                    reference_price, commission, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    fill.fill_id, fill.order_id, fill.epic, fill.side,
                    fill.quantity, fill.price, fill.reference_price,
                    fill.commission, fill.timestamp.isoformat(),
                ),
            )

    def load_fills(self) -> list[Any]:
        from datetime import datetime

        from agentic_finance.execution_ledger import Fill

        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM fills ORDER BY rowid"
            ).fetchall()
        return [
            Fill(
                fill_id=row["fill_id"],
                order_id=row["order_id"] or "",
                epic=row["epic"],
                side=row["side"],
                quantity=row["quantity"],
                price=row["price"],
                reference_price=row["reference_price"],
                commission=row["commission"],
                timestamp=datetime.fromisoformat(row["timestamp"]),
            )
            for row in rows
        ]

    # ------------------------------------------------------------------
    # Audit
    # ------------------------------------------------------------------

    def save_audit_entry(self, entry: Any) -> None:
        decision = entry.decision
        with self._lock, self._conn:
            self._conn.execute(
                """INSERT OR REPLACE INTO audit
                   (decision_id, timestamp, agent, decision_type, status, payload)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    decision.decision_id,
                    decision.timestamp.isoformat(),
                    decision.agent,
                    decision.decision_type.value,
                    decision.status.value,
                    json.dumps(entry.to_dict(), default=str),
                ),
            )

    def load_audit_entries(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT payload FROM audit ORDER BY rowid"
            ).fetchall()
        return [json.loads(row["payload"]) for row in rows]

    # ------------------------------------------------------------------
    # Risk state
    # ------------------------------------------------------------------

    def save_risk_state(self, state: dict[str, Any]) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """INSERT OR REPLACE INTO risk_state
                   (id, halted, daily_start, peak, current_capital, daily_pnl,
                    last_reset_date)
                   VALUES (1, ?, ?, ?, ?, ?, ?)""",
                (
                    int(bool(state["halted"])),
                    float(state["daily_start"]),
                    float(state["peak"]),
                    float(state["current_capital"]),
                    float(state["daily_pnl"]),
                    str(state["last_reset_date"]),
                ),
            )

    def load_risk_state(self) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM risk_state WHERE id = 1").fetchone()
        if row is None:
            return None
        return {
            "halted": bool(row["halted"]),
            "daily_start": row["daily_start"],
            "peak": row["peak"],
            "current_capital": row["current_capital"],
            "daily_pnl": row["daily_pnl"],
            "last_reset_date": row["last_reset_date"],
        }

    # ------------------------------------------------------------------
    # Runs and equity
    # ------------------------------------------------------------------

    def save_run(self, run_id: str, payload: dict[str, Any]) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO runs (run_id, payload) VALUES (?, ?)",
                (run_id, json.dumps(payload, indent=2, default=str)),
            )

    def load_run(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT payload FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        return json.loads(row["payload"]) if row else None

    def append_equity(self, run_id: str, sequence: int, equity: float) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO equity_curve (run_id, sequence, equity) VALUES (?, ?, ?)",
                (run_id, sequence, float(equity)),
            )

    def load_equity_curve(self, run_id: str) -> list[float]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT equity FROM equity_curve WHERE run_id = ? ORDER BY sequence",
                (run_id,),
            ).fetchall()
        return [row["equity"] for row in rows]

    def counts(self) -> dict[str, int]:
        with self._lock:
            return {
                table: self._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table in ("orders", "fills", "audit", "runs", "equity_curve")
            }

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def default_state_path() -> str:
    import os

    return os.environ.get("QUANT_STATE_DB", "state/quant.db")


def default_store() -> StateStore:
    return StateStore(default_state_path())
