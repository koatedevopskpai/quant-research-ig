"""
risk_manager.py — Risk Manager (Bryan Downing's Risk Framework)

The Risk Manager monitors portfolio exposure, computes VaR, enforces drawdown and
daily-loss limits, and can halt trading.

WHAT CHANGED IN PHASE 6
    * P&L, exposure and capital now come from the execution ledger when one is
      attached. The previous runner booked `(mid_price - 100) * 10` on every
      refresh, regardless of holdings, and exposure was never wired at all.
    * The daily loss limit resets on a UTC date boundary and is measured against
      the equity at the start of that day.
    * Halting is a state change with an idempotent transition and a *notification*
      (`risk.halted` / `risk.resumed`). It used to publish `risk.kill`, the same
      topic its caller subscribed to, which re-entered the handler and recursed.
    * `can_trade()` is the single question the order paths ask: not halted and
      not breached.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from datetime import datetime, timezone

from agentic_finance.message_bus import MessageBus, AgentMessage
from agentic_finance.decision import Decision, DecisionType, DecisionStatus
from agentic_finance.audit_trail import AuditTrail


@dataclass
class RiskMetrics:
    var_historical: float
    var_parametric: float
    var_monte_carlo: float
    current_drawdown: float
    daily_pnl: float
    total_exposure: float
    n_positions: int
    is_breached: bool
    breaches: list[str] = field(default_factory=list)
    equity: float = 0.0
    exposure_pct: float = 0.0
    is_halted: bool = False


class RiskManager:
    def __init__(self, message_bus: MessageBus, audit_trail: AuditTrail,
                 initial_capital: float = 100_000.0,
                 max_drawdown_limit: float = 0.15,
                 daily_loss_limit: float = 0.03,
                 var_limit: float = 0.02,
                 var_confidence: float = 0.95,
                 clock: Optional[Callable[[], datetime]] = None):
        self.message_bus = message_bus
        self.audit_trail = audit_trail
        self.initial_capital = initial_capital
        self.peak_capital = initial_capital
        self.current_capital = initial_capital
        self.max_drawdown_limit = max_drawdown_limit
        self.daily_loss_limit = daily_loss_limit
        self.var_limit = var_limit
        self.var_confidence = var_confidence
        self.daily_pnl = 0.0

        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._daily_start_capital = initial_capital
        self._last_reset_date = self._clock().date()
        self._positions: dict[str, float] = {}
        self._returns_history: list[float] = []
        self._halted = False
        self._ledger: Any = None
        self._marks: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Ledger integration
    # ------------------------------------------------------------------

    def set_ledger(self, ledger: Any) -> None:
        """Use the execution ledger as the source of truth for P&L and exposure."""
        self._ledger = ledger
        self._refresh_from_ledger()

    def update_mark(self, epic: str, price: float) -> None:
        if price > 0:
            self._marks[epic] = float(price)

    def update_marks(self, marks: dict[str, float]) -> None:
        for epic, price in marks.items():
            self.update_mark(epic, price)

    def _refresh_from_ledger(self) -> None:
        if self._ledger is None:
            return
        equity = self._ledger.equity(self._marks)
        self.current_capital = equity
        self.peak_capital = max(self.peak_capital, equity)
        self.daily_pnl = equity - self._daily_start_capital
        self._positions = {
            epic: position.quantity
            for epic, position in self._ledger.open_positions().items()
        }

    def _reset_daily_if_needed(self) -> None:
        today = self._clock().date()
        if today == self._last_reset_date:
            return
        self._last_reset_date = today
        self._refresh_from_ledger()
        self._daily_start_capital = self.current_capital
        self.daily_pnl = 0.0

    # ------------------------------------------------------------------
    # Legacy manual P&L (kept for callers without a ledger / tests)
    # ------------------------------------------------------------------

    def update_positions(self, positions: dict[str, float]) -> None:
        self._positions = positions

    def record_pnl(self, pnl: float) -> None:
        if self._ledger is not None:
            # The ledger is authoritative; ad-hoc P&L would double count.
            self._refresh_from_ledger()
            return
        self.daily_pnl += pnl
        self._returns_history.append(
            pnl / self.current_capital if self.current_capital > 0 else 0.0
        )
        self.current_capital += pnl
        self.peak_capital = max(self.peak_capital, self.current_capital)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def current_drawdown(self) -> float:
        return ((self.peak_capital - self.current_capital) / self.peak_capital
                if self.peak_capital > 0 else 0.0)

    def total_exposure(self) -> float:
        if self._ledger is not None:
            return float(self._ledger.gross_exposure(self._marks))
        return sum(abs(v) for v in self._positions.values()) if self._positions else 0.0

    def _historical_var(self) -> float:
        if len(self._returns_history) < 20:
            return 0.0
        alpha = 1.0 - self.var_confidence
        return float(np.percentile(self._returns_history, alpha * 100))

    def _parametric_var(self) -> float:
        if len(self._returns_history) < 5:
            return 0.0
        mu = float(np.mean(self._returns_history))
        sigma = float(np.std(self._returns_history))
        if sigma <= 0:
            return 0.0
        from scipy import stats
        z = stats.norm.ppf(1.0 - self.var_confidence)
        return mu + z * sigma

    def _monte_carlo_var(self, n_simulations: int = 10000) -> float:
        if len(self._returns_history) < 5:
            return 0.0
        mu = float(np.mean(self._returns_history))
        sigma = float(np.std(self._returns_history))
        if sigma <= 0:
            return 0.0
        sim_returns = np.random.normal(mu, sigma, n_simulations)
        alpha = 1.0 - self.var_confidence
        return float(np.percentile(sim_returns, alpha * 100))

    def compute_metrics(self) -> RiskMetrics:
        self._reset_daily_if_needed()
        self._refresh_from_ledger()

        var_h = self._historical_var()
        var_p = self._parametric_var()
        var_mc = self._monte_carlo_var()
        dd = self.current_drawdown()
        exposure = self.total_exposure()
        equity = self.current_capital
        n_pos = len(self._positions)

        breaches: list[str] = []
        if dd > self.max_drawdown_limit:
            breaches.append(f"drawdown={dd:.2%} > limit={self.max_drawdown_limit:.2%}")
        if self.daily_pnl < -self.daily_loss_limit * self._daily_start_capital:
            breaches.append(
                f"daily_loss={self.daily_pnl:.2f} > limit="
                f"{self.daily_loss_limit:.2%} of {self._daily_start_capital:.0f}"
            )
        if var_p < -self.var_limit:
            breaches.append(f"VaR={var_p:.2%} > limit={self.var_limit:.2%}")

        return RiskMetrics(
            var_historical=var_h,
            var_parametric=var_p,
            var_monte_carlo=var_mc,
            current_drawdown=dd,
            daily_pnl=self.daily_pnl,
            total_exposure=exposure,
            n_positions=n_pos,
            is_breached=len(breaches) > 0,
            breaches=breaches,
            equity=equity,
            exposure_pct=(exposure / equity) if equity > 0 else 0.0,
            is_halted=self._halted,
        )

    def check_risk(self) -> RiskMetrics:
        metrics = self.compute_metrics()
        decision = Decision.now(
            agent="RiskManager",
            decision_type=DecisionType.RISK_CHECK,
            description=f"Risk check: drawdown={metrics.current_drawdown:.2%}, "
                        f"VaR_95={metrics.var_parametric:.4f}, "
                        f"positions={metrics.n_positions}",
            context={"var_historical": metrics.var_historical,
                     "var_parametric": metrics.var_parametric,
                     "drawdown": metrics.current_drawdown,
                     "daily_pnl": metrics.daily_pnl,
                     "exposure": metrics.total_exposure,
                     "equity": metrics.equity,
                     "exposure_pct": metrics.exposure_pct},
            reasoning="; ".join(metrics.breaches) if metrics.breaches else "All limits OK",
        )
        decision.status = (DecisionStatus.APPROVED if not metrics.is_breached
                           else DecisionStatus.REJECTED)
        self.audit_trail.record(decision, tags=["risk", "check"])
        if metrics.is_breached:
            self.message_bus.publish_topic(
                "risk.alert", "RiskManager",
                {"breaches": metrics.breaches, "metrics": {
                    "var_historical": metrics.var_historical,
                    "drawdown": metrics.current_drawdown,
                    "daily_pnl": metrics.daily_pnl,
                    "equity": metrics.equity,
                    "exposure_pct": metrics.exposure_pct,
                }},
            )
        return metrics

    # ------------------------------------------------------------------
    # Halt state
    # ------------------------------------------------------------------

    @property
    def is_halted(self) -> bool:
        return self._halted

    def can_trade(self) -> bool:
        """The question every order path must ask before creating risk."""
        if self._halted:
            return False
        return not self.compute_metrics().is_breached

    def halt(self, reason: str = "manual", metrics: RiskMetrics | None = None) -> bool:
        """Idempotent halt. Returns True only if this call changed the state."""
        if self._halted:
            return False
        self._halted = True
        decision = Decision.now(
            agent="RiskManager",
            decision_type=DecisionType.KILL_SWITCH,
            description=f"Trading halted: {reason}",
            context={"reason": reason,
                     "breaches": metrics.breaches if metrics else []},
            reasoning=reason,
        )
        decision.status = DecisionStatus.EXECUTED
        self.audit_trail.record(decision, tags=["risk", "kill_switch"])
        self.message_bus.publish_topic(
            "risk.halted", "RiskManager",
            {"reason": reason,
             "metrics": {
                 "drawdown": metrics.current_drawdown if metrics else 0.0,
                 "daily_pnl": metrics.daily_pnl if metrics else self.daily_pnl,
                 "equity": metrics.equity if metrics else self.current_capital,
             }},
        )
        return True

    def resume(self) -> bool:
        """Idempotent resume. Returns True only if this call changed the state."""
        if not self._halted:
            return False
        self._halted = False
        decision = Decision.now(
            agent="RiskManager",
            decision_type=DecisionType.KILL_SWITCH,
            description="Trading resumed",
            context={"active": True},
            reasoning="Manual resume",
        )
        decision.status = DecisionStatus.EXECUTED
        self.audit_trail.record(decision, tags=["risk", "resume"])
        self.message_bus.publish_topic("risk.resumed", "RiskManager", {})
        return True

    def kill_if_needed(self) -> bool:
        """Halt once if any limit is breached. Repeated calls are no-ops."""
        metrics = self.check_risk()
        if not metrics.is_breached:
            return False
        return self.halt("; ".join(metrics.breaches), metrics)
