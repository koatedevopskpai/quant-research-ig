"""
risk_manager.py — Risk Manager (Bryan Downing's Risk Framework)

The Risk Manager monitors portfolio exposure, computes VaR, enforces
drawdown limits, and can trigger a kill switch — halting all trading.

Three VaR methods:
    1. Historical — empirical percentile of past returns
    2. Parametric — assumes normal distribution
    3. Monte Carlo — simulated price paths

The kill switch is triggered when:
    - Portfolio drawdown exceeds max_drawdown_limit
    - Daily P&L exceeds daily_loss_limit
    - Portfolio VaR exceeds VaR_limit
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Optional
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


class RiskManager:
    def __init__(self, message_bus: MessageBus, audit_trail: AuditTrail,
                 initial_capital: float = 100_000.0,
                 max_drawdown_limit: float = 0.15,
                 daily_loss_limit: float = 0.03,
                 var_limit: float = 0.02,
                 var_confidence: float = 0.95):
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
        self._daily_start_capital = initial_capital
        self._positions: dict[str, float] = {}
        self._returns_history: list[float] = []
        self._active = True

    def update_positions(self, positions: dict[str, float]) -> None:
        self._positions = positions

    def record_pnl(self, pnl: float) -> None:
        self.daily_pnl += pnl
        self._returns_history.append(pnl / self.current_capital if self.current_capital > 0 else 0.0)
        self.current_capital += pnl
        self.peak_capital = max(self.peak_capital, self.current_capital)

    def current_drawdown(self) -> float:
        return (self.peak_capital - self.current_capital) / self.peak_capital if self.peak_capital > 0 else 0.0

    def total_exposure(self) -> float:
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
        var_h = self._historical_var()
        var_p = self._parametric_var()
        var_mc = self._monte_carlo_var()
        dd = self.current_drawdown()
        exp = self.total_exposure()
        n_pos = len(self._positions)
        breaches = []
        if dd > self.max_drawdown_limit:
            breaches.append(f"drawdown={dd:.2%} > limit={self.max_drawdown_limit:.2%}")
        if self.daily_pnl < -self.daily_loss_limit * self.current_capital:
            breaches.append(f"daily_loss={self.daily_pnl:.0f} > limit={self.daily_loss_limit:.2%}")
        if var_p < -self.var_limit:
            breaches.append(f"VaR={var_p:.2%} > limit={self.var_limit:.2%}")
        return RiskMetrics(
            var_historical=var_h,
            var_parametric=var_p,
            var_monte_carlo=var_mc,
            current_drawdown=dd,
            daily_pnl=self.daily_pnl,
            total_exposure=exp,
            n_positions=n_pos,
            is_breached=len(breaches) > 0,
            breaches=breaches,
        )

    def check_risk(self) -> RiskMetrics:
        metrics = self.compute_metrics()
        d = Decision.now(
            agent="RiskManager",
            decision_type=DecisionType.RISK_CHECK,
            description=f"Risk check: drawdown={metrics.current_drawdown:.2%}, "
                        f"VaR_95={metrics.var_parametric:.4f}, "
                        f"positions={metrics.n_positions}",
            context={"var_historical": metrics.var_historical,
                     "var_parametric": metrics.var_parametric,
                     "drawdown": metrics.current_drawdown,
                     "daily_pnl": metrics.daily_pnl,
                     "exposure": metrics.total_exposure},
            reasoning="; ".join(metrics.breaches) if metrics.breaches else "All limits OK",
        )
        d.status = DecisionStatus.APPROVED if not metrics.is_breached else DecisionStatus.REJECTED
        self.audit_trail.record(d, tags=["risk", "check"])
        if metrics.is_breached:
            self.message_bus.publish_topic(
                "risk.alert", "RiskManager",
                {"breaches": metrics.breaches, "metrics": {
                    "var_historical": metrics.var_historical,
                    "drawdown": metrics.current_drawdown,
                    "daily_pnl": metrics.daily_pnl,
                }},
            )
        return metrics

    def kill_if_needed(self) -> bool:
        metrics = self.check_risk()
        if metrics.is_breached:
            decision = Decision.now(
                agent="RiskManager",
                decision_type=DecisionType.KILL_SWITCH,
                description="Kill switch triggered by risk breach",
                context={"breaches": metrics.breaches},
                reasoning="; ".join(metrics.breaches),
            )
            decision.status = DecisionStatus.EXECUTED
            self.audit_trail.record(decision, tags=["risk", "kill_switch"])
            self.message_bus.publish_topic(
                "risk.kill", "RiskManager",
                {"reason": metrics.breaches, "metrics": {
                    "drawdown": metrics.current_drawdown,
                    "daily_pnl": metrics.daily_pnl,
                }},
            )
            self._active = False
            return True
        return False
