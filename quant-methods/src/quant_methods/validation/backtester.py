"""
validation/backtester.py — Vectorized Backtester

Core engine for Quant Beckman's Research→Know→Do→Get cycle. Evaluates
trading strategies by computing performance metrics from price series
and signal vectors. All operations are vectorised (pandas/numpy) for
speed — no loop-over-rows.

The backtester supports:
    - Equal-weighted, volatility-target, and signal-weighted position sizing
    - Benchmark-relative performance (alpha, beta, TE, IR)
    - Both simple and log returns
    - Transaction cost deductibility
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BacktestResult:
    total_return: float
    annualised_return: float
    annualised_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_holding_period: float
    alpha: Optional[float] = None
    beta: Optional[float] = None
    tracking_error: Optional[float] = None
    information_ratio: Optional[float] = None
    returns: Optional[pd.Series] = None
    equity_curve: Optional[pd.Series] = None
    drawdown_series: Optional[pd.Series] = None


class Backtester:
    def __init__(self, risk_free_rate: float = 0.0, periods_per_year: int = 252,
                 transaction_cost_bps: float = 0.0):
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year
        self.transaction_cost_bps = transaction_cost_bps

    def _compute_returns(self, prices: pd.Series, signal: pd.Series,
                         weight_scheme: str = "signal") -> pd.Series:
        aligned = pd.concat([prices, signal], axis=1).dropna()
        aligned.columns = ["price", "signal"]
        pct_ret = aligned["price"].pct_change().fillna(0.0)
        if weight_scheme == "equal":
            positions = pd.Series(1.0, index=aligned.index)
        elif weight_scheme == "signal":
            positions = aligned["signal"].clip(-1.0, 1.0)
        elif weight_scheme == "vol_target":
            # Warm-up stays flat. Full-sample fallback volatility leaks future returns.
            vol = pct_ret.rolling(20, min_periods=20).std()
            target = 0.15 / np.sqrt(self.periods_per_year)
            positions = (aligned["signal"].clip(-1.0, 1.0) * target
                         / vol.where(vol > 0)).fillna(0.0)
        else:
            raise ValueError(f"Unknown weight_scheme: {weight_scheme}")
        strategy_ret = pct_ret * positions.shift(1).fillna(0.0)
        if self.transaction_cost_bps > 0.0:
            turnover = positions.diff().abs().fillna(0.0)
            cost = turnover * self.transaction_cost_bps / 10000
            strategy_ret -= cost
        return strategy_ret

    def run(self, prices: pd.Series, signal: pd.Series,
            benchmark: Optional[pd.Series] = None,
            weight_scheme: str = "signal") -> BacktestResult:
        strategy_ret = self._compute_returns(prices, signal, weight_scheme)
        eq = (1.0 + strategy_ret).cumprod()
        dd = eq / eq.cummax() - 1.0
        n = len(strategy_ret)
        ny = self.periods_per_year
        total_ret = eq.iloc[-1] / eq.iloc[0] - 1.0 if len(eq) > 0 else 0.0
        ann_ret = (1.0 + total_ret) ** (ny / n) - 1.0 if n > 0 else 0.0
        ann_vol = strategy_ret.std() * np.sqrt(ny) if n > 0 else 0.0
        excess = strategy_ret - self.risk_free_rate / ny
        sharpe = (excess.mean() / excess.std() * np.sqrt(ny)) if excess.std() > 0 else 0.0
        downside = strategy_ret[strategy_ret < 0].std() * np.sqrt(ny) if n > 0 else 1e-10
        sortino = (ann_ret - self.risk_free_rate) / downside if downside > 0 else 0.0
        mdd = dd.min() if len(dd) > 0 else 0.0
        calmar = ann_ret / abs(mdd) if mdd < 0 else 0.0
        dd_dur = 0
        current_dur = 0
        for v in dd:
            if v < 0:
                current_dur += 1
                dd_dur = max(dd_dur, current_dur)
            else:
                current_dur = 0
        trades = strategy_ret[strategy_ret != 0]
        total_trades = max(1, len(trades))
        win_rate = (trades > 0).sum() / total_trades if len(trades) > 0 else 0.0
        profit_factor = (trades[trades > 0].sum() / abs(trades[trades < 0].sum())
                         if (trades < 0).sum() > 0 and trades[trades < 0].sum() != 0
                         else float('inf'))
        avg_hold = n / total_trades if total_trades > 0 else 0.0

        alpha = beta = te = ir = None
        if benchmark is not None:
            bm_ret = benchmark.pct_change().fillna(0.0)
            aligned = pd.concat([strategy_ret, bm_ret], axis=1).dropna()
            if len(aligned) > 1:
                cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])
                beta_val = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 0.0
                beta = beta_val
                alpha = (aligned.iloc[:, 0].mean() - beta * aligned.iloc[:, 1].mean()) * ny
                te = (aligned.iloc[:, 0] - beta * aligned.iloc[:, 1]).std() * np.sqrt(ny)
                ir = alpha / te if te and te > 0 else 0.0

        return BacktestResult(
            total_return=total_ret,
            annualised_return=ann_ret,
            annualised_volatility=ann_vol,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown=mdd,
            max_drawdown_duration=dd_dur,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=total_trades,
            avg_holding_period=avg_hold,
            alpha=alpha, beta=beta,
            tracking_error=te, information_ratio=ir,
            returns=strategy_ret, equity_curve=eq, drawdown_series=dd,
        )
