# quant-methods — Quantitative Methods & Pricing Models

## Objective
Implement classical quant finance models: Black-Scholes, binomial trees,
Monte Carlo simulation, GARCH volatility modelling, and portfolio
optimisation — all backtested against real IG market data.

This is the **Phase 2a** deliverable in the STRATEGY.md roadmap. These models
convert raw IG market data into actionable trading signals: option prices,
volatility forecasts, and portfolio allocations. Every model includes heavy
narrative comments explaining WHY each design decision was made — not just
WHAT the code does.

## Projects (Completed — Phase 2a)
1. **Black-Scholes Pricer** — European option pricing + Greeks (delta, gamma,
   vega, theta, rho). Validated against Hull (2022) reference values.
2. **Binomial Tree** — CRR lattice for American/European options. Validated
   by convergence to BSM within 0.02 (N=500 steps).
3. **Monte Carlo Engine** — GBM simulation with standard, Asian (arithmetic
   average), and barrier (up-and-out, down-and-out) payoff functions.
4. **GARCH Volatility** — GARCH(1,1) model with gradient descent MLE.
   Forecasts future vol with mean-reversion diagnostics.
5. **Portfolio Optimisation** — Mean-variance (Markowitz 1952), risk parity
   (equal risk contribution), and Black-Litterman (Bayesian view blending).

## Key Module Map
| Module | Class | Purpose |
|--------|-------|---------|
| pricers/black_scholes.py | `BlackScholesPricer` | European options (closed-form) |
| pricers/binomial_tree.py | `BinomialTreePricer` | American/European via CRR lattice |
| pricers/monte_carlo.py | `MonteCarloPricer` | Path-dependent options via GBM MC |
| volatility/garch.py | `GARCH` | GARCH(1,1) fit + forecast |
| portfolio/mean_variance.py | `MeanVarianceOptimiser` | Min var, max Sharpe, efficient frontier |
| portfolio/risk_parity.py | `RiskParity` | Equal risk contribution portfolio |
| portfolio/black_litterman.py | `BlackLitterman` | Bayesian return blending |

## Status
- [x] Project skeleton (pyproject.toml, package structure)
- [x] Data dependency on ig-market-data established
- [x] Black-Scholes with Greeks (11 tests)
- [x] CRR Binomial Tree — European convergence, American early exercise (9 tests)
- [x] Monte Carlo Engine — standard, Asian, barrier payoffs (10 tests)
- [x] GARCH(1,1) — persistence, forecast, diagnostics (7 tests)
- [x] Portfolio — Markowitz, risk parity, Black-Litterman (18 tests)
- [x] **All 55 tests passing**

---

## Architecture Decisions

### ADR-001: Closed-Form BSM + Numerical Methods for Options

**Status**: Accepted

**Context**: Option pricing requires both analytical and numerical methods.
European options have closed-form solutions; American options and path-
dependent payoffs require numerical approximation.

**Decision**: Implement three complementary pricing methods:
1. Black-Scholes-Merton (closed form) — European options only
2. CRR Binomial Tree — American options, early exercise
3. Monte Carlo simulation — path-dependent payoffs (Asian, barrier)

**Rationale**:
- BSM provides an instantaneous, exact benchmark for European options —
  critical for validating the numerical methods.
- The binomial tree handles American exercise (which MC cannot) and converges
  to BSM European prices as N increases — providing cross-validation.
- MC handles arbitrary path-dependent payoffs (which the tree cannot) and
  provides a standard error for the estimate.

**Trade-offs**:
- Three implementations = three times the maintenance and testing surface.
- The binomial tree has O(N²) time complexity; MC has O(N_steps × N_paths).
  Choose N=100 (tree) or N_paths=10,000 (MC) for interactive use.

**Consequences**:
- The `__init__.py` exports all three pricers under the same namespace.
- All pricers share the same parameter conventions (annualised, continuous).

---

### ADR-002: GARCH(1,1) Instead of Higher-Order GARCH

**Status**: Accepted

**Context**: Volatility modelling must capture clustering, mean reversion,
and fat tails. The GARCH family offers (p,q) configurations of increasing
complexity.

**Decision**: GARCH(1,1) — one lag of squared error, one lag of variance.

**Rationale**:
- GARCH(1,1) is the "workhorse" model for financial volatility
  (Bollerslev 1986). It captures the three key stylised facts of asset
  returns with only three parameters.
- Higher-order models (GARCH(2,2), EGARCH, GJR-GARCH) add marginal
  forecast improvement at the cost of overfitting risk and parameter
  non-identifiability (Hull 2022, Ch. 24).
- The three parameters (omega, alpha, beta) are directly interpretable:
  alpha = news sensitivity, beta = persistence, omega = long-run level.

**Trade-offs**:
- GARCH(1,1) cannot capture leverage effects (asymmetric vol response to
  positive vs negative returns). EGARCH or GJR-GARCH would be needed.
- Normal error distribution understates tail risk. A Student-t or GED
  distribution would be more realistic.
- These extensions are deferred to a future enhancement (Phase 2c?).

**Consequences**:
- The GARCH class returns persistence (alpha+beta) and half-life diagnostics.
- Maximum likelihood estimation uses gradient descent. For production use
  with thousands of time series, replace with scipy.optimize.

---

### ADR-003: Pure-Python Matrix Operations (No NumPy Dependency)

**Status**: Accepted

**Context**: The portfolio optimisation module requires matrix inversion,
multiplication, and vector operations. NumPy provides these natively.

**Decision**: Implement all matrix operations in pure Python.

**Rationale**:
- The portfolio module operates on small matrices (N ≤ 50 assets typically,
  often 2-5). NumPy's vectorisation overhead dwarfs the actual computation.
- Avoiding NumPy means the portfolio module has zero external dependencies —
  it can be imported and tested in any Python environment.
- Gauss-Jordan elimination for up to 50×50 matrices completes in <1ms in
  pure Python.

**Trade-offs**:
- For N > 100, NumPy's BLAS-optimised operations would be 10-100× faster.
- The `_invert_gauss_jordan` method is O(N³) — acceptable for N ≤ 50, but
  not scalable to large universes.

**Consequences**:
- The `MeanVarianceOptimiser`, `RiskParity`, and `BlackLitterman` classes are
  self-contained with no NumPy or SciPy dependency.
- If future work requires N > 100 (e.g., FTSE 100 or S&P 500 optimisation),
  add a NumPy backend.
- The `_invert` method is defined once in `mean_variance.py` and reused
  (via copy-paste) in `black_litterman.py` — kept separate to avoid
  cross-module coupling of portfolio sub-modules.

---

### ADR-004: Closed-Form Min-Variance and Max-Sharpe Without QP Solver

**Status**: Accepted

**Context**: Mean-variance optimisation typically requires a quadratic
programming (QP) solver for constrained problems.

**Decision**: Use closed-form solutions for the unconstrained case.

**Rationale**:
- The global minimum variance portfolio has a closed form:
  w = Sigma⁻¹·1 / (1ᵀ·Sigma⁻¹·1)
- The maximum Sharpe (tangency) portfolio also has a closed form:
  w = Sigma⁻¹·(mu - r_f·1) / (1ᵀ·Sigma⁻¹·(mu - r_f·1))
- These solutions avoid a QP solver dependency and are trivial to implement.
- For long-only or box constraints, the closed form is a good starting point
  that can be adjusted.

**Trade-offs**:
- No upper/lower bound constraints (e.g., no short selling).
- The efficient frontier with a target return constraint requires solving a
  (N+2)×(N+2) system, which we implement analytically.

**Consequences**:
- The `MeanVarianceOptimiser` class returns unconstrained weights by default.
- Users can apply their own constraints (e.g., clip negatives to 0 and
  renormalise) — documented in the class docstring.

---

## Implementation Guide

### Option Pricing (pricers/)

**BlackScholesPricer** — Call `compute("call")` or `compute("put")` to get a
`BSMResult` with premium and all five Greeks. Validated against Hull (2022)
reference values.

**BinomialTreePricer** — Call `compute("call", "american")` for American
options. Use N=500 for ~0.02 convergence to BSM. The `early_exercise` field
in the result indicates whether early exercise was optimal.

**MonteCarloPricer** — Pass a custom payoff function for path-dependent
options. Use `standard_payoff()`, `asian_payoff()`, or `barrier_payoff()`
factory methods for common cases. The `standard_error` field provides a 95%
confidence interval.

### Volatility (volatility/)

**GARCH** — Fit to daily returns via `GARCH().fit(returns)`. Check
`result.convergence` — if False, the gradient descent may need more
iterations or a different learning rate. Forecast with `.forecast(steps=10)`.

### Portfolio (portfolio/)

**MeanVarianceOptimiser** — Use `minimum_variance(cov, tickers)` or
`max_sharpe(cov, returns, tickers)`. The `efficient_frontier()` method
traces the curve for a target return.

**RiskParity** — Call `optimise(cov, tickers)`. The
`risk_contributions` dict shows each asset's percentage of total risk.

**BlackLitterman** — Call `blend(cov, mkt_weights, tickers, P=P, Q=Q)`
with a pick matrix P and view vector Q. Without views, returns the CAPM
equilibrium.

---

## References

- Black, F., & Scholes, M. (1973). The Pricing of Options and Corporate
  Liabilities. *Journal of Political Economy*, 81(3).
- Black, F., & Litterman, R. (1992). Global Portfolio Optimization.
  *Financial Analysts Journal*, 48(5).
- Bollerslev, T. (1986). Generalized Autoregressive Conditional
  Heteroskedasticity. *Journal of Econometrics*, 31(3).
- Boyle, P. (1977). Options: A Monte Carlo Approach. *Journal of Financial
  Economics*, 4(3).
- Cox, J., Ross, S., & Rubinstein, M. (1979). Option Pricing: A Simplified
  Approach. *Journal of Financial Economics*, 7(3).
- Hull, J. (2022). *Options, Futures, and Other Derivatives* (11th ed.),
  Chapters 13, 21, 22, 24.
- Maillard, S., Roncalli, T., & Teiletche, J. (2010). The Properties of
  Equally Weighted Risk Contribution Portfolios. *Journal of Portfolio
  Management*, 36(4).
- Markowitz, H. (1952). Portfolio Selection. *The Journal of Finance*, 7(1).
- Beckman, Q. (2025). From Research to Real Trading: The RKDG Framework.
  *QuantlabsNet Blog*.
