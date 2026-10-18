# quant-methods — Quantitative Methods & Pricing Models

## Objective
Implement classical quant finance models and Quant Beckman's statistical
validation framework: Black-Scholes, binomial trees, Monte Carlo, GARCH,
portfolio optimisation, plus falsification via PSR/DSR/CPCV/WFA.

This spans **Phase 2a (Classical Models)** and **Phase 4 (Validation
Framework)** in the STRATEGY.md roadmap.

## Projects

### Phase 2a — Classical Models (Completed)
1. **Black-Scholes Pricer** — European options + Greeks (delta, gamma, vega,
   theta, rho). Validated against Hull (2022).
2. **Binomial Tree** — CRR lattice for American/European options. Converges
   to BSM within 0.02 (N=500 steps).
3. **Monte Carlo Engine** — GBM with standard, Asian, and barrier payoffs.
4. **GARCH Volatility** — GARCH(1,1) via gradient descent MLE with
   mean-reversion diagnostics.
5. **Portfolio Optimisation** — Mean-variance, risk parity, Black-Litterman.

### Phase 4 — Validation Framework (Completed)
6. **Backtester** — Vectorised strategy evaluation (Sharpe, Sortino, Calmar,
   max DD, win rate, profit factor) with benchmark alpha/beta/IR.
7. **Monkey Test** — Quant Beckman's permutation falsification: shuffle
   returns N times, compare actual Sharpe to null distribution, get p-value.
8. **Probabilistic Sharpe Ratio** — Bailey & López de Prado (2012). PSR(SR*)
   = probability that true Sharpe exceeds a threshold, corrected for
   skewness and kurtosis.
9. **Deflated Sharpe Ratio** — Bailey & López de Prado (2014). Corrects PSR
   for multiple testing using expected max Sharpe under N trials.
10. **Block Bootstrap** — Circular block bootstrap (Politis & Romano) for
    confidence intervals on any metric, preserving serial dependence.
11. **Combinatorial Purged CV** — López de Prado (2018). C(N,K) train/test
    splits with embargo purging, OOS Sharpe distribution.
12. **Walk-Forward Analysis** — Anchored/rolling re-optimisation windows
    with OOS concatenation and robustness diagnostics.

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
| validation/backtester.py | `Backtester` | Vectorised strategy performance metrics |
| validation/monkey_test.py | `MonkeyTest` | Permutation falsification test |
| validation/psr.py | `ProbabilisticSharpeRatio` | PSR with skew/kurtosis correction |
| validation/dsr.py | `DeflatedSharpeRatio` | DSR with multiple-testing correction |
| validation/bootstrap.py | `BlockBootstrap` | Circular block bootstrap CI |
| validation/cpcv.py | `CombinatorialPurgedCV` | Purged cross-validation |
| validation/wfa.py | `WalkForwardAnalysis` | Anchored/rolling walk-forward |

## Status
- [x] Phase 2a — Black-Scholes, binomial tree, Monte Carlo, GARCH, portfolio (unit checks)
- [x] Phase 4 — Backtester, Monkey Test, PSR, DSR, Bootstrap, CPCV, WFA (unit checks)
- [x] **Unit checks are listed here; no current pass result is established by this phase note**

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

---

### ADR-005: PSR/DSR over Simple Sharpe Ratio for Strategy Selection

**Status**: Accepted

**Context**: A strategy's Sharpe ratio is a point estimate with no measure
of uncertainty. Strategies with high Sharpe from short histories are likely
overfitted. We need a probabilistic framework for comparing strategies.

**Decision**: Use PSR (Probabilistic Sharpe Ratio) for single-strategy
evaluation and DSR (Deflated Sharpe Ratio) when comparing across multiple
strategies.

**Rationale**:
- PSR accounts for skewness and kurtosis in returns — a strategy with
  positive skew (occasional big wins) is more likely to be real than one
  with negative skew (occasional big losses), even at the same Sharpe.
- DSR corrects for the multiple-testing bias: if you try 100 strategies,
  the best one will have inflated Sharpe by chance. The expected max Sharpe
  under the null is estimated and used as the deflated threshold.
- The standard Sharpe ratio ignores these effects, leading to overly
  optimistic strategy selection.

**Consequences**:
- PSR >= 0.95 → strategy's true Sharpe likely exceeds threshold.
- DSR >= 0.95 → strategy survives multiple-testing correction.
- Both are implemented in the validation framework and integrated with
  the Monkey Test for a comprehensive falsification pipeline.

---

### ADR-006: Permutation Test (Monkey Test) over Parametric Null

**Status**: Accepted

**Context**: Testing whether a strategy's performance is statistically
significant requires a null distribution. Options: parametric (assume
normal returns) or non-parametric (permutation test).

**Decision**: Non-parametric permutation test that shuffles actual returns.

**Rationale**:
- Financial returns are non-normal (fat tails, skewness, autocorrelation).
  A parametric null (e.g., t-test) will overstate significance.
- Shuffling preserves the empirical distribution of returns — magnitudes
  are unchanged, only the sequence is randomised.
- This directly answers the question: "What fraction of random timing
  decisions would produce this result?"

**Consequences**:
- p-value = (count_null >= actual + 1) / (N + 1). p < 0.05 is significant.
- Custom statistic functions are supported (default is Sharpe ratio).
- Deterministic with random_seed for reproducibility.

---

### ADR-007: Circular Block Bootstrap over IID Bootstrap

**Status**: Accepted

**Context**: Confidence intervals for performance metrics require
resampling that preserves the dependence structure of time series data.

**Decision**: Circular block bootstrap (Politis & Romano, 1992) with
optimal block size L = ceil(T^(1/3)).

**Rationale**:
- IID bootstrap destroys autocorrelation and volatility clustering,
  leading to artificially narrow confidence intervals.
- Circular block bootstrap wraps the series into a circle, handling
  end-of-series boundary effects naturally.
- The optimal block size L ~ T^(1/3) minimises mean squared error
  of the variance estimate under weak dependence conditions.
- Block size defaults to this value but is user-configurable.

**Consequences**:
- Returns are resampled in blocks, preserving short-term dependence.
- The result includes both percentile-based CI and bootstrap standard
  error.
- A dedicated `sharpe_ratio()` method wraps the general
  `confidence_interval()` for convenience.

---

### ADR-008: Combinatorial Purged CV over Standard Time-Series Split

**Status**: Accepted

**Context**: Walk-forward and standard time-series splits evaluate a
strategy on a single path of train/test assignments. Each observation
appears in the test set only once, leading to high variance.

**Decision**: Combinatorial Purged Cross-Validation (CPCV) with embargo
purging and sub-sampling of C(N, K) combinations.

**Rationale**:
- CPCV puts every observation in a test set multiple times (across
  different combinations), reducing the variance of OOS performance.
- The embargo period prevents forward-looking bias by purging training
  data too close to the test boundary.
- Sub-sampling C(N, K) combinations (default max 100) keeps computation
  tractable while maintaining the diversity benefit.

**Consequences**:
- OOS Sharpe distribution (mean, std, min, max) across all combinations.
- Significant if mean - 2 * std > 0 (i.e., 95% of OOS Sharpes are positive).
- The WFA (Walk-Forward Analysis) module offers complementary anchored
  and rolling modes for production-like re-optimisation simulation.

---

## References

- Bailey, D. & López de Prado, M. (2012). The Sharpe Ratio Efficient
  Frontier. *Journal of Risk*, 15(2).
- Bailey, D. & López de Prado, M. (2014). The Deflated Sharpe Ratio:
  Correcting for Multiple Testing, Selection Bias, and Non-Normality.
  *Journal of Portfolio Management*, 40(3).
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
- López de Prado, M. (2018). *Advances in Financial Machine Learning*.
  Wiley. Chapters 12-14.
- Maillard, S., Roncalli, T., & Teiletche, J. (2010). The Properties of
  Equally Weighted Risk Contribution Portfolios. *Journal of Portfolio
  Management*, 36(4).
- Markowitz, H. (1952). Portfolio Selection. *The Journal of Finance*, 7(1).
- Politis, D. & Romano, J. (1992). A Circular Block-Resampling Procedure
  for Stationary Data. *Exploring the Limits of Bootstrap*.
- Beckman, Q. (2025). From Research to Real Trading: The RKDG Framework.
  *QuantlabsNet Blog*.
