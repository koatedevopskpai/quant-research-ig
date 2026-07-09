"""
garch.py — GARCH(1,1) volatility model.

WHY THIS EXISTS:
    The Black-Scholes model assumes constant volatility, but real financial
    markets exhibit:
    1. Volatility clustering — periods of high volatility tend to follow
       high volatility, and low follows low.
    2. Mean reversion — volatility tends to revert to a long-term average.
    3. Fat tails — return distributions have heavier tails than the normal
       distribution.

    GARCH (Generalized AutoRegressive Conditional Heteroskedasticity) models
    capture these features by modelling volatility as a time-varying process
    that depends on past returns and past volatility.

THE GARCH(1,1) MODEL (Bollerslev 1986):
    The conditional variance at time t is:
        sigma_t^2 = omega + alpha * epsilon_{t-1}^2 + beta * sigma_{t-1}^2

    where:
        omega: Long-run variance (constant term).
        alpha: Sensitivity to recent squared returns (news coefficient).
        beta: Sensitivity to past variance (persistence coefficient).
        epsilon_{t-1}: The return residual at time t-1.

    KEY PROPERTIES:
        - Persistence: alpha + beta measures the persistence of volatility
          shocks. Values close to 1 mean shocks decay slowly.
        - Unconditional variance: omega / (1 - alpha - beta).
        - Stationarity: alpha + beta < 1 ensures mean-reverting volatility.

WHY GARCH(1,1) specifically (vs. GARCH(p,q) with higher orders):
    The (1,1) specification — one lag of squared error and one lag of
    variance — is sufficient for most financial time series (Hull 2022,
    Chapter 24). Higher-order models (e.g., GARCH(2,2)) rarely improve
    forecast accuracy enough to justify the additional parameters and
    overfitting risk.

USAGE:
    from quant_methods.volatility.garch import GARCH

    model = GARCH()
    model.fit(returns)  # returns: list or numpy array of daily returns
    forecast = model.forecast(steps=10)
    print(f"Predicted vol next 10 days: {forecast}")

METHODOLOGY NOTE (Quant Beckman):
    GARCH is central to Beckman's "hypothesis testing" framework. Before
    building an options strategy, a researcher must understand the current
    volatility regime. GARCH provides:
    1. A statistically rigorous estimate of current volatility.
    2. A forecast of future volatility (for option pricing).
    3. Diagnostic statistics (persistence, long-run vol) that inform model
       selection for downstream strategies.

REFERENCES:
    Bollerslev, T. (1986). Generalized Autoregressive Conditional
        Heteroskedasticity. Journal of Econometrics, 31(3).
    Hull, J. (2022). Options, Futures, and Other Derivatives (11th ed.),
        Chapter 24.
"""

from dataclasses import dataclass, field
from math import sqrt


@dataclass
class GARCHResult:
    """
    Fitted GARCH(1,1) model parameters and diagnostics.

    WHY these specific diagnostics are included:
        - persistence (alpha + beta): How long a vol shock lasts.
          > 0.99: Very persistent (typical for equities).
        - half_life: How many periods for a vol shock to decay by 50%.
          = ln(0.5) / ln(persistence).
        - unconditional_vol: The long-run average volatility the model
          reverts to. This is the "baseline" vol when no recent shocks
          have occurred.
        - convergence: Whether the maximum likelihood estimation converged.
          If False, the parameter estimates may be unreliable.

    INTERPRETATION (for portfolio reviewers):
        A GARCH(1,1) fitted to S&P 500 daily returns typically yields:
            omega:  ~0.000001 (very small — daily variance is tiny)
            alpha:  ~0.08     (8% weight on recent squared return)
            beta:   ~0.91     (91% weight on past variance)
            persistence: ~0.99 (shocks decay very slowly)
            half-life: ~70 days
            unconditional vol: ~15-20% annualised
    """

    omega: float
    alpha: float
    beta: float
    persistence: float
    half_life: float
    unconditional_vol: float
    fitted_volatilities: list[float] = field(repr=False)
    convergence: bool = True


class GARCH:
    """
    GARCH(1,1) volatility model with maximum likelihood estimation.

    Parameters:
        omega_init, alpha_init, beta_init: Starting values for MLE.
            Defaults are typical for financial time series.

    USAGE:
        # Daily returns as a list or array
        returns = [0.01, -0.005, 0.02, ...]

        model = GARCH()
        result = model.fit(returns)

        # Access diagnostics
        print(f"Persistence: {result.persistence:.4f}")
        print(f"Unconditional vol: {result.unconditional_vol:.4%}")

        # Forecast future volatility
        forecast = model.forecast(steps=20)

    WHY NOT USING a dedicated optimisation library like scipy.optimize:
        GARCH likelihood optimisation is simple enough for manual gradient
        descent with reasonable initial guesses. scipy.optimize would add
        a dependency for a function we can implement in 20 lines. However,
        for production use with hundreds of time series, scipy's optimisers
        are more robust (faster convergence, better handling of boundary
        conditions).

    LIMITATION:
        This implementation uses a simple fixed-step gradient descent.
        For pathological data (e.g., all-zero returns), the optimisation
        may not converge. Always check result.convergence.
    """

    def __init__(
        self,
        omega_init: float = 0.0001,
        alpha_init: float = 0.1,
        beta_init: float = 0.8,
    ) -> None:
        # WHY default alpha + beta = 0.9:
        #   Typical financial data has persistence between 0.9 and 0.99.
        #   Starting with alpha=0.1, beta=0.8 gives persistence=0.9 — a
        #   conservative starting point that usually converges.
        self._params = [omega_init, alpha_init, beta_init]
        self._result: GARCHResult | None = None

    # ------------------------------------------------------------------
    # Conditional variance computation
    # ------------------------------------------------------------------

    def _compute_variance_series(
        self, returns: list[float], params: list[float]
    ) -> list[float]:
        """
        Compute the conditional variance series given GARCH parameters.

        sigma_t^2 = omega + alpha * epsilon_{t-1}^2 + beta * sigma_{t-1}^2

        WHY using the sample variance as the initial sigma_0^2:
            The first variance estimate needs a starting value. Using the
            sample variance of the entire series is standard practice and
            ensures the starting point is reasonable. The influence of this
            initial value decays exponentially — after ~50 observations,
            its effect is negligible.
        """
        omega, alpha, beta = params
        n = len(returns)

        # Initial variance: sample variance of the returns series.
        mean_r = sum(returns) / n
        sample_var = sum((r - mean_r) ** 2 for r in returns) / (n - 1)

        variances = [sample_var]
        for i in range(1, n):
            new_var = omega + alpha * (returns[i - 1] ** 2) + beta * variances[-1]
            variances.append(new_var)

        return variances

    # ------------------------------------------------------------------
    # Negative log-likelihood
    # ------------------------------------------------------------------

    def _neg_log_likelihood(
        self, returns: list[float], params: list[float]
    ) -> float:
        """
        Compute the negative log-likelihood of the GARCH(1,1) model.

        The likelihood assumes normally distributed errors:
            epsilon_t | F_{t-1} ~ N(0, sigma_t^2)

        The log-likelihood for observation t is:
            l_t = -0.5 * log(2*pi) - 0.5 * log(sigma_t^2) - 0.5 * epsilon_t^2 / sigma_t^2

        WHY NLL (minimisation) instead of LL (maximisation):
            Numerical optimisers are typically written for minimisation.
            Minimising NLL is equivalent to maximising LL.

        WHY omitting the -0.5*log(2*pi) constant:
            It is constant across all parameter values and does not affect
            the optimal parameter choice. Omitting it saves computation.

        WHY dividing by the number of observations (AVERAGE NLL instead of
        total NLL):
            The total log-likelihood scales with N (number of observations).
            Using average NLL makes the gradient scale independent of N,
            allowing a single default learning rate to work for datasets of
            any size.
        """
        variances = self._compute_variance_series(returns, params)
        n = len(returns)
        # WHY starting from i=1:
        #   The first variance estimate uses the sample variance (not
        #   model-generated), so we exclude it from likelihood computation
        #   to avoid bias.
        ll = 0.0
        for i in range(1, n):
            ll += -0.5 * (  # -0.5*log(2*pi) omitted
                variances[i]
                + (returns[i] ** 2) / variances[i]
            )
        return -ll / (n - 1)  # AVERAGE negative log-likelihood

    # ------------------------------------------------------------------
    # Parameter estimation via gradient descent
    # ------------------------------------------------------------------

    def fit(
        self,
        returns: list[float],
        learning_rate: float = 1e-4,
        max_iter: int = 5000,
        tol: float = 1e-7,
    ) -> GARCHResult:
        """
        Estimate GARCH(1,1) parameters by maximum likelihood.

        Args:
            returns: List of asset returns (e.g., daily log returns).
            learning_rate: Step size for gradient descent. The NLL is
                averaged (not summed), so a single default works across
                dataset sizes. Increase if convergence is too slow,
                decrease if it oscillates.
            max_iter: Maximum optimisation iterations.
            tol: Convergence tolerance (stop when NLL change < tol).

        Returns:
            GARCHResult with fitted parameters and diagnostics.

        WHY gradient descent vs. closed-form estimation:
            Unlike linear regression, GARCH has no closed-form solution.
            Parameters must be estimated numerically. Gradient descent is
            simple and transparent — each step is interpretable.

        WHY constraints are NOT enforced here:
            alpha >= 0, beta >= 0, omega > 0, alpha + beta < 1 are required
            for a valid GARCH model. If the optimisation produces invalid
            parameters, the caller should check the result and potentially
            re-run with different initial values. Enforcing constraints
            would complicate the gradient descent code and may hide
            data issues (e.g., non-stationary returns).
        """
        n = len(returns)
        if n < 10:
            raise ValueError(
                f"Need at least 10 observations, got {n}"
            )

        params = list(self._params)
        prev_nll = float("inf")

        for iteration in range(max_iter):
            variances = self._compute_variance_series(returns, params)

            # ------------------------------------------------------------------
            # Numerical gradient computation
            # ------------------------------------------------------------------
            # WHY numerical (finite-difference) gradients vs. analytical:
            #   Analytical gradients for GARCH likelihood are complex and
            #   error-prone. Numerical gradients with a small epsilon are
            #   simpler and accurate enough for parameter estimation.
            eps = 1e-6
            grad = [0.0, 0.0, 0.0]

            for j in range(3):
                params_plus = list(params)
                params_plus[j] += eps
                nll_plus = self._neg_log_likelihood(returns, params_plus)

                params_minus = list(params)
                params_minus[j] -= eps
                nll_minus = self._neg_log_likelihood(returns, params_minus)

                grad[j] = (nll_plus - nll_minus) / (2 * eps)

            # Gradient descent step
            for j in range(3):
                params[j] -= learning_rate * grad[j]

            # Enforce non-negativity constraints
            # WHY clamping parameters:
            #   Valid GARCH require: omega > 0, alpha >= 0, beta >= 0.
            #   Negative values are financially meaningless (negative
            #   variance?). We clamp to a small positive number.
            params[0] = max(params[0], 1e-8)  # omega > 0
            params[1] = max(params[1], 1e-8)  # alpha >= 0
            params[2] = max(params[2], 1e-8)  # beta >= 0

            # Check convergence
            current_nll = self._neg_log_likelihood(returns, params)
            if abs(current_nll - prev_nll) < tol:
                break
            prev_nll = current_nll

        # ------------------------------------------------------------------
        # Compute diagnostics
        # ------------------------------------------------------------------
        omega, alpha, beta = params
        persistence = alpha + beta

        # Half-life: how many periods for a vol shock to decay by 50%.
        half_life = 0.0
        if persistence > 0 and persistence < 1:
            half_life = abs(__import__("math").log(0.5) / __import__("math").log(persistence))

        # Unconditional (long-run) annualised volatility.
        # WHY annualising:
        #   If returns are daily, the unconditional variance is per day.
        #   Multiply by 252 (trading days) and take sqrt to get annual vol.
        if persistence >= 1:
            # Non-stationary — unconditional vol is infinite
            unconditional_vol = float("inf")
        else:
            unconditional_var = omega / (1 - persistence)
            unconditional_vol = sqrt(unconditional_var * 252)

        fitted_variances = self._compute_variance_series(returns, params)

        self._result = GARCHResult(
            omega=round(omega, 8),
            alpha=round(alpha, 6),
            beta=round(beta, 6),
            persistence=round(persistence, 6),
            half_life=round(half_life, 2),
            unconditional_vol=round(unconditional_vol, 6),
            fitted_volatilities=[
                sqrt(v * 252) for v in fitted_variances
            ],  # annualised
            convergence=iteration < max_iter - 1,
        )
        return self._result

    # ------------------------------------------------------------------
    # Forecasting
    # ------------------------------------------------------------------

    def forecast(self, steps: int = 10) -> list[float]:
        """
        Forecast future conditional volatility.

        The GARCH(1,1) forecast for h steps ahead converges to the
        unconditional variance:
            sigma_{t+h}^2 = unconditional_var + (persistence)^h * (sigma_t^2 - unconditional_var)

        This means:
            - Short-term forecasts are closer to the last observed vol.
            - Long-term forecasts converge to the long-run average vol.
            - Higher persistence = slower convergence.

        Args:
            steps: Number of steps ahead to forecast.

        Returns:
            List of annualised volatility forecasts (one per step).

        Raises:
            RuntimeError: If the model has not been fitted yet.
        """
        if self._result is None:
            raise RuntimeError("Model must be fitted before forecasting")

        omega = self._result.omega
        alpha = self._result.alpha
        beta = self._result.beta
        # GARCH persistence = alpha + beta, NOT omega + alpha + beta.
        # omega is the constant term, not part of the autoregressive
        # structure, so it does not affect the decay rate of shocks.
        persistence = alpha + beta

        # Last fitted variance (daily, not annualised)
        last_var = self._result.fitted_volatilities[-1]
        last_var_daily = (last_var / sqrt(252)) ** 2

        # Unconditional daily variance
        if persistence >= 1:
            unc_var_daily = last_var_daily  # no mean reversion
        else:
            unc_var_daily = omega / (1 - persistence)

        forecast_daily_vars = []
        for h in range(1, steps + 1):
            fcast_var = unc_var_daily + (persistence ** h) * (
                last_var_daily - unc_var_daily
            )
            forecast_daily_vars.append(fcast_var)

        # Convert daily variance to annualised vol for readability
        return [sqrt(v * 252) for v in forecast_daily_vars]
