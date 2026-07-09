"""
risk_parity.py — Risk parity portfolio (equal risk contribution).

WHY RISK PARITY:
    Traditional 60/40 portfolios allocate capital, not risk. During a bond
    crash, the 40% bond allocation can dominate portfolio risk. Risk parity
    solves this by ensuring each asset contributes equally to total
    portfolio risk.

THE INTUITION:
    Risk parity weights are NOT equal capital weights. They equalise each
    asset's contribution to portfolio volatility:
        weight_i * (Sigma * w)_i = weight_j * (Sigma * w)_j  for all i, j

    In a two-asset portfolio:
        - Stocks (vol 15%): weight 60% → risk contribution ~95%
        - Bonds (vol 5%):   weight 40% → risk contribution ~5%

    Risk parity would give:
        - Stocks: ~25% weight → 50% risk contribution
        - Bonds:  ~75% weight → 50% risk contribution

    This requires leverage on bonds to achieve the same target volatility
    as 60/40 — hence risk parity portfolios typically use 1.5x-2x leverage
    on the bond component.

WHY BECKMAN LIKES IT:
    Risk parity does not require expected return estimates — it is purely
    a risk-based allocation. This avoids the estimation error that plagues
    mean-variance optimisation (the "Markowitz curse").

MATHEMATICAL FORMULATION:
    Find w that minimises:
        sum_i (RC_i - target_risk)^2

    where:
        RC_i = w_i * (Sigma * w)_i / sqrt(w^T Sigma w)

    Subject to:
        sum_i w_i = 1 (or some target leverage)
        w_i >= 0

    This is a non-convex optimisation because RC_i depends on w in a
    nonlinear way. We use iterative gradient-based optimisation.

REFERENCES:
    Maillard, S., Roncalli, T., & Teiletche, J. (2010). The Properties
        of Equally Weighted Risk Contribution Portfolios.
        Journal of Portfolio Management, 36(4).
"""

from dataclasses import dataclass, field
from math import sqrt


@dataclass
class RiskParityResult:
    """
    Output of the risk parity optimisation.

    Attributes:
        weights: dict mapping asset name -> weight.
        risk_contributions: Each asset's percentage contribution to total
            portfolio risk. In a perfect solution, all values are equal.
        portfolio_vol: Portfolio standard deviation.
        convergence: Whether the optimiser converged.
    """

    weights: dict[str, float]
    risk_contributions: dict[str, float]
    portfolio_vol: float
    convergence: bool


class RiskParity:
    """
    Risk parity portfolio — equal risk contribution across assets.

    USAGE:
        parity = RiskParity()
        result = parity.optimise(cov_matrix, tickers)
        print(result.weights)
        print(result.risk_contributions)  # all ~1/N

    NOTE:
        The optimisation may not converge for large N or pathological
        covariance matrices. Check result.convergence.

    COMPUTATIONAL NOTE:
        This uses gradient descent to minimise the variance of risk
        contributions. A perfect risk parity solution exists only if
        the covariance matrix is positive definite (which it always
        should be for a valid covariance matrix).
    """

    def __init__(
        self,
        learning_rate: float = 0.05,
        max_iter: int = 20000,
        tol: float = 1e-8,
    ) -> None:
        self._lr = learning_rate
        self._max_iter = max_iter
        self._tol = tol

    # ------------------------------------------------------------------
    # Risk contribution computation
    # ------------------------------------------------------------------

    @staticmethod
    def _risk_contributions(
        weights: list[float], cov_matrix: list[list[float]]
    ) -> list[float]:
        """
        Compute each asset's marginal risk contribution.

        RC_i = w_i * (Sigma * w)_i / sqrt(w^T Sigma w)

        WHY marginal risk contribution (MRC):
            The derivative of portfolio volatility with respect to weight_i
            is (Sigma * w)_i / sqrt(w^T Sigma w). Multiplied by w_i, this
            gives the total contribution of asset i to portfolio risk.

        PROPERTIES:
            - sum(RC_i) = portfolio volatility
            - RC_i >= 0 (for long-only portfolios)
        """
        n = len(weights)
        sigma_w = [sum(cov_matrix[i][j] * weights[j] for j in range(n)) for i in range(n)]
        port_var = sum(weights[i] * sigma_w[i] for i in range(n))
        port_vol = sqrt(port_var) if port_var > 0 else 0.0

        if port_vol == 0:
            return [0.0] * n

        return [weights[i] * sigma_w[i] / port_vol for i in range(n)]

    # ------------------------------------------------------------------
    # Optimisation
    # ------------------------------------------------------------------

    def optimise(
        self,
        cov_matrix: list[list[float]],
        tickers: list[str],
    ) -> RiskParityResult:
        """
        Find risk parity weights by minimising variance of risk
        contributions.

        The objective function penalises deviations from equal risk
        contribution:
            f(w) = var(RC_1, RC_2, ..., RC_N)

        WHY variance of RC (not e.g. sum of squared deviations):
            Minimising the variance automatically pushes all RC_i toward
            their mean, which is the equal risk contribution.

        Args:
            cov_matrix: NxN covariance matrix of asset returns.
            tickers: List of N asset names.

        Returns:
            RiskParityResult with weights and risk contributions.
        """
        n = len(tickers)
        if len(cov_matrix) != n:
            raise ValueError("cov_matrix and tickers must have same length")

        # Initialise with equal weights (a common starting point)
        weights = [1.0 / n] * n

        prev_obj = float("inf")

        for iteration in range(self._max_iter):
            rc = self._risk_contributions(weights, cov_matrix)

            # Objective: variance of risk contributions
            mean_rc = sum(rc) / n
            obj = sum((r - mean_rc) ** 2 for r in rc) / n

            if abs(obj - prev_obj) < self._tol:
                break
            prev_obj = obj

            # Numerical gradient (finite difference)
            eps = 1e-6
            grad = [0.0] * n
            for i in range(n):
                w_plus = list(weights)
                w_plus[i] += eps
                rc_plus = self._risk_contributions(w_plus, cov_matrix)
                mean_plus = sum(rc_plus) / n
                obj_plus = sum((r - mean_plus) ** 2 for r in rc_plus) / n

                w_minus = list(weights)
                w_minus[i] -= eps
                rc_minus = self._risk_contributions(w_minus, cov_matrix)
                mean_minus = sum(rc_minus) / n
                obj_minus = sum((r - mean_minus) ** 2 for r in rc_minus) / n

                grad[i] = (obj_plus - obj_minus) / (2 * eps)

            # Gradient descent with projection onto simplex
            for i in range(n):
                weights[i] -= self._lr * grad[i]

            # Project onto probability simplex (w_i >= 0, sum(w) = 1)
            # WHY using the soft projection algorithm:
            #   Simple clipping and renormalisation may not converge.
            #   Soft projection (by subtracting the mean of positives) is
            #   the correct Euclidean projection onto the simplex.
            weights = self._project_simplex(weights)

        # Final risk contributions
        final_rc = self._risk_contributions(weights, cov_matrix)
        port_var = sum(weights[i] * sum(cov_matrix[i][j] * weights[j] for j in range(n)) for i in range(n))
        port_vol = sqrt(port_var)

        return RiskParityResult(
            weights={t: w for t, w in zip(tickers, weights)},
            risk_contributions={t: rc / sum(final_rc) for t, rc in zip(tickers, final_rc)},
            portfolio_vol=round(port_vol, 6),
            convergence=iteration < self._max_iter - 1,
        )

    # ------------------------------------------------------------------
    # Simplex projection
    # ------------------------------------------------------------------

    @staticmethod
    def _project_simplex(v: list[float]) -> list[float]:
        """
        Project a vector onto the probability simplex.

        The simplex is defined as:
            { w | w_i >= 0, sum_i w_i = 1 }

        This implements the algorithm from:
            Duchi et al. (2008). Efficient Projections onto the L1-Ball.

        The idea:
            1. Sort descending.
            2. Find the threshold rho such that sum of (v_i - tau) clipped
               to >= 0 equals 1.
            3. w_i = max(v_i - tau, 0).

        WHY this specific projection:
            Simple clipping (max(v_i, 0)) and renormalisation does not
            satisfy sum-to-1. This algorithm is the exact Euclidean
            projection and converges in O(N log N).
        """
        n = len(v)
        # Sort descending
        sorted_v = sorted(v, reverse=True)

        # Find the threshold
        cumulative = 0.0
        tau = 0.0
        for i in range(n):
            cumulative += sorted_v[i]
            tau = (cumulative - 1.0) / (i + 1)
            if i == n - 1 or sorted_v[i + 1] <= tau:
                break

        # Project: w_i = max(v_i - tau, 0)
        w = [max(x - tau, 0.0) for x in v]

        # Numerical safeguard: renormalise to sum to exactly 1
        total = sum(w)
        if total > 0:
            w = [x / total for x in w]

        return w
