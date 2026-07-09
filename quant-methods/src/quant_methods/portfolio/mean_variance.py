"""
mean_variance.py — Mean-Variance portfolio optimisation (Markowitz 1952).

WHY MARKOWITZ MATTERS:
    Harry Markowitz's 1952 paper "Portfolio Selection" formalised the
    trade-off between risk and return. The key insight: diversification
    reduces portfolio risk below the weighted average of individual asset
    risks because asset returns are not perfectly correlated.

THE MATHEMATICS:
    Minimise:      w^T Sigma w
    Subject to:    w^T mu = target_return
                   sum(w) = 1
                   w_i >= 0 (long only, optional)

    Where:
        w:      Vector of portfolio weights
        Sigma:  Covariance matrix of asset returns
        mu:     Vector of expected returns

THE EFFICIENT FRONTIER:
    The set of portfolios that offer the highest expected return for each
    level of risk. Portfolios on the frontier are "efficient" — no other
    portfolio offers higher return for the same risk.

QUANT BECKMAN NOTE:
    Beckman emphasises that raw Markowitz is fragile — small changes in
    input estimates (mu, Sigma) produce drastically different weights.
    This is why he advocates for:
    1. Shrinkage estimators for the covariance matrix (Ledoit-Wolf).
    2. Black-Litterman to stabilise expected return estimates.
    3. Walk-forward analysis to validate out-of-sample performance.

LIMITATIONS:
    - Max return portfolios concentrate in the single highest-return asset.
    - Min variance portfolios ignore return entirely.
    - Solutions are extremely sensitive to input estimates.
    - Estimation error dominates for large universes (>100 assets).
"""

from dataclasses import dataclass, field
from math import sqrt


@dataclass
class PortfolioResult:
    """
    Output of a portfolio optimisation.

    Attributes:
        weights: dict mapping asset name -> weight (sums to 1.0).
        expected_return: Portfolio expected return (same units as inputs).
        volatility: Portfolio standard deviation (same units as inputs).
        sharpe_ratio: (expected_return - risk_free_rate) / volatility.
    """

    weights: dict[str, float]
    expected_return: float
    volatility: float
    sharpe_ratio: float


class MeanVarianceOptimiser:
    """
    Mean-Variance portfolio optimiser.

    Supports:
        - Minimum variance portfolio.
        - Maximum Sharpe ratio (tangency) portfolio.
        - Target return efficient frontier portfolio.

    USAGE:
        optimizer = MeanVarianceOptimiser()
        result = optimizer.minimum_variance(cov_matrix, tickers)

        # Or find the portfolio with maximum Sharpe ratio
        result = optimizer.max_sharpe(cov_matrix, expected_returns, tickers)

    WHY NOT using cvxopt or scipy.optimise:
        For a portfolio of N assets, the unconstrained minimum variance
        solution has a closed form:
            w = Sigma^{-1} * 1 / (1^T * Sigma^{-1} * 1)

        The maximum Sharpe ratio also has a closed form with no constraints:
            w = (Sigma^{-1} * mu) / (1^T * Sigma^{-1} * mu)

        By implementing these closed forms, we avoid dependency on convex
        optimisation libraries and make the code self-contained. However,
        for long-only or box constraints, a proper QP solver is required.

    PROPAGATION OF ERROR (Beckman warning):
        The optimiser is only as good as its inputs. Use shrunk covariance
        estimates and Black-Litterman expected returns for stable results.
    """

    def __init__(self, risk_free_rate: float = 0.0) -> None:
        """
        Args:
            risk_free_rate: Annual risk-free rate (e.g., 0.05 for 5%).
                Used in Sharpe ratio calculation. Default 0.0 for
                simplicity, but in practice use the risk-free rate
                corresponding to the investment horizon.
        """
        self._risk_free_rate = risk_free_rate

    # ------------------------------------------------------------------
    # Matrix helpers (pure Python without numpy)
    # ------------------------------------------------------------------

    @staticmethod
    def _mat_mul(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
        """Matrix multiplication A @ B."""
        n, m, p = len(A), len(A[0]), len(B[0])
        return [
            [sum(A[i][k] * B[k][j] for k in range(m)) for j in range(p)]
            for i in range(n)
        ]

    @staticmethod
    def _mat_vec_mul(A: list[list[float]], v: list[float]) -> list[float]:
        """Matrix-vector multiplication A @ v."""
        return [sum(row[i] * v[i] for i in range(len(v))) for row in A]

    @staticmethod
    def _vec_dot(a: list[float], b: list[float]) -> float:
        """Dot product of two vectors."""
        return sum(x * y for x, y in zip(a, b))

    @staticmethod
    def _scalar_mul_mat(c: float, A: list[list[float]]) -> list[list[float]]:
        """Scalar * matrix."""
        return [[c * x for x in row] for row in A]

    @staticmethod
    def _transpose(A: list[list[float]]) -> list[list[float]]:
        """Matrix transpose."""
        return [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]

    # ------------------------------------------------------------------
    # Matrix inversion (Gauss-Jordan elimination)
    # ------------------------------------------------------------------

    def _invert_3x3(self, A: list[list[float]]) -> list[list[float]]:
        """
        Invert a 3x3 matrix using Gauss-Jordan elimination.

        WHY a 3x3-specific implementation:
            The two-fund separation theorem reduces the portfolio
            optimisation to a 3x3 system involving:
                Sigma (covariance), mu (returns), and 1 (ones vector).
            For N assets, after forming the Lagrangian, we end up solving
            a (N+2)x(N+2) system. But for the efficient frontier with no
            constraints, we solve a 2x2 system involving:
                a = mu^T Sigma^{-1} mu
                b = mu^T Sigma^{-1} 1
                c = 1^T Sigma^{-1} 1

            This method is kept for the constraint-free case where we only
            need to invert the 2x2 system [a b; b c].

        NOTE: This is not used for the full covariance matrix inversion.
        Instead, we require the user to pass a pre-computed inverse or
        use the closed-form solution directly.

        Raises:
            ValueError: If the matrix is singular.
        """
        # Verify dimensions
        if len(A) != 3 or any(len(row) != 3 for row in A):
            raise ValueError("Expected a 3x3 matrix")

        # Augment with identity matrix
        aug = [row[:] + [1.0 if i == j else 0.0 for j in range(3)] for i, row in enumerate(A)]

        # Forward elimination
        for col in range(3):
            # Find pivot
            pivot = aug[col][col]
            if abs(pivot) < 1e-12:
                raise ValueError(
                    "Matrix is singular — assets are perfectly correlated"
                )

            # Scale row
            for j in range(6):
                aug[col][j] /= pivot

            # Eliminate other rows
            for row in range(3):
                if row != col:
                    factor = aug[row][col]
                    for j in range(6):
                        aug[row][j] -= factor * aug[col][j]

        # Extract inverse
        return [[aug[i][j + 3] for j in range(3)] for i in range(3)]

    # ------------------------------------------------------------------
    # Minimum variance portfolio
    # ------------------------------------------------------------------

    def minimum_variance(
        self,
        cov_matrix: list[list[float]],
        tickers: list[str],
    ) -> PortfolioResult:
        """
        Compute the minimum variance portfolio (global minimum risk).

        Closed form (no constraints):
            w_min = (Sigma^{-1} * 1) / (1^T * Sigma^{-1} * 1)

        This portfolio ignores expected returns entirely. It lies at the
        leftmost point of the efficient frontier — the portfolio with the
        lowest possible volatility.

        HOW TO INTERPRET:
            The minimum variance portfolio often has surprising weights
            because it exploits negative correlations to cancel risk.
            Assets with high individual volatility but negative correlation
            to others may receive large weights.

        Args:
            cov_matrix: NxN covariance matrix of asset returns.
            tickers: List of N asset ticker names.

        Returns:
            PortfolioResult with optimal weights and diagnostics.
        """
        n = len(tickers)
        if len(cov_matrix) != n:
            raise ValueError("cov_matrix dimensions must match tickers")

        # Compute Sigma^{-1} * 1
        inv_cov = self._invert_gauss_jordan(cov_matrix)
        ones = [1.0] * n
        w_unnormalised = self._mat_vec_mul(inv_cov, ones)

        # Normalise: w = w_unnormalised / sum(w_unnormalised)
        total = sum(w_unnormalised)
        if abs(total) < 1e-12:
            raise ValueError(
                "Sum of weights is zero — check covariance matrix for "
                "linear dependencies (e.g., one asset is a linear "
                "combination of others)"
            )
        weights = [w / total for w in w_unnormalised]

        # Portfolio return and volatility
        port_ret = 0.0  # no return estimate for min variance
        port_var = self._portfolio_variance(weights, cov_matrix)
        port_vol = sqrt(port_var)

        return PortfolioResult(
            weights={t: w for t, w in zip(tickers, weights)},
            expected_return=port_ret,
            volatility=round(port_vol, 6),
            sharpe_ratio=0.0 if port_vol == 0 else (
                (port_ret - self._risk_free_rate) / port_vol
            ),
        )

    # ------------------------------------------------------------------
    # Maximum Sharpe ratio (tangency) portfolio
    # ------------------------------------------------------------------

    def max_sharpe(
        self,
        cov_matrix: list[list[float]],
        expected_returns: list[float],
        tickers: list[str],
    ) -> PortfolioResult:
        """
        Compute the maximum Sharpe ratio (tangency) portfolio.

        Closed form (long-only):
            w_tan = (Sigma^{-1} * mu) / (1^T * Sigma^{-1} * mu)

        The tangency portfolio maximises the Sharpe ratio:
            Sharpe = (mu_p - r_f) / sigma_p

        This is the portfolio that, when combined with the risk-free asset,
        gives the highest possible Sharpe ratio. The line connecting the
        risk-free rate to the tangency portfolio is the Capital Market
        Line (CML).

        WHY THIS IS THE "BEST" PORTFOLIO:
            According to CAPM, all rational investors should hold a
            combination of the risk-free asset and the tangency portfolio.
            The tangency portfolio is the optimal risky portfolio —
            the one with the highest reward-to-risk ratio.

        Args:
            cov_matrix: NxN covariance matrix of asset returns.
            expected_returns: List of N expected annual returns.
            tickers: List of N asset ticker names.

        Returns:
            PortfolioResult with optimal weights.
        """
        n = len(tickers)
        if len(cov_matrix) != n or len(expected_returns) != n:
            raise ValueError("Dimensions of inputs must match")

        # Excess returns over risk-free rate
        excess = [r - self._risk_free_rate for r in expected_returns]

        # Compute Sigma^{-1} * (mu - r_f)
        inv_cov = self._invert_gauss_jordan(cov_matrix)
        w_unnormalised = self._mat_vec_mul(inv_cov, excess)

        # Normalise
        total = sum(w_unnormalised)
        if abs(total) < 1e-12:
            raise ValueError(
                "Sum of unnormalised weights is zero — check inputs for "
                "zero excess returns or singular covariance"
            )
        weights = [w / total for w in w_unnormalised]

        # Portfolio return, volatility, Sharpe
        port_ret = self._vec_dot(weights, expected_returns)
        port_var = self._portfolio_variance(weights, cov_matrix)
        port_vol = sqrt(port_var)
        sharpe = (port_ret - self._risk_free_rate) / port_vol if port_vol > 0 else 0.0

        return PortfolioResult(
            weights={t: w for t, w in zip(tickers, weights)},
            expected_return=round(port_ret, 6),
            volatility=round(port_vol, 6),
            sharpe_ratio=round(sharpe, 6),
        )

    # ------------------------------------------------------------------
    # Efficient frontier portfolio (target return)
    # ------------------------------------------------------------------

    def efficient_frontier(
        self,
        cov_matrix: list[list[float]],
        expected_returns: list[float],
        tickers: list[str],
        target_return: float,
    ) -> PortfolioResult:
        """
        Find the minimum variance portfolio for a given target return.

        This traces out the efficient frontier — for each target return,
        find the portfolio with minimum variance.

        HOW TO USE:
            Iterate over target returns between min and max asset returns
            to plot the entire frontier:
                min_r = min(expected_returns)
                max_r = max(expected_returns)
                for r in np.linspace(min_r, max_r, 50):
                    pf = optimiser.efficient_frontier(cov, mu, tickers, r)

        Args:
            cov_matrix: NxN covariance matrix.
            expected_returns: N expected returns.
            tickers: N ticker names.
            target_return: Desired portfolio return.

        Returns:
            PortfolioResult with weights minimising variance at target
            return.
        """
        n = len(tickers)
        inv_cov = self._invert_gauss_jordan(cov_matrix)
        ones = [1.0] * n

        # Compute the three fundamental scalars:
        # a = mu^T Sigma^{-1} mu
        # b = mu^T Sigma^{-1} 1
        # c = 1^T Sigma^{-1} 1
        inv_mu = self._mat_vec_mul(inv_cov, expected_returns)
        inv_one = self._mat_vec_mul(inv_cov, ones)

        a = self._vec_dot(expected_returns, inv_mu)
        b = self._vec_dot(expected_returns, inv_one)
        c = self._vec_dot(ones, inv_one)

        # Solve for Lagrangian multipliers
        # For target return mu_target, the optimal weights are:
        # w = Sigma^{-1} * (lambda * mu + gamma * 1)
        # where lambda and gamma solve:
        #   mu_target = lambda * a + gamma * b
        #   1         = lambda * b + gamma * c
        denom = a * c - b * b
        if abs(denom) < 1e-12:
            raise ValueError(
                "Efficient frontier is degenerate — a*c - b^2 ≈ 0. "
                "This suggests all assets have identical Sharpe ratios "
                "or the covariance is singular."
            )

        lam = (c * target_return - b) / denom
        gam = (a - b * target_return) / denom

        # Optimal weights
        weights = [lam * inv_mu[i] + gam * inv_one[i] for i in range(n)]

        port_var = self._portfolio_variance(weights, cov_matrix)
        port_vol = sqrt(port_var)
        port_ret = self._vec_dot(weights, expected_returns)
        sharpe = (port_ret - self._risk_free_rate) / port_vol if port_vol > 0 else 0.0

        return PortfolioResult(
            weights={t: w for t, w in zip(tickers, weights)},
            expected_return=round(port_ret, 6),
            volatility=round(port_vol, 6),
            sharpe_ratio=round(sharpe, 6),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _portfolio_variance(
        weights: list[float], cov_matrix: list[list[float]]
    ) -> float:
        """Compute portfolio variance = w^T Sigma w."""
        # Compute w^T Sigma
        n = len(weights)
        w_sigma = [0.0] * n
        for i in range(n):
            w_sigma[i] = sum(weights[j] * cov_matrix[j][i] for j in range(n))
        return sum(weights[i] * w_sigma[i] for i in range(n))

    # ------------------------------------------------------------------
    # General matrix inversion via Gauss-Jordan
    # ------------------------------------------------------------------

    @staticmethod
    def _invert_gauss_jordan(A: list[list[float]]) -> list[list[float]]:
        """
        Invert a square matrix using Gauss-Jordan elimination.

        This works for any N x N invertible matrix. For the portfolio
        problem, N is typically small (5-50 assets).

        WHY Gauss-Jordan vs. LU decomposition:
            Gauss-Jordan is the simplest to implement correctly from
            scratch. For matrices under ~100x100, the performance
            difference is negligible.

        Raises:
            ValueError: If the matrix is singular.
        """
        n = len(A)
        if n == 0 or any(len(row) != n for row in A):
            raise ValueError("Input must be a square matrix")

        # Augmented matrix [A | I]
        aug = [
            row[:] + [1.0 if i == j else 0.0 for j in range(n)]
            for i, row in enumerate(A)
        ]

        for col in range(n):
            # Partial pivot: find the row with the largest absolute value
            pivot_row = max(
                range(col, n),
                key=lambda r: abs(aug[r][col]),
            )
            if abs(aug[pivot_row][col]) < 1e-12:
                raise ValueError("Singular matrix — cannot invert")

            # Swap rows
            aug[col], aug[pivot_row] = aug[pivot_row], aug[col]

            # Scale pivot row
            pivot = aug[col][col]
            for j in range(2 * n):
                aug[col][j] /= pivot

            # Eliminate all other rows
            for row in range(n):
                if row != col:
                    factor = aug[row][col]
                    for j in range(2 * n):
                        aug[row][j] -= factor * aug[col][j]

        # Extract inverse from the right half
        return [[aug[i][j + n] for j in range(n)] for i in range(n)]
