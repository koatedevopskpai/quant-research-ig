"""
black_litterman.py — Black-Litterman portfolio model (1992).

WHY BLACK-LITTERMAN EXISTS:
    Mean-variance optimisation requires expected returns as input, but
    historical average returns are notoriously poor estimates (low
    signal-to-noise ratio). The Black-Litterman model solves this by
    combining:

    1. MARKET EQUILIBRIUM RETURNS (prior):
       Implied returns from CAPM: mu_eq = delta * Sigma * w_mkt
       where:
           delta: Risk aversion coefficient (market Sharpe ratio)
           Sigma: Covariance matrix
           w_mkt: Market capitalisation weights

    2. INVESTOR VIEWS (conditional distribution):
       The investor provides views on specific assets:
           "Asset A will outperform Asset B by 3%"
           "Asset C will return 8% with 90% confidence"

    3. BAYESIAN MIXTURE (posterior):
       The posterior expected returns blend the prior and the views,
       weighted by their respective confidence levels:
           mu_post = [(tau*Sigma)^{-1} + P^T Omega^{-1} P]^{-1}
                   * [(tau*Sigma)^{-1} * mu_eq + P^T * Omega^{-1} * Q]

KEY INSIGHT:
    The resulting expected returns are much more stable than historical
    averages because:
    1. The prior (CAPM equilibrium) is well-structured and sensible.
    2. Views are expressed with explicit confidence levels.
    3. The Bayesian framework automatically downweights noisy views.

QUANT BECKMAN'S VIEW:
    Beckman often uses Black-Litterman as the "return estimation" step
    in his research pipeline. He combines it with:
    - A shrunk covariance matrix (Ledoit-Wolf) for Sigma.
    - A small number of high-conviction views.
    - Walk-forward analysis to validate posterior returns.

REFERENCES:
    Black, F., & Litterman, R. (1992). Global Portfolio Optimization.
        Financial Analysts Journal, 48(5).
    Idzorek, T. (2005). A Step-by-Step Guide to the Black-Litterman Model.
"""

from dataclasses import dataclass
from math import sqrt


@dataclass
class BLResult:
    """
    Output of the Black-Litterman model.

    Attributes:
        posterior_returns: Combined expected returns (posterior).
        posterior_cov: Posterior covariance matrix (uncertainty around
            posterior returns).
        prior_returns: Market equilibrium returns (CAPM-implied).
        weights: Optimal portfolio weights given posterior returns and
            the original covariance matrix (using max Sharpe).
    """

    posterior_returns: list[float]
    posterior_cov: list[list[float]]
    prior_returns: list[float]
    weights: dict[str, float]


class BlackLitterman:
    """
    Black-Litterman asset allocation model.

    Combines market equilibrium returns with investor views to produce
    stable expected returns for portfolio optimisation.

    USAGE:
        # Market data
        cov = [[0.04, 0.02], [0.02, 0.05]]
        mkt_weights = [0.6, 0.4]  # market cap weights
        tickers = ["Stocks", "Bonds"]

        # Views: Stocks will outperform Bonds by 2%
        P = [[1, -1]]   # pick matrix
        Q = [0.02]      # view returns
        Omega = [[0.01]] # view uncertainty (variance)

        model = BlackLitterman()
        result = model.blend(
            cov, mkt_weights, tickers,
            tau=0.05, risk_aversion=2.5,
            P=P, Q=Q, Omega=Omega,
        )
        print(result.posterior_returns)
        print(result.weights)

    SCALING NOTE:
        tau * Sigma is the uncertainty around the prior. tau is typically
        set to 1/N (where N is the number of observations) or a small
        value like 0.01-0.05. Smaller tau = more confidence in the prior.
    """

    # ------------------------------------------------------------------
    # Matrix helpers (borrowed from mean_variance.py style)
    # ------------------------------------------------------------------

    @staticmethod
    def _mat_mul(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
        n, m, p = len(A), len(A[0]), len(B[0])
        return [
            [sum(A[i][k] * B[k][j] for k in range(m)) for j in range(p)]
            for i in range(n)
        ]

    @staticmethod
    def _mat_vec_mul(A: list[list[float]], v: list[float]) -> list[float]:
        return [sum(row[j] * v[j] for j in range(len(v))) for row in A]

    @staticmethod
    def _vec_dot(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b))

    @staticmethod
    def _transpose(A: list[list[float]]) -> list[list[float]]:
        return [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]

    @staticmethod
    def _scalar_mul_mat(c: float, A: list[list[float]]) -> list[list[float]]:
        return [[c * x for x in row] for row in A]

    @staticmethod
    def _identity(n: int) -> list[list[float]]:
        return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    @staticmethod
    def _mat_add(
        A: list[list[float]], B: list[list[float]]
    ) -> list[list[float]]:
        return [
            [A[i][j] + B[i][j] for j in range(len(A[0]))]
            for i in range(len(A))
        ]

    @staticmethod
    def _mat_sub(
        A: list[list[float]], B: list[list[float]]
    ) -> list[list[float]]:
        return [
            [A[i][j] - B[i][j] for j in range(len(A[0]))]
            for i in range(len(A))
        ]

    def _invert(self, A: list[list[float]]) -> list[list[float]]:
        """Matrix inversion via Gauss-Jordan."""
        n = len(A)
        if n == 0 or any(len(row) != n for row in A):
            raise ValueError("Square matrix required")

        aug = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(A)]

        for col in range(n):
            pivot_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
            if abs(aug[pivot_row][col]) < 1e-12:
                raise ValueError("Singular matrix")
            aug[col], aug[pivot_row] = aug[pivot_row], aug[col]
            pivot = aug[col][col]
            for j in range(2 * n):
                aug[col][j] /= pivot
            for row in range(n):
                if row != col:
                    factor = aug[row][col]
                    for j in range(2 * n):
                        aug[row][j] -= factor * aug[col][j]

        return [[aug[i][j + n] for j in range(n)] for i in range(n)]

    # ------------------------------------------------------------------
    # Market equilibrium returns (CAPM-implied)
    # ------------------------------------------------------------------

    def _implied_returns(
        self,
        cov_matrix: list[list[float]],
        market_weights: list[float],
        risk_aversion: float,
    ) -> list[float]:
        """
        Compute CAPM-implied equilibrium returns.

        mu_eq = delta * Sigma * w_mkt

        WHY this formula:
            In CAPM equilibrium, the market portfolio is the optimal risky
            portfolio. Reversing the mean-variance optimisation gives:
                mu_eq = delta * Sigma * w_mkt

            delta (risk aversion) = (mu_mkt - r_f) / sigma_mkt^2
            This is the market's Sharpe ratio divided by market volatility.

        Args:
            cov_matrix: NxN covariance matrix.
            market_weights: Market capitalisation weights.
            risk_aversion: Market risk aversion coefficient (typically
                2.0-4.0 for equity markets).

        Returns:
            List of N equilibrium returns.
        """
        n = len(cov_matrix)
        sigma_w = [sum(cov_matrix[i][j] * market_weights[j] for j in range(n)) for i in range(n)]
        return [risk_aversion * sw for sw in sigma_w]

    # ------------------------------------------------------------------
    # Black-Litterman posterior
    # ------------------------------------------------------------------

    def blend(
        self,
        cov_matrix: list[list[float]],
        market_weights: list[float],
        tickers: list[str],
        tau: float = 0.05,
        risk_aversion: float = 2.5,
        P: list[list[float]] | None = None,
        Q: list[float] | None = None,
        Omega: list[list[float]] | None = None,
    ) -> BLResult:
        """
        Compute Black-Litterman posterior expected returns.

        The posterior is:
            mu_post = [(tau*Sigma)^{-1} + P^T Omega^{-1} P]^{-1}
                    * [(tau*Sigma)^{-1} * mu_eq + P^T Omega^{-1} Q]

        and the posterior covariance is:
            Sigma_post = Sigma + [(tau*Sigma)^{-1} + P^T Omega^{-1} P]^{-1}

        NOTE: If no views are provided (P=None), the posterior equals the
        prior (market equilibrium).

        Args:
            cov_matrix: NxN asset return covariance matrix.
            market_weights: N market capitalisation weights.
            tickers: N asset names.
            tau: Uncertainty scaling factor for the prior (default 0.05).
            risk_aversion: Market risk aversion coefficient (default 2.5).
            P: KxN pick matrix encoding K investor views. Each row is a
               linear combination of asset returns.
            Q: K-vector of expected returns for each view.
            Omega: KxK diagonal covariance matrix of view uncertainty.
                   Larger values = less confident views.

        Returns:
            BLResult with posterior returns, covariance, and weights.
        """
        n = len(tickers)

        # Step 1: Prior (market equilibrium)
        prior = self._implied_returns(cov_matrix, market_weights, risk_aversion)

        # If no views, return prior
        if P is None or Q is None or Omega is None:
            # Use max Sharpe weights from prior
            from quant_methods.portfolio.mean_variance import MeanVarianceOptimiser
            opt = MeanVarianceOptimiser()
            result = opt.max_sharpe(cov_matrix, prior, tickers)
            return BLResult(
                posterior_returns=prior,
                posterior_cov=cov_matrix,
                prior_returns=prior,
                weights=result.weights,
            )

        k = len(Q)  # number of views
        if len(P) != k or len(Omega) != k or any(len(row) != len(Omega) for row in Omega):
            raise ValueError(
                "P (KxN), Q (K), and Omega (KxK) dimensions must match"
            )

        # ------------------------------------------------------------------
        # Compute posterior
        # ------------------------------------------------------------------
        # HOW BLACK-LITTERMAN WORKS UNDER THE HOOD:
        #
        # 1. The prior is: N(mu_eq, tau * Sigma)
        # 2. The views are: P * mu ~ N(Q, Omega)
        # 3. The posterior is the conditional distribution given the views.
        #    This is equivalent to a weighted least squares problem:
        #
        #    mu_post = argmin  (mu - mu_eq)^T (tau*Sigma)^{-1} (mu - mu_eq)
        #                   + (P*mu - Q)^T Omega^{-1} (P*mu - Q)
        #
        #    The solution is the Bayesian posterior mean (first equation above).
        # ------------------------------------------------------------------

        # Compute tau * Sigma (prior covariance of expected returns)
        tau_sigma = self._scalar_mul_mat(tau, cov_matrix)

        # Invert tau_sigma to get the precision matrix of the prior
        tau_sigma_inv = self._invert(tau_sigma)

        # Compute P^T Omega^{-1} P + tau_sigma_inv (posterior precision)
        # Step 1: Omega^{-1}
        omega_inv = self._invert(Omega)
        # Step 2: P^T Omega^{-1}
        pt = self._transpose(P)
        pt_omega_inv = self._mat_mul(pt, omega_inv)
        # Step 3: P^T Omega^{-1} P
        pt_omega_inv_p = self._mat_mul(pt_omega_inv, P)
        # Step 4: Add tau_sigma_inv
        posterior_precision = self._mat_add(tau_sigma_inv, pt_omega_inv_p)
        posterior_cov = self._invert(posterior_precision)

        # Compute the right-hand side: (tau*Sigma)^{-1} * mu_eq + P^T * Omega^{-1} * Q
        # Step 1: tau_sigma_inv * mu_eq
        ts_inv_mu = self._mat_vec_mul(tau_sigma_inv, prior)
        # Step 2: P^T * Omega^{-1} * Q
        pt_omega_inv_q = self._mat_vec_mul(pt_omega_inv, Q)
        # Step 3: Sum
        rhs = [ts_inv_mu[i] + pt_omega_inv_q[i] for i in range(n)]

        # Posterior returns = posterior_cov * rhs
        posterior_returns = self._mat_vec_mul(posterior_cov, rhs)

        # ------------------------------------------------------------------
        # Compute optimal weights from posterior returns
        # ------------------------------------------------------------------
        from quant_methods.portfolio.mean_variance import MeanVarianceOptimiser
        opt = MeanVarianceOptimiser()
        pf_result = opt.max_sharpe(cov_matrix, posterior_returns, tickers)

        return BLResult(
            posterior_returns=[round(r, 6) for r in posterior_returns],
            posterior_cov=[[round(x, 6) for x in row] for row in posterior_cov],
            prior_returns=[round(r, 6) for r in prior],
            weights=pf_result.weights,
        )
