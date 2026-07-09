"""
test_portfolio.py — Unit tests for portfolio optimisation modules.

Covers:
  - MeanVariance: min variance, max Sharpe, efficient frontier, 2-asset
    cases with known closed-form solutions.
  - RiskParity: equal risk contribution property.
  - BlackLitterman: no views = prior, posterior blend with 1 view.
"""

from math import isclose, sqrt

import pytest

from quant_methods.portfolio.mean_variance import MeanVarianceOptimiser
from quant_methods.portfolio.risk_parity import RiskParity
from quant_methods.portfolio.black_litterman import BlackLitterman


# ======================================================================
# Mean-Variance Optimisation tests
# ======================================================================

class TestMeanVarianceOptimiser:
    """Test with a 2-asset case where closed-form answers are known.

    Two assets with:
        sigma_1 = 0.20, sigma_2 = 0.10, rho = 0.30
        mu_1 = 0.12, mu_2 = 0.06

    Covariance matrix:
        [[0.04, 0.006],
         [0.006, 0.01]]
    """

    cov = [[0.04, 0.006], [0.006, 0.01]]
    mu = [0.12, 0.06]
    tickers = ["Asset_A", "Asset_B"]

    # ------------------------------------------------------------------
    # Global minimum variance portfolio
    # ------------------------------------------------------------------

    def test_minimum_variance_weights_sum_to_one(self) -> None:
        opt = MeanVarianceOptimiser()
        result = opt.minimum_variance(self.cov, self.tickers)
        total_weight = sum(result.weights.values())
        assert isclose(total_weight, 1.0, abs_tol=1e-6)

    def test_minimum_variance_no_negative_weights(self) -> None:
        opt = MeanVarianceOptimiser()
        result = opt.minimum_variance(self.cov, self.tickers)
        for w in result.weights.values():
            assert w >= -1e-10  # allow tiny numerical negatives

    def test_minimum_variance_lower_vol_than_either_asset(self) -> None:
        opt = MeanVarianceOptimiser()
        result = opt.minimum_variance(self.cov, self.tickers)
        # Minimum variance should be lower than both individual vols
        assert result.volatility < 0.10  # lower than bond vol

    # ------------------------------------------------------------------
    # Maximum Sharpe ratio portfolio
    # ------------------------------------------------------------------

    def test_max_sharpe_weights_sum_to_one(self) -> None:
        opt = MeanVarianceOptimiser()
        result = opt.max_sharpe(self.cov, self.mu, self.tickers)
        total_weight = sum(result.weights.values())
        assert isclose(total_weight, 1.0, abs_tol=1e-6)

    def test_max_sharpe_positive(self) -> None:
        opt = MeanVarianceOptimiser()
        result = opt.max_sharpe(self.cov, self.mu, self.tickers)
        assert result.sharpe_ratio > 0

    def test_max_sharpe_better_than_either_asset(self) -> None:
        """Tangency portfolio should have higher Sharpe than individual
        assets."""
        opt = MeanVarianceOptimiser()
        result = opt.max_sharpe(self.cov, self.mu, self.tickers)

        # Individual Sharpe ratios: mu / sigma (r_f = 0)
        sharpe_a = 0.12 / 0.20  # = 0.6
        sharpe_b = 0.06 / 0.10  # = 0.6
        # Both are 0.6, so any mix should also be 0.6 (since rho = 0.3
        # diversification could improve Sharpe)
        # Actually let's verify Sharpe is at least as high.
        assert result.sharpe_ratio >= 0.6 - 0.01

    # ------------------------------------------------------------------
    # Efficient frontier
    # ------------------------------------------------------------------

    def test_efficient_frontier_higher_return_higher_vol(self) -> None:
        opt = MeanVarianceOptimiser()
        low = opt.efficient_frontier(self.cov, self.mu, self.tickers, 0.07)
        high = opt.efficient_frontier(self.cov, self.mu, self.tickers, 0.10)
        assert high.volatility >= low.volatility

    def test_efficient_frontier_target_between_min_max(self) -> None:
        opt = MeanVarianceOptimiser()
        result = opt.efficient_frontier(self.cov, self.mu, self.tickers, 0.08)
        assert 0.06 <= result.expected_return <= 0.12
        assert isclose(sum(result.weights.values()), 1.0, abs_tol=1e-6)

    def test_efficient_frontier_min_var_matches(self) -> None:
        """Efficient frontier at the minimum variance point should match
        the global minimum variance portfolio.

        The minimum variance portfolio has an achievable return based on
        the expected returns of the individual assets. We compute the
        actual expected return of the min variance weights and verify
        that the efficient frontier at that return matches the min
        variance portfolio's volatility.
        """
        opt = MeanVarianceOptimiser()
        min_var_result = opt.minimum_variance(self.cov, self.tickers)
        # Compute actual expected return of the min variance weights
        actual_return = sum(
            min_var_result.weights[t] * self.mu[i]
            for i, t in enumerate(self.tickers)
        )
        ef_result = opt.efficient_frontier(
            self.cov, self.mu, self.tickers, actual_return,
        )
        assert isclose(ef_result.volatility, min_var_result.volatility, abs_tol=1e-6)

    # ------------------------------------------------------------------
    # 2-asset closed form verification
    # ------------------------------------------------------------------

    def test_two_asset_min_variance_closed_form(self) -> None:
        """For two assets, the minimum variance weights are known:
        w1 = (sigma_2^2 - rho * sigma_1 * sigma_2) /
             (sigma_1^2 + sigma_2^2 - 2 * rho * sigma_1 * sigma_2)
        """
        s1, s2 = 0.20, 0.10
        rho = 0.30
        # cov_12 = rho * s1 * s2
        w1 = (s2 ** 2 - rho * s1 * s2) / (s1 ** 2 + s2 ** 2 - 2 * rho * s1 * s2)

        opt = MeanVarianceOptimiser()
        result = opt.minimum_variance(self.cov, self.tickers)
        assert isclose(result.weights["Asset_A"], w1, abs_tol=0.01)
        assert isclose(result.weights["Asset_B"], 1 - w1, abs_tol=0.01)

    # ------------------------------------------------------------------
    # Single-asset edge case
    # ------------------------------------------------------------------

    def test_single_asset(self) -> None:
        """With one asset, weight must be 1.0."""
        cov = [[0.04]]
        opt = MeanVarianceOptimiser()
        result = opt.minimum_variance(cov, ["Only"])
        assert isclose(result.weights["Only"], 1.0, abs_tol=1e-6)
        assert isclose(result.volatility, 0.20, abs_tol=1e-6)


# ======================================================================
# Risk Parity tests
# ======================================================================

class TestRiskParity:
    """Test risk parity: for a 2-asset case, verify that risk contributions
    are approximately equal."""

    def test_risk_contributions_equal(self) -> None:
        """Risk parity should equalise risk contributions (not weights)."""
        cov = [[0.04, 0.006], [0.006, 0.01]]
        tickers = ["A", "B"]

        rp = RiskParity()
        result = rp.optimise(cov, tickers)

        rc_values = list(result.risk_contributions.values())
        assert isclose(rc_values[0], rc_values[1], abs_tol=0.04)
        assert isclose(sum(result.weights.values()), 1.0, abs_tol=1e-6)

    def test_risk_parity_different_vols(self) -> None:
        """When one asset has much higher vol, risk parity should give it
        lower weight."""
        cov = [[0.16, 0.01], [0.01, 0.01]]
        tickers = ["HighVol", "LowVol"]

        rp = RiskParity()
        result = rp.optimise(cov, tickers)

        assert result.convergence
        # High vol asset should have lower weight
        assert result.weights["HighVol"] < result.weights["LowVol"]

    def test_risk_contributions_sum_to_one(self) -> None:
        cov = [[0.04, 0.006], [0.006, 0.01]]
        tickers = ["A", "B"]

        rp = RiskParity()
        result = rp.optimise(cov, tickers)
        total_rc = sum(result.risk_contributions.values())
        assert isclose(total_rc, 1.0, abs_tol=0.02)

    def test_three_assets(self) -> None:
        cov = [
            [0.04, 0.01, 0.005],
            [0.01, 0.02, 0.003],
            [0.005, 0.003, 0.01],
        ]
        tickers = ["A", "B", "C"]

        rp = RiskParity()
        result = rp.optimise(cov, tickers)

        assert result.convergence
        rc_values = list(result.risk_contributions.values())
        # All three should be close to 1/3
        for rc in rc_values:
            assert isclose(rc, 1.0 / 3.0, abs_tol=0.05)

    def test_single_asset_risk_parity(self) -> None:
        """Single asset: weight = 1, risk contribution = 1."""
        rp = RiskParity()
        result = rp.optimise([[0.04]], ["Only"])
        assert isclose(result.weights["Only"], 1.0, abs_tol=1e-6)
        assert isclose(result.risk_contributions["Only"], 1.0, abs_tol=1e-6)


# ======================================================================
# Black-Litterman tests
# ======================================================================

class TestBlackLitterman:
    """Test Black-Litterman model with a simple 2-asset case."""

    cov = [[0.04, 0.006], [0.006, 0.01]]
    mkt_weights = [0.6, 0.4]
    tickers = ["Stocks", "Bonds"]

    def test_no_views_equals_prior(self) -> None:
        """With no views, posterior should equal prior."""
        bl = BlackLitterman()
        result = bl.blend(
            self.cov, self.mkt_weights, self.tickers,
            tau=0.05, risk_aversion=2.5,
        )
        assert isclose(result.posterior_returns[0], result.prior_returns[0], abs_tol=1e-6)
        assert isclose(result.posterior_returns[1], result.prior_returns[1], abs_tol=1e-6)

    def test_no_views_weights_sum_to_one(self) -> None:
        bl = BlackLitterman()
        result = bl.blend(
            self.cov, self.mkt_weights, self.tickers,
            tau=0.05, risk_aversion=2.5,
        )
        total_weight = sum(result.weights.values())
        assert isclose(total_weight, 1.0, abs_tol=1e-6)

    def test_posterior_with_views(self) -> None:
        """With a bullish view on Stocks, the posterior should shift toward
        Stocks (but not fully, weighted by confidence)."""
        bl = BlackLitterman()
        # View: Stocks will return 15% (vs ~7-8% equilibrium)
        P = [[1.0, 0.0]]
        Q = [0.15]
        Omega = [[0.01]]  # moderate confidence

        result = bl.blend(
            self.cov, self.mkt_weights, self.tickers,
            tau=0.05, risk_aversion=2.5,
            P=P, Q=Q, Omega=Omega,
        )
        # Posterior Stocks return should be between prior and view
        assert result.posterior_returns[0] > result.prior_returns[0]
        assert result.posterior_returns[0] < 0.15 + 0.01

    def test_highly_confident_view(self) -> None:
        """With very low Omega (high confidence), posterior should approach
        the view."""
        bl = BlackLitterman()
        # Very confident view: Stocks return = 10%
        P = [[1.0, 0.0]]
        Q = [0.10]
        Omega = [[1e-6]]  # extremely confident

        result = bl.blend(
            self.cov, self.mkt_weights, self.tickers,
            tau=0.05, risk_aversion=2.5,
            P=P, Q=Q, Omega=Omega,
        )
        # Posterior stock return should be very close to the view
        assert isclose(result.posterior_returns[0], 0.10, abs_tol=0.005)
