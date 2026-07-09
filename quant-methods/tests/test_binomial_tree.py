"""
test_binomial_tree.py — Unit tests for the CRR binomial tree pricer.

Covers:
  - European convergence to Black-Scholes
  - American premium (American > European for puts)
  - Deep ITM / OTM boundary behaviour
  - Invalid input validation
"""

from math import exp, isclose

import pytest

from quant_methods.pricers.binomial_tree import BinomialTreePricer
from quant_methods.pricers.black_scholes import BlackScholesPricer


class TestBinomialTreePricer:
    S = 100.0
    K = 100.0
    T = 1.0
    r = 0.05
    sigma = 0.20

    # ------------------------------------------------------------------
    # European convergence to BSM
    # ------------------------------------------------------------------

    def test_european_call_converges_to_bsm(self) -> None:
        """With N=500, European call should be within 0.02 of BSM."""
        bt = BinomialTreePricer(
            S=self.S, K=self.K, T=self.T,
            r=self.r, sigma=self.sigma, N=500,
        )
        bsm = BlackScholesPricer(
            S=self.S, K=self.K, T=self.T,
            r=self.r, sigma=self.sigma,
        )
        bt_result = bt.compute("call", "european")
        bsm_result = bsm.compute("call")
        assert isclose(bt_result.premium, bsm_result.premium, abs_tol=0.02)

    def test_european_put_converges_to_bsm(self) -> None:
        bt = BinomialTreePricer(
            S=self.S, K=self.K, T=self.T,
            r=self.r, sigma=self.sigma, N=500,
        )
        bsm = BlackScholesPricer(
            S=self.S, K=self.K, T=self.T,
            r=self.r, sigma=self.sigma,
        )
        bt_result = bt.compute("put", "european")
        bsm_result = bsm.compute("put")
        assert isclose(bt_result.premium, bsm_result.premium, abs_tol=0.02)

    # ------------------------------------------------------------------
    # American premium
    # ------------------------------------------------------------------

    def test_american_call_equals_european_no_dividend(self) -> None:
        """American calls on non-dividend stocks are never exercised early."""
        bt = BinomialTreePricer(
            S=self.S, K=self.K, T=self.T,
            r=self.r, sigma=self.sigma, q=0.0, N=100,
        )
        eu = bt.compute("call", "european")
        am = bt.compute("call", "american")
        assert isclose(am.premium, eu.premium, abs_tol=0.01)
        assert not am.early_exercise

    def test_american_put_more_valuable_than_european(self) -> None:
        """American puts have early exercise premium (typically)."""
        bt = BinomialTreePricer(
            S=self.S, K=self.K, T=self.T,
            r=self.r, sigma=self.sigma, N=100,
        )
        eu = bt.compute("put", "european")
        am = bt.compute("put", "american")
        assert am.premium >= eu.premium - 0.001

    def test_american_put_deep_itm_early_exercise(self) -> None:
        """Deep ITM puts on high-rate stocks should trigger early exercise."""
        bt = BinomialTreePricer(
            S=80, K=100, T=0.5,
            r=0.10, sigma=0.15, N=100,
        )
        am = bt.compute("put", "american")
        assert am.early_exercise

    # ------------------------------------------------------------------
    # Boundary behaviour
    # ------------------------------------------------------------------

    def test_deep_itm_call(self) -> None:
        bt = BinomialTreePricer(
            S=200, K=10, T=1.0,
            r=0.05, sigma=0.20,
        )
        r = bt.compute("call", "european")
        # Intrinsic value = 190, time value should be tiny
        assert r.premium > 190.0

    def test_deep_otm_call(self) -> None:
        bt = BinomialTreePricer(
            S=10, K=200, T=1.0,
            r=0.05, sigma=0.20,
        )
        r = bt.compute("call", "european")
        assert r.premium < 0.01

    # ------------------------------------------------------------------
    # Put-call parity (European)
    # ------------------------------------------------------------------

    def test_put_call_parity(self) -> None:
        bt = BinomialTreePricer(
            S=self.S, K=self.K, T=self.T,
            r=self.r, sigma=self.sigma, N=200,
        )
        call = bt.compute("call", "european")
        put = bt.compute("put", "european")
        parity = call.premium - put.premium
        expected = self.S - self.K * exp(-self.r * self.T)
        assert isclose(parity, expected, abs_tol=0.10)

    # ------------------------------------------------------------------
    # Invalid inputs
    # ------------------------------------------------------------------

    def test_invalid_inputs(self) -> None:
        with pytest.raises(ValueError):
            BinomialTreePricer(S=-1, K=100, T=1, r=0.05, sigma=0.20)
        with pytest.raises(ValueError):
            BinomialTreePricer(S=100, K=100, T=0, r=0.05, sigma=0.20)
        with pytest.raises(ValueError):
            BinomialTreePricer(S=100, K=100, T=1, r=0.05, sigma=0)
        with pytest.raises(ValueError):
            BinomialTreePricer(S=100, K=100, T=1, r=0.05, sigma=0.20, N=0)
