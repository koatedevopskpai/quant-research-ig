"""
test_black_scholes.py — Unit tests for the Black-Scholes pricer.

Covers:
  - Known edge cases (ATM, deep ITM, deep OTM)
  - Greeks at ATM
  - Put-call parity
"""

from math import exp, isclose

import pytest

from quant_methods.pricers.black_scholes import BlackScholesPricer


class TestBlackScholesPricer:
    S = 100.0
    K = 100.0
    T = 1.0
    r = 0.05
    sigma = 0.20

    def test_atm_call_premium(self) -> None:
        p = BlackScholesPricer(
            S=self.S, K=self.K, T=self.T,
            r=self.r, sigma=self.sigma,
        )
        r = p.compute("call")
        # Expected ~10.45 for these params (manually verified against Hull)
        assert isclose(r.premium, 10.45, abs_tol=0.01)

    def test_atm_put_premium(self) -> None:
        p = BlackScholesPricer(
            S=self.S, K=self.K, T=self.T,
            r=self.r, sigma=self.sigma,
        )
        r = p.compute("put")
        assert isclose(r.premium, 5.57, abs_tol=0.01)

    def test_deep_itm_call(self) -> None:
        p = BlackScholesPricer(S=200, K=10, T=1, r=0.05, sigma=0.20)
        r = p.compute("call")
        assert isclose(r.delta, 1.0, abs_tol=0.001)
        assert r.gamma < 0.001

    def test_deep_otm_call(self) -> None:
        p = BlackScholesPricer(S=10, K=200, T=1, r=0.05, sigma=0.20)
        r = p.compute("call")
        assert isclose(r.delta, 0.0, abs_tol=0.001)
        assert r.premium < 0.01

    def test_put_call_parity(self) -> None:
        p = BlackScholesPricer(S=100, K=100, T=1, r=0.05, sigma=0.20)
        call = p.compute("call")
        put = p.compute("put")
        parity = call.premium - put.premium
        expected = self.S - self.K * exp(-self.r * self.T)
        assert isclose(parity, expected, abs_tol=0.001)

    def test_gamma_positive(self) -> None:
        p = BlackScholesPricer(S=100, K=100, T=1, r=0.05, sigma=0.20)
        call = p.compute("call")
        put = p.compute("put")
        assert call.gamma > 0
        assert isclose(call.gamma, put.gamma, abs_tol=0.0001)

    def test_vega_positive(self) -> None:
        p = BlackScholesPricer(S=100, K=100, T=1, r=0.05, sigma=0.20)
        call = p.compute("call")
        assert call.vega > 0

    def test_theta_negative_atm(self) -> None:
        p = BlackScholesPricer(S=100, K=100, T=1, r=0.05, sigma=0.20)
        call = p.compute("call")
        assert call.theta < 0  # long options lose value with time

    def test_rho_positive_call(self) -> None:
        p = BlackScholesPricer(S=100, K=100, T=1, r=0.05, sigma=0.20)
        call = p.compute("call")
        put = p.compute("put")
        assert call.rho > 0
        assert put.rho < 0

    def test_invalid_inputs(self) -> None:
        with pytest.raises(ValueError):
            BlackScholesPricer(S=-1, K=100, T=1, r=0.05, sigma=0.20)
        with pytest.raises(ValueError):
            BlackScholesPricer(S=100, K=100, T=0, r=0.05, sigma=0.20)
        with pytest.raises(ValueError):
            BlackScholesPricer(S=100, K=100, T=1, r=0.05, sigma=0)