"""
black_scholes.py — Black-Scholes-Merton option pricer with Greeks.

Implements:
  - European call / put pricing via the closed-form BSM formula
  - Greeks: delta, gamma, vega, theta, rho

All angles in radians (math.erf works on Normal CDF references).

References:
  Hull, J. Options, Futures, and Other Derivatives (11th ed.)
  Black, F. & Scholes, M. (1973). The Pricing of Options and
  Corporate Liabilities. Journal of Political Economy, 81(3).
"""

from dataclasses import dataclass
from math import exp, log, sqrt, erf, pi
from typing import Literal

OptionType = Literal["call", "put"]


def _ncdf(x: float) -> float:
    """Standard Normal CDF via the error function."""
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _npdf(x: float) -> float:
    """Standard Normal PDF."""
    return exp(-0.5 * x * x) / sqrt(2.0 * pi)


@dataclass(frozen=True)
class BSMResult:
    """Results container for a single option price + Greeks."""

    premium: float
    delta: float
    gamma: float
    vega: float
    theta: float  # per calendar day (not trading day)
    rho: float


class BlackScholesPricer:
    """
    Black-Scholes-Merton European option pricer.

    Parameters are annualised:
        S       — spot price of the underlying
        K       — strike price
        T       — time to expiry in years
        r       — risk-free rate (annualised, e.g. 0.05 for 5 %)
        sigma   — annualised volatility (e.g. 0.20 for 20 %)
        q       — continuous dividend yield (default 0)
    """

    def __init__(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
    ) -> None:
        if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
            raise ValueError("S, K, T, sigma must all be positive")
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.q = q

    def _d12(self) -> tuple[float, float]:
        """Return (d1, d2)."""
        sig_sq = self.sigma * self.sigma
        d1 = (
            log(self.S / self.K)
            + (self.r - self.q + 0.5 * sig_sq) * self.T
        ) / (self.sigma * sqrt(self.T))
        d2 = d1 - self.sigma * sqrt(self.T)
        return d1, d2

    def price(self) -> float:
        d1, d2 = self._d12()
        F = self.S * exp((self.r - self.q) * self.T)
        return exp(-self.r * self.T) * F * _ncdf(d1) - self.K * exp(-self.r * self.T) * _ncdf(d2)

    def price_put(self) -> float:
        d1, d2 = self._d12()
        return (
            self.K * exp(-self.r * self.T) * _ncdf(-d2)
            - self.S * exp(-self.q * self.T) * _ncdf(-d1)
        )

    def compute(self, option_type: OptionType = "call") -> BSMResult:
        d1, d2 = self._d12()
        sqrt_t = sqrt(self.T)

        if option_type == "call":
            premium = (
                self.S * exp(-self.q * self.T) * _ncdf(d1)
                - self.K * exp(-self.r * self.T) * _ncdf(d2)
            )
            delta = exp(-self.q * self.T) * _ncdf(d1)
            theta = (
                - (self.S * exp(-self.q * self.T) * _npdf(d1) * self.sigma) / (2 * sqrt_t)
                - self.r * self.K * exp(-self.r * self.T) * _ncdf(d2)
                + self.q * self.S * exp(-self.q * self.T) * _ncdf(d1)
            ) / 365.0
            rho = self.K * self.T * exp(-self.r * self.T) * _ncdf(d2) / 100.0
        else:
            premium = (
                self.K * exp(-self.r * self.T) * _ncdf(-d2)
                - self.S * exp(-self.q * self.T) * _ncdf(-d1)
            )
            delta = -exp(-self.q * self.T) * _ncdf(-d1)
            theta = (
                - (self.S * exp(-self.q * self.T) * _npdf(d1) * self.sigma) / (2 * sqrt_t)
                + self.r * self.K * exp(-self.r * self.T) * _ncdf(-d2)
                - self.q * self.S * exp(-self.q * self.T) * _ncdf(-d1)
            ) / 365.0
            rho = -self.K * self.T * exp(-self.r * self.T) * _ncdf(-d2) / 100.0

        gamma = exp(-self.q * self.T) * _npdf(d1) / (self.S * self.sigma * sqrt_t)
        vega = (
            self.S * exp(-self.q * self.T) * _npdf(d1) * sqrt_t / 100.0
        )

        return BSMResult(
            premium=round(premium, 4),
            delta=round(delta, 4),
            gamma=round(gamma, 4),
            vega=round(vega, 4),
            theta=round(theta, 4),
            rho=round(rho, 4),
        )