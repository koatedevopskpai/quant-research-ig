"""
black_scholes.py — Black-Scholes-Merton option pricer with Greeks.

WHY THIS EXISTS:
    The Black-Scholes model is the foundation of modern option pricing. It
    provides closed-form formulas for European option prices and their
    sensitivities (Greeks). This implementation is used for:
    1. Pricing vanilla European calls and puts on IG index CFDs.
    2. Computing Greeks for risk management and hedging.
    3. As a benchmark for validating the Binomial Tree and Monte Carlo
       engines (which should converge to BSM for European options).

THE MODEL (Black & Scholes 1973, Merton 1973):
    Under the risk-neutral measure, the price of a European call is:
        C = S * e^(-qT) * N(d1) - K * e^(-rT) * N(d2)
    where:
        d1 = [ln(S/K) + (r - q + sigma^2/2) * T] / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)

    N(x) is the standard Normal CDF, computed here via the error function
    (math.erf) for numerical stability.

KEY ASSUMPTIONS (limitations to be aware of):
    1. Lognormal distribution of returns — does not capture fat tails or
       skewness observed in real markets (see GARCH model for this).
    2. Constant volatility — the "volatility smile" shows markets price
       different implied volatilities for different strikes.
    3. Continuous trading and no transaction costs — not realistic for
       retail CFD trading (bid-ask spreads matter).
    4. European exercise only — for American options, use the Binomial Tree.

REFERENCES:
    Hull, J. (2022). Options, Futures, and Other Derivatives (11th ed.),
        Chapters 13-15, 17-19.
    Black, F. & Scholes, M. (1973). The Pricing of Options and Corporate
        Liabilities. Journal of Political Economy, 81(3).
    Merton, R. (1973). Theory of Rational Option Pricing. Bell Journal of
        Economics and Management Science, 4(1).

METHODOLOGY NOTE (Quant Beckman):
    The BSM model is an example of Beckman's "theory-driven (substantive)
    model" approach. We start from a well-established financial theory and
    implement it with rigorous numerical methods. The model's limitations
    are explicitly documented so that researchers know when to use more
    advanced models (GARCH for non-constant vol, Monte Carlo for
    path-dependent payoffs).
"""

from dataclasses import dataclass
from math import exp, log, sqrt, erf, pi
from typing import Literal

# WHY Literal["call", "put"] instead of an Enum:
#   Literal types are more ergonomic for interactive use (no Enum import,
#   no .value access). The caller writes compute("call") instead of
#   compute(OptionType.CALL). This is the convention in the quant finance
#   literature and in financial data APIs (Bloomberg, Reuters).
OptionType = Literal["call", "put"]


def _ncdf(x: float) -> float:
    """
    Standard Normal cumulative distribution function.

    WHY erf (error function) instead of stats.norm.cdf:
        math.erf is implemented in C and is approximately 10x faster than
        scipy.stats.norm.cdf for scalar values. For this pricer, we only
        need scalar CDF evaluations (not array operations), so erf is the
        better choice.

    FORMULA:
        N(x) = 0.5 * (1 + erf(x / sqrt(2)))

    NUMERICAL ACCURACY:
        erf is accurate to approximately 1e-15 for |x| < 6, which covers
        all practical option pricing scenarios (d1/d2 rarely exceed |6|).
        Beyond |x| > 6, N(x) approaches 0 or 1 to within floating-point
        precision anyway.
    """
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _npdf(x: float) -> float:
    """
    Standard Normal probability density function.

    f(x) = exp(-x^2/2) / sqrt(2*pi)

    Used in computing Greeks (particularly gamma and vega).
    """
    return exp(-0.5 * x * x) / sqrt(2.0 * pi)


@dataclass(frozen=True)
class BSMResult:
    """
    Container for a single option's price and Greeks.

    WHY frozen=True:
        Once computed, a result should never change. Frozen dataclasses
        are immutable, preventing accidental modification and making them
        safe to pass between threads or cache in dictionaries.

    UNITS:
        premium: Same units as spot price (e.g., USD for US 500).
        delta: Change in premium per $1 change in spot (0 to 1 for calls).
        gamma: Change in delta per $1 change in spot.
        vega: Change in premium per 1% change in volatility (divided by 100).
        theta: Change in premium per calendar day (divided by 365).
        rho: Change in premium per 1% change in interest rate (divided by 100).

    WHY vega and rho are divided by 100:
        In practice, volatility changes of 1% (not 100%) are meaningful.
        A vega of 0.05 means the option gains $0.05 if vol increases from
        20% to 21%. Similarly for rho with interest rates.
    """

    premium: float
    delta: float
    gamma: float
    vega: float  # per 1% vol change
    theta: float  # per calendar day
    rho: float  # per 1% rate change


class BlackScholesPricer:
    """
    Black-Scholes-Merton European option pricer with Greeks.

    All parameters are annualised:
        S       — Spot price of the underlying asset.
        K       — Strike price of the option.
        T       — Time to expiry in years (e.g., 30/365 for 30 days).
        r       — Annual risk-free interest rate (e.g., 0.05 for 5%).
        sigma   — Annualised volatility (e.g., 0.20 for 20%).
        q       — Continuous dividend yield (0 for index CFDs without dividends).

    USAGE:
        pricer = BlackScholesPricer(S=100, K=100, T=30/365, r=0.05, sigma=0.20)
        result = pricer.compute("call")
        print(f"Call premium: {result.premium}")
        print(f"Delta: {result.delta}, Gamma: {result.gamma}")

    VALIDATION (Hull textbook values):
        For S=100, K=100, T=1 year, r=5%, sigma=20%, q=0:
          - ATM call premium: ~10.45
          - ATM put premium: ~5.57 (put-call parity: p = c - S + K*e^(-rT))
          - Call delta: ~0.64
          - Put delta: ~-0.36
          - Gamma (same for call and put): ~0.018

    NUMERICAL NOTES:
        - Handles extreme values (deep ITM/OTM) gracefully.
        - Raises ValueError if S, K, T, or sigma are <= 0 (financial
          nonsense).
        - Theta is returned per CALENDAR day (division by 365). Some
          implementations use trading days (252). Calendar days are more
          conservative for risk management.
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
        """
        Initialise the pricer with option parameters.

        WHY validation in __init__ vs. compute():
            Fail fast — if parameters are invalid, we want to know
            immediately when the pricer is created, not when we try to
            price it. This prevents silent errors propagating through
            batch pricing workflows.
        """
        if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
            raise ValueError(
                f"All of S={S}, K={K}, T={T}, sigma={sigma} must be > 0"
            )
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.q = q

    def _d12(self) -> tuple[float, float]:
        """
        Compute the d1 and d2 terms in the BSM formula.

        d1 = [ln(S/K) + (r - q + sigma^2/2) * T] / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)

        WHY separating _d12 from compute():
            Both the call and put pricing formulas use d1 and d2.
            Computing them once and reusing avoids redundant work and
            ensures consistency between call and put calculations.
        """
        sig_sq = self.sigma * self.sigma
        d1 = (
            log(self.S / self.K)
            + (self.r - self.q + 0.5 * sig_sq) * self.T
        ) / (self.sigma * sqrt(self.T))
        d2 = d1 - self.sigma * sqrt(self.T)
        return d1, d2

    def price(self) -> float:
        """
        Compute the European call option premium.

        Wrapper around compute("call").premium for convenience when only
        the price (not Greeks) is needed.
        """
        return self.compute("call").premium

    def price_put(self) -> float:
        """
        Compute the European put option premium.

        Wrapper around compute("put").premium for convenience.
        """
        return self.compute("put").premium

    def compute(self, option_type: OptionType = "call") -> BSMResult:
        """
        Compute option premium and all Greeks.

        GREEK INTERPRETATIONS (for portfolio reviewers):
            - Delta: Directional risk. A call delta of 0.64 means the
              option moves $0.64 for every $1 move in the underlying.
            - Gamma: Convexity risk. High gamma means delta changes
              rapidly — the position is hard to hedge.
            - Vega: Volatility risk. Vega is highest for ATM options and
              increases with time to expiry.
            - Theta: Time decay. Long options have negative theta (they
              lose value each day). Short options have positive theta.
            - Rho: Interest rate risk. Usually the smallest Greek for
              short-dated options. More significant for long-dated options.

        WHY rounding to 4 decimal places:
            IG quotes prices to 4 decimal places for indices and 5 for
            forex. Four decimal places is a reasonable default that avoids
            displaying spurious precision from floating-point arithmetic.
            Researchers can access the unrounded values through the
            _d12() and intermediate calculations if needed.
        """
        d1, d2 = self._d12()
        sqrt_t = sqrt(self.T)

        # ------------------------------------------------------------------
        # Premium and Greeks for CALL options
        # ------------------------------------------------------------------
        if option_type == "call":
            # Premium: C = S * e^(-qT) * N(d1) - K * e^(-rT) * N(d2)
            premium = (
                self.S * exp(-self.q * self.T) * _ncdf(d1)
                - self.K * exp(-self.r * self.T) * _ncdf(d2)
            )
            # Delta: dC/dS = e^(-qT) * N(d1)
            delta = exp(-self.q * self.T) * _ncdf(d1)
            # Theta: dC/dt (per calendar day)
            theta = (
                - (self.S * exp(-self.q * self.T) * _npdf(d1) * self.sigma)
                / (2 * sqrt_t)
                - self.r * self.K * exp(-self.r * self.T) * _ncdf(d2)
                + self.q * self.S * exp(-self.q * self.T) * _ncdf(d1)
            ) / 365.0
            # Rho: dC/dr (per 1%)
            rho = (
                self.K * self.T * exp(-self.r * self.T) * _ncdf(d2) / 100.0
            )

        # ------------------------------------------------------------------
        # Premium and Greeks for PUT options
        # ------------------------------------------------------------------
        else:
            # Premium: P = K * e^(-rT) * N(-d2) - S * e^(-qT) * N(-d1)
            premium = (
                self.K * exp(-self.r * self.T) * _ncdf(-d2)
                - self.S * exp(-self.q * self.T) * _ncdf(-d1)
            )
            # Delta: dP/dS = -e^(-qT) * N(-d1)
            delta = -exp(-self.q * self.T) * _ncdf(-d1)
            # Theta: dP/dt (per calendar day)
            theta = (
                - (self.S * exp(-self.q * self.T) * _npdf(d1) * self.sigma)
                / (2 * sqrt_t)
                + self.r * self.K * exp(-self.r * self.T) * _ncdf(-d2)
                - self.q * self.S * exp(-self.q * self.T) * _ncdf(-d1)
            ) / 365.0
            # Rho: dP/dr (per 1%) — negative for puts
            rho = (
                -self.K * self.T * exp(-self.r * self.T) * _ncdf(-d2) / 100.0
            )

        # ------------------------------------------------------------------
        # Greeks shared by call and put
        # ------------------------------------------------------------------
        # Gamma: same for call and put. High gamma near ATM at expiry.
        gamma = (
            exp(-self.q * self.T)
            * _npdf(d1)
            / (self.S * self.sigma * sqrt_t)
        )
        # Vega: same for call and put. High vega for long-dated ATM options.
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
