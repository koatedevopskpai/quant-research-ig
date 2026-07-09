"""
binomial_tree.py — CRR Binomial Tree for American and European option pricing.

WHY THIS EXISTS:
    The Black-Scholes model (black_scholes.py) prices European options in
    closed form, but cannot handle:
    1. American exercise — options that can be exercised at any time before
       expiry (most exchange-traded options, some IG CFD options).
    2. Early exercise decisions — dividends, interest rates, and other
       factors can make early exercise optimal.

    The binomial tree solves both problems by modelling all possible price
    paths through a discrete-time lattice and using backwards induction
    to determine the option value at each node, checking for early exercise.

THE CRR MODEL (Cox, Ross, Rubinstein 1979):
    The asset price follows a binomial process over N discrete time steps:
        - Up move: S * u, where u = exp(sigma * sqrt(dt))
        - Down move: S * d, where d = 1/u (recombining tree)

    Risk-neutral probability of an up move:
        p = (exp((r - q) * dt) - d) / (u - d)

    The tree is "recombining" — an up then down arrives at the same price
    as down then up. This makes the lattice O(N^2) instead of O(2^N).

AMERICAN VS EUROPEAN PRICING:
    European: Value at each node = discounted expected value of children.
    American: Value at each node = max(discounted expected value,
                                      intrinsic value at exercise).

    This means American options are always at least as valuable as
    European options (the extra optionality can only add value).

CONVERGENCE:
    As N (number of steps) increases, the binomial tree converges to:
    - BSM price for European options (typical convergence: O(1/N))
    - True American option price (requires N >= 100 for ~1% accuracy)

    Modern enhancements (Leisen-Reimer, Tian) improve convergence speed,
    but the basic CRR model with N=500 steps provides adequate accuracy
    for most practical purposes.

REFERENCES:
    Cox, J., Ross, S., & Rubinstein, M. (1979). Option Pricing: A
        Simplified Approach. Journal of Financial Economics, 7(3).
    Hull, J. (2022). Options, Futures, and Other Derivatives (11th ed.),
        Chapter 21.

METHODOLOGY NOTE (Quant Beckman):
    The binomial tree exemplifies Beckman's "model taxonomy" concept:
    it is a theory-driven, parametric model that can be verified against
    closed-form benchmarks (BSM). When the binomial tree matches BSM for
    European options, we have confidence in its American option pricing.
"""

from dataclasses import dataclass
from math import exp, sqrt, log
from typing import Literal

OptionType = Literal["call", "put"]
ExerciseType = Literal["european", "american"]


@dataclass(frozen=True)
class BinomialResult:
    """
    Result from the binomial tree pricer.

    Attributes:
        premium: The option price.
        steps: Number of time steps used (convergence diagnostic).
        early_exercise: Whether early exercise was optimal at any node
                        (American options only).

    WHY include steps and early_exercise:
        These diagnostic fields help researchers assess convergence.
        If early_exercise is True, the option has "American premium" —
        additional value beyond the European price.
    """

    premium: float
    steps: int
    early_exercise: bool


class BinomialTreePricer:
    """
    CRR binomial tree pricer for European and American options.

    Parameters are annualised (same convention as BlackScholesPricer):
        S       — Spot price of the underlying.
        K       — Strike price.
        T       — Time to expiry in years.
        r       — Annual risk-free rate.
        sigma   — Annualised volatility.
        q       — Continuous dividend yield (default 0).
        N       — Number of time steps (default 100).

    USAGE:
        pricer = BinomialTreePricer(S=100, K=100, T=30/365, r=0.05, sigma=0.20)
        result = pricer.compute("call", "american")
        print(f"American call: {result.premium}")

    VALIDATION:
        For European options with N=1000, the binomial tree should converge
        to the Black-Scholes price within 0.01.

    WHY use a list of floats for the lattice instead of a 2D array:
        A recombining tree only needs to store one value per node at the
        current level. We use a single Python list and update it in-place
        during backwards induction. This is O(N) memory instead of O(N^2).

    PERFORMANCE:
        - O(N^2) time complexity (N steps, N nodes at the widest level).
        - O(N) memory complexity (single array of length N+1).
        - For N=1000: ~1 million operations, completes in <0.1s.
        - For N=10000: ~100 million operations, completes in ~1s.
    """

    def __init__(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
        N: int = 100,
    ) -> None:
        if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
            raise ValueError(
                f"All of S={S}, K={K}, T={T}, sigma={sigma} must be > 0"
            )
        if N < 1:
            raise ValueError(f"N={N} must be >= 1")
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.q = q
        self.N = N

    # ------------------------------------------------------------------
    # Tree parameters
    # ------------------------------------------------------------------

    def _tree_params(self) -> tuple[float, float, float]:
        """
        Compute CRR tree parameters: u, d, p.

        u = exp(sigma * sqrt(dt))     — up factor
        d = exp(-sigma * sqrt(dt))    — down factor (= 1/u, recombining)
        p = (exp((r-q)*dt) - d)/(u-d) — risk-neutral up probability

        WHY dt = T / N:
            Each step represents a period of dt = T/N years.
            As N increases, the discrete tree converges to continuous time.

        WHY d = 1/u:
            A recombining tree requires u * d = 1. This ensures that an
            up-move followed by a down-move returns to the original price.
            Non-recombining trees would have O(2^N) nodes.
        """
        dt = self.T / self.N
        u = exp(self.sigma * sqrt(dt))
        d = 1.0 / u
        p = (exp((self.r - self.q) * dt) - d) / (u - d)

        # WHY clipping p to [0, 1]:
        #   If p is outside [0, 1], the tree is unstable (arbitrage
        #   opportunities exist in the lattice). This happens when
        #   dt is too large relative to sigma. Increasing N fixes it.
        if not (0 <= p <= 1):
            raise ValueError(
                f"Risk-neutral probability p={p:.4f} is outside [0, 1]. "
                f"Increase N (currently {self.N}) or reduce dt."
            )
        return u, d, p

    # ------------------------------------------------------------------
    # Payoff functions
    # ------------------------------------------------------------------

    @staticmethod
    def _payoff(S_T: float, K: float, option_type: OptionType) -> float:
        """
        Option payoff at expiry.

        Call: max(S_T - K, 0)
        Put:  max(K - S_T, 0)

        WHY static method:
            The payoff function is pure arithmetic — it does not depend on
            the pricer's state. Making it static allows reuse by other
            pricers (e.g., Monte Carlo engine).
        """
        if option_type == "call":
            return max(S_T - K, 0.0)
        return max(K - S_T, 0.0)

    # ------------------------------------------------------------------
    # Main pricing method
    # ------------------------------------------------------------------

    def compute(
        self,
        option_type: OptionType = "call",
        exercise_type: ExerciseType = "european",
    ) -> BinomialResult:
        """
        Price the option using the CRR binomial tree.

        ALGORITHM:
        1. Forward pass: compute asset prices at each node at expiry.
        2. Backward induction: starting from expiry, discount option values
           back to the present, checking for early exercise (American only).

        WHY forward pass then backward induction:
            The forward pass builds the price lattice. The backward pass
            values the option. Separating the two phases makes the code
            clearer and allows the forward prices to be reused if needed.

        Args:
            option_type: "call" or "put".
            exercise_type: "european" or "american".

        Returns:
            BinomialResult with the option premium and diagnostics.

        RAISES:
            ValueError: If the risk-neutral probability is outside [0, 1].
        """
        u, d, p = self._tree_params()
        disc = exp(-self.r * self.T / self.N)  # discount per step

        # ------------------------------------------------------------------
        # Forward pass: build asset prices at expiry
        # ------------------------------------------------------------------
        # At step N, node j (0-indexed from the bottom), the asset price is:
        #   S_Nj = S * u^j * d^(N-j)
        # We only need the prices at expiry (step N) for payoffs.
        prices_at_expiry = [
            self.S * (u ** j) * (d ** (self.N - j))
            for j in range(self.N + 1)
        ]

        # ------------------------------------------------------------------
        # Backward induction: compute option values
        # ------------------------------------------------------------------
        # Start with payoffs at expiry nodes.
        V = [
            self._payoff(prices_at_expiry[j], self.K, option_type)
            for j in range(self.N + 1)
        ]

        early_exercise = False

        # Step backwards through the tree, from expiry to present.
        # At each step i (from N-1 down to 0), we have i+1 nodes.
        for i in range(self.N - 1, -1, -1):
            for j in range(i + 1):
                # Asset price at this node (for early exercise check).
                S_node = self.S * (u ** j) * (d ** (i - j))

                # Discounted expected value from the two child nodes.
                # V[j] is the down-child (one more down move), V[j+1] is
                # the up-child (one more up move). p is the risk-neutral
                # probability of an UP move, so it multiplies V[j+1].
                hold = disc * ((1 - p) * V[j] + p * V[j + 1])

                if exercise_type == "american":
                    # Intrinsic value if exercised immediately.
                    intrinsic = self._payoff(S_node, self.K, option_type)
                    if intrinsic > hold:
                        early_exercise = True
                    V[j] = intrinsic if intrinsic > hold else hold
                else:
                    V[j] = hold

        # V[0] is the option value at the root node (time 0).
        return BinomialResult(
            premium=round(V[0], 4),
            steps=self.N,
            early_exercise=early_exercise,
        )
