"""
monte_carlo.py — Monte Carlo simulation engine for path-dependent options.

WHY THIS EXISTS:
    The Black-Scholes formula prices European options analytically, but many
    real-world options have path-dependent payoffs that cannot be priced in
    closed form:

    1. Asian options — payoff depends on the average price over time, not
       just the price at expiry. Reduces volatility of the option price.
    2. Barrier options — option activates or deactivates if the underlying
       crosses a threshold. Common in structured products.
    3. Lookback options — payoff based on the maximum or minimum price over
       the option's life.

    Monte Carlo simulation handles all of these by simulating thousands of
    possible price paths and computing the payoff for each, then averaging.

WHY MONTE CARLO (vs. Binomial Tree):
    - Binomial Tree: Good for American options (early exercise), but
      computationally expensive for path-dependent payoffs.
    - Monte Carlo: Excellent for path-dependent payoffs, but cannot handle
      American exercise efficiently (no backward induction step).

    For pricing American options, use the Binomial Tree (binomial_tree.py).
    For pricing European path-dependent options, use Monte Carlo.

THE GEOMETRIC BROWNIAN MOTION MODEL:
    dS = (r - q) * S * dt + sigma * S * dW

    Under the risk-neutral measure, the asset price evolves as:
    S(t+dt) = S(t) * exp((r - q - sigma^2/2) * dt + sigma * sqrt(dt) * Z)

    where Z ~ N(0, 1) is a standard normal random variable.

CONVERGENCE:
    The standard error of the Monte Carlo estimate decreases as
    1/sqrt(N_paths). To halve the error, quadruple the number of paths.

    Variance reduction techniques (antithetic variates, control variates)
    can improve convergence without increasing N_paths.

REFERENCES:
    Boyle, P. (1977). Options: A Monte Carlo Approach. Journal of Financial
        Economics, 4(3).
    Glasserman, P. (2003). Monte Carlo Methods in Financial Engineering.
        Springer.
    Hull, J. (2022). Options, Futures, and Other Derivatives (11th ed.),
        Chapter 22.

METHODOLOGY NOTE (Quant Beckman):
    Monte Carlo simulation is a key tool in Beckman's "Research" stage for
    hypothesis testing. By simulating price paths under different
    assumptions, we can estimate the distribution of possible outcomes
    without relying on closed-form approximations. The standard error
    provides a built-in measure of estimation uncertainty.
"""

import math
import random
from dataclasses import dataclass
from typing import Callable, Literal

OptionType = Literal["call", "put"]

# Type alias for a payoff function.
# Takes the full price path (list of prices at each time step) and returns
# the payoff at expiry. This enables arbitrary path-dependent payoffs.
PayoffFunction = Callable[[list[float]], float]


@dataclass(frozen=True)
class MCResult:
    """
    Result from a Monte Carlo simulation.

    Attributes:
        premium: Estimated option price (mean of simulated payoffs,
                 discounted to present).
        standard_error: Standard error of the estimate (sigma / sqrt(N)).
            A 95% confidence interval is approximately premium +/- 2 * SE.
        paths: Number of simulated paths used.

    WHY include standard_error:
        Statistical significance is central to Quant Beckman's methodology.
        The standard error tells the researcher how reliable the estimate
        is. If the SE is large relative to the premium, increase N_paths.
    """

    premium: float
    standard_error: float
    paths: int


class MonteCarloPricer:
    """
    Monte Carlo option pricer for path-dependent payoffs.

    Parameters (annualised):
        S       — Spot price.
        T       — Time to expiry in years.
        r       — Annual risk-free rate.
        sigma   — Annualised volatility.
        q       — Continuous dividend yield (default 0).
        N_steps — Number of time steps per path (default 252 for daily).
        N_paths — Number of simulated price paths (default 100000).

    USAGE:
        # Define a payoff function (e.g., Asian arithmetic average call)
        def asian_call(path):
            avg = sum(path) / len(path)
            return max(avg - K, 0)

        pricer = MonteCarloPricer(S=100, T=1, r=0.05, sigma=0.20)
        result = pricer.price(S=100, K=100, payoff=asian_call)
        print(f"Asian call: {result.premium} +/- {result.standard_error}")

    STANDARD PAYOFFS:
        Use class methods for standard payoffs:
        - standard_payoff("call"): max(S_T - K, 0)
        - standard_payoff("put"): max(K - S_T, 0)

        For path-dependent payoffs, write a custom function.

    PERFORMANCE:
        - O(N_paths * N_steps) time complexity.
        - For N_paths=100000, N_steps=252: ~25M operations, ~1-2 seconds.
        - For production, consider Numba JIT compilation or C++.
    """

    def __init__(
        self,
        S: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
        N_steps: int = 252,
        N_paths: int = 100000,
    ) -> None:
        if S <= 0 or T <= 0 or sigma <= 0:
            raise ValueError(f"S={S}, T={T}, sigma={sigma} must be > 0")
        if N_steps < 1:
            raise ValueError(f"N_steps={N_steps} must be >= 1")
        if N_paths < 1:
            raise ValueError(f"N_paths={N_paths} must be >= 1")
        self.S = S
        self.T = T
        self.r = r
        self.sigma = sigma
        self.q = q
        self.N_steps = N_steps
        self.N_paths = N_paths

    # ------------------------------------------------------------------
    # Standard payoff functions
    # ------------------------------------------------------------------

    @staticmethod
    def standard_payoff(option_type: OptionType, K: float) -> PayoffFunction:
        """
        Create a standard European call or put payoff function.

        WHY a factory method (returning a function) instead of computing
        directly:
            The payoff function must match the PayoffFunction signature
            (takes a path, returns a float). This factory creates functions
            with K bound to the closure, keeping the pricing code generic.

        USAGE:
            call_payoff = MonteCarloPricer.standard_payoff("call", K=100)
            result = pricer.price(S=100, K=100, payoff=call_payoff)
        """

        def _payoff(path: list[float]) -> float:
            S_T = path[-1]  # price at expiry (last element)
            if option_type == "call":
                return max(S_T - K, 0.0)
            return max(K - S_T, 0.0)

        return _payoff

    @staticmethod
    def asian_payoff(option_type: OptionType, K: float) -> PayoffFunction:
        """
        Create an Asian (arithmetic average) option payoff.

        Asian options are popular in practice because averaging reduces
        the volatility of the option price (making them cheaper than
        equivalent European options).

        USAGE:
            asian_call = MonteCarloPricer.asian_payoff("call", K=100)
            result = pricer.price(S=100, K=100, payoff=asian_call)
        """

        def _payoff(path: list[float]) -> float:
            avg = sum(path) / len(path)
            if option_type == "call":
                return max(avg - K, 0.0)
            return max(K - avg, 0.0)

        return _payoff

    @staticmethod
    def barrier_payoff(
        option_type: OptionType,
        K: float,
        barrier: float,
        barrier_type: Literal["up-and-out", "down-and-out"] = "up-and-out",
        rebate: float = 0.0,
    ) -> PayoffFunction:
        """
        Create a barrier option payoff.

        Barrier options activate/deactivate when the underlying crosses a
        threshold. "Up-and-out" means the option becomes worthless if the
        price goes above the barrier. "Down-and-out" means it knocks out
        if the price goes below the barrier.

        Args:
            option_type: "call" or "put".
            K: Strike price.
            barrier: Barrier level.
            barrier_type: "up-and-out" or "down-and-out".
            rebate: Cash rebate paid if the barrier is breached.

        WHY rebate:
            Most barrier options pay a small rebate (e.g., 1-5% of notional)
            if the barrier is breached. This makes the option cheaper than
            a vanilla option but still provides some downside protection.
        """

        def _payoff(path: list[float]) -> float:
            # Check if the barrier was breached at any point along the path.
            breached = False
            for p in path:
                if barrier_type == "up-and-out" and p >= barrier:
                    breached = True
                    break
                if barrier_type == "down-and-out" and p <= barrier:
                    breached = True
                    break

            if breached:
                return rebate

            S_T = path[-1]
            if option_type == "call":
                return max(S_T - K, 0.0)
            return max(K - S_T, 0.0)

        return _payoff

    # ------------------------------------------------------------------
    # Path simulation
    # ------------------------------------------------------------------

    def _simulate_path(self) -> list[float]:
        """
        Simulate a single GBM price path.

        Returns a list of prices at each time step, including the spot
        price at t=0.

        GBM DISCRETISATION:
            S[i+1] = S[i] * exp((r - q - sigma^2/2) * dt + sigma * sqrt(dt) * Z)

        WHY this discretisation (log-Euler) vs. Euler-Maruyama:
            The log-Euler scheme is exact for GBM — it introduces no
            discretisation error because the SDE has a closed-form solution.
            Euler-Maruyama (S += drift * dt + diffusion * sqrt(dt) * Z) would
            introduce O(dt) bias.

        WHY random.gauss vs. numpy.random:
            numpy adds a dependency for a single function. random.gauss is
            part of the standard library and provides adequate randomness
            for Monte Carlo simulation. For large-scale simulation
            (N_paths > 1M), use numpy for vectorised generation.
        """
        dt = self.T / self.N_steps
        drift = (self.r - self.q - 0.5 * self.sigma * self.sigma) * dt
        diffusion = self.sigma * math.sqrt(dt)

        path = [self.S]
        price = self.S
        for _ in range(self.N_steps):
            Z = random.gauss(0.0, 1.0)
            price = price * math.exp(drift + diffusion * Z)
            path.append(price)

        return path

    # ------------------------------------------------------------------
    # Pricing
    # ------------------------------------------------------------------

    def price(
        self,
        K: float,
        payoff: PayoffFunction,
    ) -> MCResult:
        """
        Price an option using Monte Carlo simulation.

        Args:
            K: Strike price (used only for standard payoffs — custom payoffs
               can ignore this parameter or use it in their closure).
            payoff: A function(path) -> float that computes the payoff for
                    a given price path.

        Returns:
            MCResult with the estimated premium, standard error, and path
            count.

        ALGORITHM:
        1. Simulate N_paths independent price paths.
        2. Compute the payoff for each path.
        3. Discount payoffs back to present: payoff * exp(-r * T).
        4. Average the discounted payoffs = estimated premium.
        5. Standard error = std(discounted payoffs) / sqrt(N_paths).

        WHY not parallelising:
            Python's GIL limits CPU-bound parallelism. For sequential
            simulation of N_paths, a single loop is simplest. For
            production, use a compiled language (C++ via pybind11, Phase 2b)
            or numpy vectorisation.
        """
        discounted_payoffs = []
        disc = math.exp(-self.r * self.T)

        for _ in range(self.N_paths):
            path = self._simulate_path()
            payoff_value = payoff(path)
            discounted_payoffs.append(payoff_value * disc)

        # WHY computing mean and std manually instead of using statistics
        # module: Manual computation gives us control over precision and
        # avoids an import for two function calls.
        mean = sum(discounted_payoffs) / len(discounted_payoffs)

        if len(discounted_payoffs) > 1:
            variance = sum(
                (x - mean) ** 2 for x in discounted_payoffs
            ) / (len(discounted_payoffs) - 1)
            std_err = math.sqrt(variance) / math.sqrt(len(discounted_payoffs))
        else:
            std_err = 0.0

        return MCResult(
            premium=round(mean, 4),
            standard_error=round(std_err, 4),
            paths=self.N_paths,
        )
