"""
run_analysis.py — Black-Scholes analysis: price a US 500 index option.

Fetches the latest close from IG market data (via ig-market-data),
computes a hypothetical ATM call and put, and prints a pricing table.

Usage:
    python -m quant_methods.run_analysis
"""

import logging
from math import exp

from ig_market_data.client import IGClient

from quant_methods.pricers import BlackScholesPricer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("run_analysis")


def main() -> None:
    print("=" * 65)
    print("Black-Scholes Option Pricer — US 500 Index")
    print("=" * 65)

    client = IGClient()
    markets = client.search_markets("US 500")
    if not markets:
        print("Could not find US 500 market.  Ensure .env has valid IG credentials.")
        return

    epic = markets[0]["epic"]
    name = markets[0].get("instrumentName", "US 500")
    prices = client.fetch_prices(epic, resolution="DAY", max_points=1)
    if not prices:
        print("No prices returned.")
        return

    close = prices[0].get("closePrice", {})
    spot = float(close.get("bid", 0))

    print(f"\nUnderlying:  {name} ({epic})")
    print(f"Spot price:  {spot:.2f}")
    print()

    # Hypothetical inputs
    strike = spot                # ATM
    expiry_days = 30
    T = expiry_days / 365.0
    r = 0.052                    # ~5.2 % risk-free (SOFR)
    sigma = 0.18                 # 18 % vol (rough SPX vol)

    print(f"Strike:      {strike:.2f}  (ATM)")
    print(f"Expiry:      {expiry_days} days ({T:.4f} yr)")
    print(f"Risk-free:   {r * 100:.2f} %")
    print(f"Volatility:  {sigma * 100:.0f} %")
    print()

    pricer = BlackScholesPricer(S=spot, K=strike, T=T, r=r, sigma=sigma)

    for opt_type in ("call", "put"):
        result = pricer.price() if opt_type == "call" else pricer.price_put()
        # Actually use the full compute for Greeks
        result = pricer.compute(opt_type)  # type: ignore
        print(f"--- {opt_type.upper()} ---")
        print(f"  Premium: {result.premium:>10.4f}")
        print(f"  Delta:   {result.delta:>10.4f}")
        print(f"  Gamma:   {result.gamma:>10.4f}")
        print(f"  Vega:    {result.vega:>10.4f}")
        print(f"  Theta:   {result.theta:>10.4f}")
        print(f"  Rho:     {result.rho:>10.4f}")
        print()


def test_price_consistency() -> None:
    """
    Verify BSM put-call mathematically.
    European call - put = S - K * exp(-r * T) (no dividend).
    """
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
    p = BlackScholesPricer(S=S, K=K, T=T, r=r, sigma=sigma)

    call = p.compute("call")
    put = p.compute("put")
    put_call_parity = call.premium - put.premium
    expected = S - K * exp(-r * T)
    assert abs(put_call_parity - expected) < 0.001, (
        f"Put-call parity violated: {put_call_parity} != {expected}"
    )
    print("Put-call parity: OK")


if __name__ == "__main__":
    test_price_consistency()
    print()
    main()