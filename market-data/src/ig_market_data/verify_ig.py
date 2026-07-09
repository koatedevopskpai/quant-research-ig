"""
verify_ig.py — End-to-end IG API credential verification.

Reads credentials from .env (via config.py), then runs:
  1. Authentication (POST /session, v3)
  2. Category listing (GET /categories, v1)
  3. Price history fetch (GET /prices/{epic}, v3)

Usage:
    python -m ig_market_data.verify_ig
"""

import logging

from ig_market_data.client import IGClient
from ig_market_data.config import ig_settings

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

print("=" * 60)
print("Step 1: Authentication")
print("=" * 60)

client = IGClient()
print(f"  Account ID:  {client.account_id}")
print(f"  Base URL:    {ig_settings.base_url}")
print(f"  Token:       {client.token[:40]}...")
print()

print("=" * 60)
print("Step 2: List market categories")
print("=" * 60)

categories = client.list_categories()
print(f"  Found {len(categories)} categories:")
for c in categories:
    code = c.get("code", "?")
    nt = "(non-tradeable)" if c.get("nonTradeable") else "(tradeable)"
    print(f"    - {code:20s} {nt}")
print()

print("=" * 60)
print("Step 3: Fetch price history (US 500, daily, last 5)")
print("=" * 60)

markets = client.search_markets("US 500")
if not markets:
    print("  No markets found for 'US 500'")
else:
    epic = markets[0]["epic"]
    name = markets[0].get("instrumentName", "?")
    print(f"  Market:  {name} ({epic})")

    prices = client.fetch_prices(epic, resolution="DAY", max_points=5)
    print(f"  Points:  {len(prices)}")
    for p in prices:
        dt = p.get("snapshotTime", "?")
        close = p.get("closePrice", {})
        bid = close.get("bid", "?")
        ask = close.get("ask", "?")
        print(f"    {dt:22s}  bid={bid:>10}  ask={ask:>10}")

print()
print("Done. IG credentials verified successfully.")
print()
print(f"Lightstreamer endpoint: {ig_settings.lightstreamer_url}")