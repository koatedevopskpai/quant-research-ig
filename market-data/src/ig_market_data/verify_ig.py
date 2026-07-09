"""
verify_ig.py — End-to-end IG API credential verification.

WHY THIS EXISTS:
    Before any quant research can begin, we must confirm that:
    1. IG API credentials are valid (authentication works).
    2. The API is reachable (network, firewall, DNS all correct).
    3. Market data is accessible (can discover and fetch prices).

    This script runs three sequential checks and reports results. Think of
    it as a "pre-flight checklist" for the quant research pipeline.

WHAT IT TESTS:
    Step 1: Authentication — POST /session (v3), confirms OAuth 2.0
            Bearer token is generated and the account ID is received.
    Step 2: Category discovery — GET /categories (v1), lists all
            tradeable market categories (indices, FX, commodities, etc.).
    Step 3: Price history — GET /prices/{epic} (v3), fetches 5 daily
            OHLCV candles for US 500 (the S&P 500 CFD).

EXPECTED OUTPUT (when credentials are valid):
    ============================================================
    Step 1: Authentication
    ============================================================
      Account ID:  XXXXXXXX
      Base URL:    https://demo-api.ig.com/gateway/deal
      Token:       eyJhbGciOiJSUzI1NiIsInR5...

    ============================================================
    Step 2: List market categories
    ============================================================
      Found 8 categories:
        - COMMODITIES           (tradeable)
        - CRYPTOCURRENCY        (tradeable)
        - FX                    (tradeable)
        - INDICES               (tradeable)
        - ...

    ============================================================
    Step 3: Fetch price history (US 500, daily, last 5)
    ============================================================
      Market:  US 500 (IX.D.SPTRD.IFD.IP)
      Points:  5
        2025-01-02  bid=  1234.5  ask=  1234.8
        ...

    Done. IG credentials verified successfully.

WHAT A FAILURE MEANS:
    - Network error: Check internet / firewall (IG requires HTTPS).
    - Auth error (401/403): Check .env credentials are correct.
    - Not found (404): The default epic may need updating.
    - Rate limit (429): Wait 60 seconds and retry.

METHODOLOGY NOTE (Quant Beckman):
    This verification step is consistent with Beckman's emphasis on data
    quality assurance. Before any hypothesis can be tested, the data
    pipeline must be validated. This script is the first gate in the
    "Research" stage of his Research → Know → Do → Get cycle.
"""

import logging

from ig_market_data.client import IGClient
from ig_market_data.config import ig_settings

# WHY logging.basicConfig with simple format:
#   Verification output should be human-readable, not structured logs.
#   The format "%(message)s" suppresses timestamp and level prefix,
#   making the output cleaner for terminal reading.
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

print("=" * 60)
print("Step 1: Authentication")
print("=" * 60)

# WHY creating IGClient here:
#   The IGClient constructor calls _authenticate() automatically. If
#   authentication fails, an exception is raised immediately with a
#   descriptive message. This is intentional — fail fast, fail clearly.
client = IGClient()
print(f"  Account ID:  {client.account_id}")

# WHY printing base_url:
#   Users should verify they are connecting to DEMO vs. LIVE. Printing
#   the URL makes this explicit and reduces the risk of accidental live
#   trading during development.
print(f"  Base URL:    {ig_settings.base_url}")

# WHY printing partial token:
#   Full token is a security concern. Showing the first 40 characters is
#   enough to confirm a token was received without exposing the complete
#   credential. Most tokens are 500-1000 characters.
print(f"  Token:       {client.token[:40]}...")
print()

print("=" * 60)
print("Step 2: List market categories")
print("=" * 60)

categories = client.list_categories()
print(f"  Found {len(categories)} categories:")
for c in categories:
    code = c.get("code", "?")
    # WHY showing tradeable vs. non-tradeable:
    #   Some categories (e.g., test categories) are non-tradeable. Users
    #   need to know which categories contain real instruments for research.
    nt = "(non-tradeable)" if c.get("nonTradeable") else "(tradeable)"
    print(f"    - {code:20s} {nt}")
print()

print("=" * 60)
print("Step 3: Fetch price history (US 500, daily, last 5)")
print("=" * 60)

# WHY searching for "US 500":
#   "US 500" is IG's identifier for the S&P 500 CFD. It is the most
#   commonly used index for quant research and is available on all IG
#   accounts (DEMO and LIVE).
markets = client.search_markets("US 500")
if not markets:
    print("  No markets found for 'US 500'")
else:
    epic = markets[0]["epic"]
    name = markets[0].get("instrumentName", "?")
    print(f"  Market:  {name} ({epic})")

    # WHY max_points=5:
    #   Fetching 5 data points is sufficient to confirm the price endpoint
    #   works without generating unnecessary API traffic or data volume.
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
