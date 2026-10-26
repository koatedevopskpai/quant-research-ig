# Phase 3 — market data: provider boundary, replay, and adapters

Status: **partial**. The provider layer, the two confirmed IG adapter defects and
a deterministic replay path are done and tested. Per-instrument routing in the
runner and ingestion/normalisation defects are not yet done. Phase 3 stays open.

Motivation: the IG account was suspended during verification, which made every
data-path claim depend on a broker account. That is the defect behind the defect.

## What changed

### Provider contract (`market-data/src/ig_market_data/providers/`)

`base.py` defines `MarketEvent` and `Capabilities`. Each event carries:

- `provider_ts_ns` — when the venue says it happened (IG `UTM`, Binance `T`/`E`,
  Dukascopy record offset). If a source has no usable timestamp, the event says
  so in `raw["provider_ts_source"]` rather than inventing one.
- `receive_ts_ns` — when we received it, so latency and clock skew are
  measurable.
- `book_action` — `"replace"` for top-of-book snapshots (IG L1) versus `"diff"`
  for incremental depth (Binance L2). Conflating these is what made stale quote
  levels persist.
- `raw` — the provider's untouched fields.

`Capabilities` states what each source can actually do, because they differ:

| Provider | L1 | L2 | Trades | L1 sizes | Historical | Live |
|---|---|---|---|---|---|---|
| `IGProvider` | yes | **no** | yes | **no** | yes | yes (unverified) |
| `DukascopyTicks` | yes | no | no | yes | yes | no |
| `BinanceProvider` | no | yes | yes | yes | no | yes |
| `ReplayProvider` | yes | yes | yes | yes | yes | no |

IG's API has no depth feed. Any earlier claim of "L1/L2 streaming on IG" was
wrong, and the capability flags now make that impossible to restate by accident.

### IG adapter defects fixed at source

Both were confirmed on a live demo run, not inferred:

- `stream.py` implemented `on_item_update`; the SDK calls `onItemUpdate`. The
  subscription was accepted and **zero** callbacks arrived. Renamed, with the old
  name kept as an alias.
- The bridge enumerated `dir(update.getValue)` (attributes of a bound method)
  instead of `update.getFields()`. Now reads the fields actually delivered, and
  upper-cases keys so `Bid`/`BID` and `Ltp`/`LTP` cannot diverge.
- `tick_consumer.py` mapped field names with `str.capitalize()`, so `"LTP"`
  became `"Ltp"` and the parser read nothing. A live replay of
  `{'LTP': '105', 'LTV': '12'}` produced `0.0` / `0.0`. Replaced with an
  explicit, case-insensitive alias table (`OFR`→`Offer`, `LTP`→`Last`,
  `LTV`→`LastSize`, …).

### Deterministic replay

`ReplayProvider` plays JSONL captures as `MarketEvent`s, ordered by provider
time, filtered per epic and kind, either as fast as possible (tests) or paced
(`speed=10` for demos). No account, no network, no clock dependence.

### Dukascopy (historical FX ticks)

`DukascopyTicks` builds the daily/hourly `.bi5` URL (months are zero-indexed),
decompresses the LZMA "alone" stream, unpacks big-endian records
`(ms_offset, ask, bid, ask_vol, bid_vol)` and scales by instrument digits.
Historical only — it cannot test reconnects. No vendor data is committed.

### Binance (real L2 + trades)

Public combined stream (`@depth@100ms`, `@trade`) with the documented snapshot
procedure. First implementation took the snapshot before opening the socket,
which guarantees a gap: the observed result was 1,231 trades and **0** depth
events in 20 s. A throttled re-snapshot closes the gap; after the fix, 5,453
depth events and 676 trades in 15 s, with a populated book.

## Verification

Offline (`uv run --locked pytest`): **528 passed**, including 32 new tests that
cover replay isolation/ordering, IG field mapping (both cases), Dukascopy
decoding, Binance stale/gap/removal handling, and the tick-casing regression.

Live, read-only, 2026-10-26:

| Source | Result |
|---|---|
| IG demo (before suspension) | auth OK; EUR/USD and US 500 `TRADEABLE`; repo adapter 0 callbacks; corrected adapter 63 ticks/30 s; UTM age −176…+549 ms; 41 ticks after reconnect |
| Binance `BTCUSDT` | 5,453 L2 events + 676 trades in 15 s; book populated |
| Dukascopy | first request returned 18,022 bytes; subsequent attempts timed out from this host (TCP connects, TLS handshake stalls). Decoder covered offline; downloader is environment-dependent. |

## Open items (phase 3 remains incomplete)

1. **IG transport unverified.** The account is suspended. The fixes are proven by
   the one authorised run and by tests, not by ongoing delivery.
2. **IG L1 `MARKET:` items delivered nothing** even with the corrected callback,
   across MERGE/DISTINCT, both EPICs and several field sets, with no subscription
   error. Ticks on `CHART:` worked on the same connection. Cause unknown.
3. **Runner routing.** `run_production` still keeps one book/analyser/price cache
   for all EPICs; the provider layer is per-instrument but the runner is not.
4. **Ingestion.** Opening prices are still filled from closing prices, and
   timezone-naive timestamps are still not normalised to UTC.
