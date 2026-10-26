# Phase 1 — Reproducible workspace baseline

The supported workspace baseline is CPython 3.12.15, managed by uv 0.13.5.
The root is a virtual workspace, not a sixth distributable package. Its five
members are installed together as editable distributions; sibling package names
resolve locally rather than from PyPI. `uv.lock` pins transitive versions and
artifact hashes. CPU PyTorch is selected explicitly to keep CI independent of CUDA.

From the repository root:

```sh
python -m pip install uv==0.13.5
uv sync --locked
uv run --locked python scripts/check_baseline.py
uv run --locked pytest --collect-only -q
uv run --locked pytest --junitxml=test-results/junit.xml
```

These commands work in PowerShell and POSIX shells. No environment activation or
manual PYTHONPATH is required. uv installs the selected Python if unavailable.
Python 3.14 is outside this workspace baseline. Member packages retain their
existing metadata; that is not a claim of CI coverage for other Python versions.

## Credential-free testing

Root pytest configuration uses importlib collection to avoid collisions between
the five directories named `tests`. The root conftest disables `.env` discovery
before test imports and blocks external httpx HTTP transports during each test.
FastAPI's custom in-process transport continues to work. This guard covers the
current HTTP provider paths, not arbitrary sockets or third-party networking APIs.
Pytest writes temporary files under the ignored `test-results/tmp` directory;
pytest clears that temporary directory at the start of a run. Do not place
hand-maintained files there or share that directory between concurrent runs.
Lightstreamer connection tests replace the streamer module with an offline stub;
LLM failure tests replace the client constructor, avoiding retries and SDK network
calls. Never supply production secrets to the baseline job.

Importing configuration, ingestion, or runner modules no longer requires IG keys.
REST client construction and streaming connection validate credentials before
opening provider sessions. Explicit integrations still use the existing `.env`
discovery by default. Set `QUANT_LOAD_DOTENV=0` to disable it outside pytest.

The Quote table arguments tuple is repaired. SQLite test IDs use an Integer
variant for autoincrement while PostgreSQL retains BigInteger IDs. The runner
imports `Any`, which Python 3.12 needs when evaluating its annotations.

## CI

`.github/workflows/ci.yml` installs the lock, checks imports, collects all modules,
and runs the suite on Windows and Ubuntu with Python 3.12. Jobs have a time limit,
read-only repository permissions, and upload JUnit results even after failure.
The optional C++ accelerator remains unbuilt and its existing tests may skip.
This workflow does not certify C++ behavior or PostgreSQL-specific behavior.

To update dependencies deliberately:

```sh
uv lock --upgrade
uv sync --locked
uv run --locked pytest
```

Review lock changes and both CI platforms before accepting the new baseline.
Do not regenerate a lock automatically during CI.

## Acceptance and scope

- Fresh locked install of all five local packages.
- Credential-free import checks and complete test collection.
- Existing tests run without contacting IG or an LLM provider.
- Windows and Linux workflow jobs configured; remote outcomes require a push.

This phase does not repair PSR/DSR, CPCV, trading accounting, signal contracts,
instrument routing, kill switches, or API authorization. Those findings remain
in their assigned later phases. A green baseline is infrastructure evidence,
not proof that those algorithms or integrations are correct.

## Local verification — 26 October 2026

Base revision: `dacb89d08f49d5e8b08ac6eeb801514e9498c8f3`.
Local branch: `phase-1-reproducible-baseline`.

- Fresh Windows CPython 3.12.15 environment installed successfully: 108 locked
  packages, 82 installed for this platform, including all five editable members.
- `uv sync --locked --offline` subsequently succeeded using the populated cache.
- The baseline script imported all five distributions successfully.
- Final full suite: **443 passed, 24 skipped, 139 warnings in 54.33 seconds**.
  The 24 skips are the unbuilt optional C++ accelerator tests. Warnings are
  existing pandas/hmmlearn deprecations; no warnings were suppressed.
- Ruff passed for the new baseline scripts/configuration tests and modified
  configuration module. This is not a whole-repository lint pass.
- `git diff --check` passed. GitHub Actions jobs have not been run remotely.

No IG credentials, PostgreSQL server, LLM service, or live trading execution
was used to establish these results. The tested suite contains 467 tests across
all five modules; passing it does not resolve the later-phase review findings.
