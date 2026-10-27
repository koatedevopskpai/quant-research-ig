# LangGraph orchestration

Status: **implemented** (2026-10-27). Research → Risk → Portfolio → Execution is
now a `StateGraph`, and the halt path is a conditional edge rather than an `if`
duplicated in every caller.

## What the graph owns

```text
START → research → risk ─┬─(allowed)→ portfolio → execution → END
                         └─(blocked)──────────────────────────→ END
```

| Node | Calls | Responsibility |
|---|---|---|
| `research` | `ResearchAgent.research_signal(publish=False)` | fit/predict through the model adapters; produce direction, confidence, regime |
| `risk` | `RiskManager.check_risk()` + halt flag | block on halt, breach, flat view, or confidence below the floor |
| `portfolio` | — | record the sizing intent that execution will act on |
| `execution` | `ExecutionAgent.process_signal` | size against the signed target and fill |

The agents still own behaviour, sizing and auditing. The graph owns ordering and
branching; it is not a second implementation of them.

## Determinism

There is no LLM inside the graph. The research agent's optional OpenAI report is
generated outside the decision path, and the graph is invoked synchronously with
explicit state. Two runs with the same prices and models produce the same
decision (asserted in `test_langgraph.py`).

`research_signal` gained `publish=False` for graph use: the graph decides whether
a result becomes an order, so it must not also emit `research.completed` and let
the runner's bus handler create a second one. External callers keep the message.

## Where it runs

- **Runner**: `_run_research` compiles the graph once and invokes it per EPIC with
  `MIN_SIGNAL_CONFIDENCE`.
- **Replay**: the deterministic replay invokes the same graph, so the integration
  proof covers the orchestration layer rather than calling agents directly.

## What this does not claim

- **No LangGraph checkpointer.** Durable state lives in the SQLite ledger, audit
  trail and risk row (`docs/persistence.md`). The graph is stateless between
  invocations.
- **No human-in-the-loop interrupts.** Every decision is automatic and audited.
- **Agents remain plain classes.** The graph wraps them; it does not replace the
  event bus for asynchronous notifications.
- **Synchronous execution.** One graph invocation per EPIC per research interval;
  no parallel node execution (there is nothing independent to parallelise).

## Verification

Six tests in `agentic-finance/tests/test_langgraph.py`:

- a directional view creates a filled order through the graph;
- a halt stops before execution and creates no order;
- a flat regime-only view does not trade;
- the confidence floor is enforced at the risk edge;
- the same inputs produce the same side, size and position;
- a risk breach blocks even a good signal.

The deterministic replay tests also now exercise the graph end to end.
