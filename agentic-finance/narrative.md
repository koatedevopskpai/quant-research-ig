# agentic-finance — Agentic Trading Systems

## Objective
Build autonomous trading agents that combine LLM reasoning (GPT-4o-mini)
with quantitative signals. Agents analyse market conditions, generate
trade ideas, manage risk, and explain their decisions — all auditable.

This is Phase 5 of the quant-research-ig platform — the capstone layer
that orchestrates all prior phases (market data → quant methods → HFT
engine → ML models) into an autonomous decision-making framework.

## Architecture

The agent system follows Beckman's Research→Know→Do→Get cycle:

```
Research Agent  →  Portfolio Agent  →  Execution Agent  →  Market
      ↓                                 ↑                        ↓
   Risk Manager ─────────────────── kills/overrides        Audit Trail
```

All communication flows through a publish-subscribe message bus, and
every decision is recorded in the audit trail.

## Completed Modules

### Decision System (`decision.py`)
- **Decision** dataclass — agent, type, status, context, reasoning, LLM explanation
- **DecisionType** enum — RESEARCH, SIGNAL, ORDER, RISK_CHECK, ALLOCATION, KILL_SWITCH, AUDIT
- **DecisionStatus** enum — PENDING, APPROVED, REJECTED, EXECUTED, FAILED
- Factory method `Decision.now()` for instantiation with UUID

### Message Bus (`message_bus.py`)
- **MessageBus** — pub-sub event bus with topic-based routing
- **AgentMessage** — dataclass with message_id, topic, sender, payload, timestamp
- Wildcard subscriber `*` catches all messages
- Error isolation — one bad callback does not crash other subscribers
- Message types: `research.completed`, `signal.generated`, `order.created`,
  `risk.alert`, `risk.kill`, `allocation.updated`

### Audit Trail (`audit_trail.py`)
- **AuditTrail** — append-only log of all decisions
- **AuditEntry** — decision + market context + signals used + tags
- Query by agent, decision_type, status, or time range
- JSON export to string or file
- Full traceability from research request → execution

### Research Agent (`research_agent.py`)
- Accepts research requests (epic, model type, parameters)
- Routes to registered ML models (regime, forecast, LSTM, sentiment)
- Generates structured summary of model outputs
- Optional LLM report via OpenAI (falls back gracefully without API key)
- Publishes `research.completed` messages

### Execution Agent (`execution_agent.py`)
- **Signal → Order pipeline**: validates signal, checks toxicity, sizes position,
  estimates market impact, creates order with stop-loss/take-profit
- Toxicity veto — rejects signals when order flow is toxic (Phase 3b)
- Kill-switch — halts execution on demand
- Position tracking across epics
- Configurable risk constraints (max position %, stop-loss %)
- Publishes `order.created` messages

### Risk Manager (`risk_manager.py`)
- Three VaR methods: historical (empirical percentile), parametric (normal),
  Monte Carlo (simulated paths)
- Drawdown tracking from peak capital
- Daily loss limit enforcement
- Kill-switch — triggers when drawdown, daily loss, or VaR limits breached
- Publishes `risk.alert` and `risk.kill` messages

### Portfolio Agent (`portfolio_agent.py`)
- Three allocation methods: equal-weight, risk-parity (inverse-vol),
  mean-variance (tangency portfolio)
- Registers strategies with expected return/vol
- Produces Allocation objects with weight + capital + reasoning
- Publishes `allocation.updated` messages

## ADRs (Architecture Decision Records)

### ADR-1: Message Bus over Direct Calls
**Context**: Agents need to communicate without tight coupling.
**Decision**: Publish-subscribe MessageBus with topic routing.
**Consequence**: Loose coupling, easy to add new agents, but harder to
trace message flows at runtime (mitigated by audit trail).

### ADR-2: Optional LLM Integration
**Context**: Research Agent should produce narrative reports, but tests
must not require API keys.
**Decision**: LLM calls gated behind `llm_api_key is not None`. Without
key, agent returns structured summary.
**Consequence**: Fully testable offline, LLM as optional enhancement.

### ADR-3: Three-Pillar Risk (VaR, Drawdown, Daily Loss)
**Context**: Single risk metric is insufficient for a trading system.
**Decision**: Monitor parametric VaR, peak-to-trough drawdown, and daily
P&L limits independently.
**Consequence**: More comprehensive risk coverage; kill-switch only fires
when any pillar is breached.

### ADR-4: Plugin Model Registry
**Context**: Research Agent needs to run different ML models without
hard-coding them.
**Decision**: `register_model(model_type, model)` lets any model be
injected, as long as it follows the expected interface (fit/predict).
**Consequence**: Easy to swap models or add new ones; no changes to
agent code needed.

### ADR-5: Toxicity Veto in Execution
**Context**: Toxic order flow (Phase 3b Hawkes model) should block
execution.
**Decision**: Execution Agent checks toxicity detector before sizing;
if toxic, signal is rejected and audit trail records the veto.
**Consequence**: Simple boolean gate; no partial fills or overrides.

## Key Files
| File | Purpose |
|------|---------|
| src/agentic_finance/decision.py | Decision, DecisionType, DecisionStatus |
| src/agentic_finance/message_bus.py | MessageBus, AgentMessage |
| src/agentic_finance/audit_trail.py | AuditTrail, AuditEntry |
| src/agentic_finance/research_agent.py | ResearchAgent, ResearchRequest, ResearchResult |
| src/agentic_finance/execution_agent.py | ExecutionAgent, Order, Signal |
| src/agentic_finance/risk_manager.py | RiskManager, RiskMetrics |
| src/agentic_finance/portfolio_agent.py | PortfolioAgent, Allocation, PortfolioState |
| tests/test_message_bus.py | unit checks |
| tests/test_decision.py | unit checks |
| tests/test_audit_trail.py | unit checks |
| tests/test_research_agent.py | unit checks |
| tests/test_execution_agent.py | unit checks |
| tests/test_risk_manager.py | unit checks |
| tests/test_portfolio_agent.py | unit checks |

## Status
- [x] Project skeleton with pyproject.toml, narrative.md
- [x] Core infrastructure: Decision, MessageBus, AuditTrail
- [x] Research Agent with model registry and LLM integration
- [x] Execution Agent with toxicity veto and position sizing
- [x] Risk Manager with VaR, drawdown, daily loss limits
- [x] Portfolio Agent with equal/risk-parity/mean-variance allocation
- [x] Unit checks are listed here; no current pass result is established by this phase note
