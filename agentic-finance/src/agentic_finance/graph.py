"""
graph.py — LangGraph orchestration for the Research → Risk → Portfolio → Execution path.

WHY A GRAPH
    The runner previously called the agents in sequence by hand, with the halt
    check duplicated in the callback. A StateGraph makes the path explicit and
    typed: the halt is a conditional edge, not an `if` scattered across callers.
    `docs/langgraph.md` records what this does and does not replace.

CONTRACT
    Nodes wrap the existing agents rather than replacing them. The graph owns
    ordering, state and branching; the agents own behaviour, sizing and audit.

    research  -> ResearchAgent.research_signal (no bus publish; the graph
                 decides whether the result becomes an order)
    risk      -> RiskManager.check_risk + halt/direction/confidence gates
    portfolio -> records the intent that execution will size against
    execution -> ExecutionAgent.process_signal

    Every node's work is already audited by the agent it calls, so the graph adds
    ordering without creating a second, divergent audit stream.
"""

from __future__ import annotations

from typing import Any, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from agentic_finance.execution_agent import ExecutionAgent, Signal
from agentic_finance.research_agent import ResearchAgent
from agentic_finance.risk_manager import RiskManager


class TradingState(TypedDict, total=False):
    epic: str
    prices: Any          # pandas Series
    price: float
    min_confidence: float
    research: dict[str, Any]
    allowed: bool
    reason: str
    intent: dict[str, Any]
    order: Optional[dict[str, Any]]


def build_trading_graph(research: ResearchAgent, risk: RiskManager,
                        execution: ExecutionAgent):
    """Compile the decision graph around the existing agents."""

    def research_node(state: TradingState) -> dict[str, Any]:
        result = research.research_signal(
            state["epic"], state["prices"], publish=False
        )
        return {"research": dict(result.model_output)}

    def risk_node(state: TradingState) -> dict[str, Any]:
        output = state.get("research", {})
        if "error" in output:
            return {"allowed": False, "reason": output["error"]}

        metrics = risk.check_risk()
        if risk.is_halted:
            return {"allowed": False, "reason": "risk halted"}
        if metrics.is_breached:
            return {"allowed": False, "reason": "; ".join(metrics.breaches)}

        direction = float(output.get("direction", 0.0))
        confidence = float(output.get("confidence", 0.0))
        if direction == 0.0:
            return {"allowed": False, "reason": "no directional view"}
        if confidence < float(state.get("min_confidence", 0.0)):
            return {
                "allowed": False,
                "reason": f"confidence {confidence:.3f} below floor",
            }
        return {"allowed": True, "reason": "approved"}

    def portfolio_node(state: TradingState) -> dict[str, Any]:
        output = state["research"]
        return {
            "intent": {
                "epic": state["epic"],
                "direction": float(output["direction"]),
                "confidence": float(output["confidence"]),
                "regime": output.get("regime"),
            }
        }

    def execution_node(state: TradingState) -> dict[str, Any]:
        intent = state["intent"]
        order = execution.process_signal(
            Signal(
                epic=state["epic"],
                direction=intent["direction"],
                confidence=intent["confidence"],
                source=f"langgraph:{intent.get('regime') or 'signal'}",
            ),
            float(state["price"]),
        )
        if order is None:
            return {"order": None}
        return {
            "order": {
                "order_id": order.order_id,
                "epic": order.epic,
                "side": order.side,
                "size": order.size,
                "status": order.status,
            }
        }

    def route_after_risk(state: TradingState) -> str:
        return "execute" if state.get("allowed") else "stop"

    graph = StateGraph(TradingState)
    graph.add_node("research", research_node)
    graph.add_node("risk", risk_node)
    graph.add_node("portfolio", portfolio_node)
    graph.add_node("execution", execution_node)

    graph.add_edge(START, "research")
    graph.add_edge("research", "risk")
    graph.add_conditional_edges(
        "risk",
        route_after_risk,
        {"execute": "portfolio", "stop": END},
    )
    graph.add_edge("portfolio", "execution")
    graph.add_edge("execution", END)
    return graph.compile()


def run_signal(graph, epic: str, prices: Any, price: float,
               min_confidence: float = 0.0) -> dict[str, Any]:
    """Invoke the graph for one EPIC and return the final state."""
    return graph.invoke({
        "epic": epic,
        "prices": prices,
        "price": float(price),
        "min_confidence": float(min_confidence),
    })
