"""
api/server.py — FastAPI REST API for the quant-research-ig platform.

Exposes all five agents (Research, Execution, Risk, Portfolio, Audit)
as REST endpoints, plus a WebSocket market data stream.

USAGE:
    # From the project root:
    python -m agentic_finance.api.server

    # Or via uvicorn directly:
    uvicorn agentic_finance.api.server:app --reload

ENDPOINTS:
    POST /research/{epic}       — Run a research model
    POST /signal                — Accept/validate an execution signal
    GET  /risk                  — Current risk metrics
    POST /risk/check            — Force risk check + audit
    POST /allocate/{method}     — Rebalance portfolio
    GET  /audit                 — Query audit trail
    GET  /audit/{decision_id}   — Get single audit entry
    GET  /audit/export          — Export all as JSON
    POST /bus/publish           — Manually publish to a bus topic
    WS   /stream/{epic}         — Live market data stream
"""

from __future__ import annotations

import json
import logging
import sys
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agentic_finance.audit_trail import AuditTrail
from agentic_finance.decision import Decision, DecisionType, DecisionStatus
from agentic_finance.execution_agent import ExecutionAgent, Signal
from agentic_finance.message_bus import MessageBus
from agentic_finance.portfolio_agent import PortfolioAgent
from agentic_finance.research_agent import ResearchAgent, ResearchRequest
from agentic_finance.risk_manager import RiskManager

logger = logging.getLogger(__name__)


# ── Pydantic request/response models ────────────────────────────────

class ResearchResponse(BaseModel):
    epic: str
    model_type: str
    model_output: dict[str, Any]
    summary: str
    llm_report: str
    request_id: str


class SignalRequest(BaseModel):
    epic: str
    direction: float = Field(..., ge=-1.0, le=1.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    source: str = "api"
    current_price: float = Field(default=100.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class OrderResponse(BaseModel):
    epic: str
    side: str
    size: float
    order_type: str
    limit_price: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    order_id: str
    status: str
    reasoning: str


class RiskMetricsResponse(BaseModel):
    var_historical: float
    var_parametric: float
    var_monte_carlo: float
    current_drawdown: float
    daily_pnl: float
    total_exposure: float
    n_positions: int
    is_breached: bool
    breaches: list[str]
    equity: float = 0.0
    exposure_pct: float = 0.0
    is_halted: bool = False


class AllocateRequest(BaseModel):
    kwargs: dict[str, Any] = Field(default_factory=dict)


class PortfolioStateResponse(BaseModel):
    total_capital: float
    allocations: list[dict[str, Any]]
    cash: float
    rebalance_threshold: float


class AuditEntryResponse(BaseModel):
    decision_id: str
    timestamp: str
    agent: str
    decision_type: str
    status: str
    description: str
    reasoning: str
    market_context: dict[str, Any]
    tags: list[str]


class BusPublishRequest(BaseModel):
    topic: str
    sender: str
    payload: dict[str, Any] = Field(default_factory=dict)


class BusPublishResponse(BaseModel):
    message_id: str
    topic: str
    sender: str


# ── Shared application state ────────────────────────────────────────

class AppState:
    """Singleton container for agent instances shared across requests."""

    def __init__(self) -> None:
        self.bus = MessageBus()
        self.audit = AuditTrail()
        self.research = ResearchAgent(self.bus, self.audit)
        self.execution = ExecutionAgent(self.bus, self.audit)
        self.risk = RiskManager(self.bus, self.audit)
        self.portfolio = PortfolioAgent(self.bus, self.audit)

        # The API path must enforce the same controls as the runner: risk owns
        # P&L/exposure from the ledger, and a halt stops execution here too.
        self.risk.set_ledger(self.execution.ledger)
        self.bus.subscribe("risk.halted", self._on_risk_halted)
        self.bus.subscribe("risk.resumed", self._on_risk_resumed)

    def _on_risk_halted(self, msg) -> None:
        reason = msg.payload.get("reason", "risk halt") if msg.payload else "risk halt"
        self.execution.halt(reason)

    def _on_risk_resumed(self, msg) -> None:
        self.execution.resume()

    def register_default_research_models(self) -> None:
        """
        Register DEMO adapters so /research returns a well-formed response with
        no credentials.

        These are not trained models: direction and confidence are fixed and
        every output carries diagnostics["demo"] = True, so a client cannot
        mistake the response for a forecast. The production runner registers the
        real ForecastAdapter/RegimeAdapter instead of calling this.
        """
        from agentic_finance.model_contract import ModelOutput

        class _DemoRegime:
            name = "regime"

            def fit(self, prices):
                return self

            def predict(self, prices):
                return ModelOutput(
                    model="regime", direction=0.0, confidence=0.0,
                    regime=1, regime_label="Medium Vol (demo)",
                    diagnostics={"demo": True},
                )

        def _demo_forecast(name: str):
            class _DemoForecast:
                def fit(self, prices):
                    return self

                def predict(self, prices):
                    return ModelOutput(
                        model=name, direction=1.0, confidence=0.25,
                        expected_return=0.001,
                        diagnostics={"demo": True, "n_forecasts": 0},
                    )

            _DemoForecast.name = name
            return _DemoForecast()

        self.research.register_model("regime", _DemoRegime())
        self.research.register_model("forecast", _demo_forecast("forecast"))
        self.research.register_model("lstm", _demo_forecast("lstm"))
        self.research.register_model("sentiment", _demo_forecast("sentiment"))


_state: AppState | None = None


def get_state() -> AppState:
    global _state
    if _state is None:
        _state = AppState()
    return _state


def _convert_decisions(entries: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "decision_id": e.decision.decision_id,
            "timestamp": e.decision.timestamp.isoformat(),
            "agent": e.decision.agent,
            "decision_type": e.decision.decision_type.value,
            "status": e.decision.status.value,
            "description": e.decision.description,
            "reasoning": e.decision.reasoning,
            "market_context": e.market_context,
            "tags": e.tags,
        }
        for e in entries
    ]


# ── Lifespan ────────────────────────────────────────────────────────

def _ensure_models() -> None:
    """Register default research models if not already done."""
    state = get_state()
    if not state.research._models:
        state.register_default_research_models()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    _ensure_models()
    logger.info("API server started — agents initialised")
    yield
    logger.info("API server shutting down")


# ── FastAPI app ─────────────────────────────────────────────────────

app = FastAPI(
    title="quant-research-ig API",
    description="REST API for the agentic quant trading platform",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Research endpoints ──────────────────────────────────────────────

class ResearchRequestBody(BaseModel):
    model_type: str = Field(default="regime", pattern=r"^(regime|forecast|lstm|sentiment)$")
    prices: list[float] = Field(default_factory=list)
    request_id: str | None = None


@app.post("/research/{epic}")
async def run_research(epic: str, body: ResearchRequestBody) -> ResearchResponse:
    state = get_state()
    _ensure_models()
    if not body.prices:
        import numpy as np
        prices = np.linspace(100, 105, 30)
    else:
        import numpy as np
        prices = np.array(body.prices)
    request = ResearchRequest(
        epic=epic,
        model_type=body.model_type,
        request_id=body.request_id or "",
    )
    result = state.research.research(request, prices)
    return ResearchResponse(
        epic=result.epic,
        model_type=result.model_type,
        model_output=result.model_output,
        summary=result.summary,
        llm_report=result.llm_report,
        request_id=result.request_id,
    )


# ── Execution endpoints ─────────────────────────────────────────────

@app.post("/signal")
async def post_signal(body: SignalRequest) -> OrderResponse | None:
    state = get_state()
    # Enforcement point 2 (API): a halt must stop new risk here, not only in the
    # runner. ExecutionAgent also refuses, but 409 makes the reason visible.
    if (state.risk.is_halted or not state.execution.is_active
            or not state.risk.can_trade()):
        raise HTTPException(status_code=409, detail="Trading is halted")
    signal = Signal(
        epic=body.epic,
        direction=body.direction,
        confidence=body.confidence,
        source=body.source,
        metadata=body.metadata,
    )
    order = state.execution.process_signal(signal, body.current_price)
    if order is None:
        raise HTTPException(status_code=400, detail="Signal rejected by execution agent")
    return OrderResponse(
        epic=order.epic,
        side=order.side,
        size=order.size,
        order_type=order.order_type,
        limit_price=order.limit_price,
        stop_loss=order.stop_loss,
        take_profit=order.take_profit,
        order_id=order.order_id,
        status=order.status,
        reasoning=order.reasoning,
    )


@app.get("/positions")
async def get_positions() -> dict[str, float]:
    return dict(get_state().execution.positions)


# ── Risk endpoints ──────────────────────────────────────────────────

@app.get("/risk")
async def get_risk() -> RiskMetricsResponse:
    metrics = get_state().risk.compute_metrics()
    return RiskMetricsResponse(
        var_historical=metrics.var_historical,
        var_parametric=metrics.var_parametric,
        var_monte_carlo=metrics.var_monte_carlo,
        current_drawdown=metrics.current_drawdown,
        daily_pnl=metrics.daily_pnl,
        total_exposure=metrics.total_exposure,
        n_positions=metrics.n_positions,
        is_breached=metrics.is_breached,
        breaches=metrics.breaches,
        equity=metrics.equity,
        exposure_pct=metrics.exposure_pct,
        is_halted=metrics.is_halted,
    )


@app.post("/risk/check")
async def check_risk() -> RiskMetricsResponse:
    metrics = get_state().risk.check_risk()
    return RiskMetricsResponse(
        var_historical=metrics.var_historical,
        var_parametric=metrics.var_parametric,
        var_monte_carlo=metrics.var_monte_carlo,
        current_drawdown=metrics.current_drawdown,
        daily_pnl=metrics.daily_pnl,
        total_exposure=metrics.total_exposure,
        n_positions=metrics.n_positions,
        is_breached=metrics.is_breached,
        breaches=metrics.breaches,
        equity=metrics.equity,
        exposure_pct=metrics.exposure_pct,
        is_halted=metrics.is_halted,
    )


@app.post("/risk/kill")
async def kill_switch() -> dict[str, bool]:
    """Unconditional emergency halt — not conditional on an existing breach."""
    state = get_state()
    halted = state.risk.halt("manual emergency halt via API")
    return {"halted": halted or state.risk.is_halted}


@app.post("/risk/resume")
async def resume_trading() -> dict[str, bool]:
    state = get_state()
    resumed = state.risk.resume()
    return {"resumed": resumed or not state.risk.is_halted}


# ── Portfolio endpoints ─────────────────────────────────────────────

@app.post("/allocate/{method}")
async def allocate(method: str, body: AllocateRequest) -> PortfolioStateResponse:
    if method not in ("equal", "risk_parity", "mean_variance"):
        raise HTTPException(status_code=400, detail=f"Unknown method: {method}")
    state = get_state()
    result = state.portfolio.allocate(method=method, **body.kwargs)
    return PortfolioStateResponse(
        total_capital=result.total_capital,
        allocations=[
            {"strategy_id": a.strategy_id, "weight": a.weight,
             "capital": a.capital, "reason": a.reason}
            for a in result.allocations
        ],
        cash=result.cash,
        rebalance_threshold=result.rebalance_threshold,
    )


@app.get("/portfolio")
async def get_portfolio() -> PortfolioStateResponse:
    agent = get_state().portfolio
    if agent.current_weights:
        allocations = []
        total_cap = agent.total_capital
        for sid, w in agent.current_weights.items():
            allocations.append({
                "strategy_id": sid,
                "weight": float(w),
                "capital": float(total_cap * w),
                "reason": "current allocation",
            })
        cash = total_cap * max(0.0, 1.0 - sum(agent.current_weights.values()))
    else:
        allocations = []
        cash = agent.total_capital
    return PortfolioStateResponse(
        total_capital=agent.total_capital,
        allocations=allocations,
        cash=cash,
        rebalance_threshold=agent.rebalance_threshold,
    )


@app.get("/portfolio/strategies")
async def get_strategies() -> dict[str, Any]:
    return dict(get_state().portfolio.strategies)


# ── Audit endpoints ─────────────────────────────────────────────────

@app.get("/audit")
async def query_audit(
    agent: str | None = None,
    decision_type: str | None = None,
    status: str | None = None,
    limit: int = 100,
) -> list[AuditEntryResponse]:
    entries = get_state().audit.query(
        agent=agent, decision_type=decision_type,
        status=status, limit=limit,
    )
    return _convert_decisions(entries)


@app.get("/audit/export")
async def export_audit():
    from fastapi.responses import JSONResponse
    import json
    data = json.loads(get_state().audit.to_json())
    return JSONResponse(content=data)


@app.get("/audit/{decision_id}")
async def get_audit_entry(decision_id: str) -> AuditEntryResponse:
    entry = get_state().audit.get(decision_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Decision not found")
    return _convert_decisions([entry])[0]


# ── Message bus endpoints ───────────────────────────────────────────

@app.post("/bus/publish")
async def bus_publish(body: BusPublishRequest) -> BusPublishResponse:
    msg = get_state().bus.publish_topic(
        topic=body.topic,
        sender=body.sender,
        payload=body.payload,
    )
    return BusPublishResponse(
        message_id=msg.message_id,
        topic=msg.topic,
        sender=msg.sender,
    )


# ── WebSocket market stream ─────────────────────────────────────────

@app.websocket("/stream/{epic}")
async def market_stream(websocket: WebSocket, epic: str) -> None:
    await websocket.accept()
    logger.info("WS connected: %s", epic)

    async def send_latest():
        import time
        while True:
            await websocket.send_json({
                "epic": epic,
                "type": "ping",
                "timestamp": time.time(),
            })
            await asyncio.sleep(5)

    import asyncio
    task = asyncio.create_task(send_latest())
    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            await websocket.send_json({
                "epic": epic,
                "echo": msg,
                "timestamp": datetime.utcnow().isoformat(),
            })
    except WebSocketDisconnect:
        logger.info("WS disconnected: %s", epic)
    finally:
        task.cancel()


# ── Entry point ─────────────────────────────────────────────────────

def main() -> None:
    import uvicorn
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )
    uvicorn.run("agentic_finance.api.server:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
