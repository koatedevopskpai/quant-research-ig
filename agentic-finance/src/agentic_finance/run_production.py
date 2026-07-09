"""
run_production.py — Production main loop for the quant-research-ig platform.

Composes all five phases into a unified real-time pipeline:

    IG Lightstreamer ──> TickConsumer ──> OrderBook
        (or MarketSimulator)      │            │
                                  │            v
                                  │    MicrostructureAnalyser
                                  v
                             ResearchAgent ──> ExecutionAgent
                                                    │
                                                    v
                                              RiskManager
                                                    │
                                                    v
                                             PortfolioAgent

Everything is wired through the MessageBus for loose coupling, and
every decision is recorded in the AuditTrail for full traceability.

USAGE:
    python -m agentic_finance.run_production

BEHAVIOUR:
    1. Connects to IG Lightstreamer if credentials are available
       (IG_API_KEY, IG_USERNAME, IG_PASSWORD in .env).
    2. Falls back to MarketSimulator (synthetic ticks) if IG is
       unavailable — useful for testing the pipeline offline.
    3. Runs a main loop that periodically:
       a) Fetches recent market data and runs ML models (research).
       b) Generates trading signals from research results.
       c) Processes signals through the Execution Agent.
       d) Checks risk metrics (VaR, drawdown) and triggers kill switch.
       e) Rebalances portfolio allocation.
    4. All decisions are recorded in the AuditTrail and published
       on the MessageBus.
    5. Cleanly disconnects on Ctrl+C.

PERFORMANCE NOTE:
    Lightstreamer runs on background threads — the main loop sleeps
    between periodic tasks. For sub-100ms reaction times, replace
    the sleep loop with a proper event-driven architecture.
"""

import logging
import signal
import sys
import time
from datetime import datetime, timezone
from typing import NoReturn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
logger = logging.getLogger("run_production")

# ── Infrastructure ──────────────────────────────────────────────────
from agentic_finance.message_bus import MessageBus
from agentic_finance.audit_trail import AuditTrail
from agentic_finance.decision import Decision, DecisionType, DecisionStatus

# ── HFT Pipeline ────────────────────────────────────────────────────
from hft_engine.feeds.tick_consumer import TickConsumer, MarketDataMessage
from hft_engine.book.order_book import OrderBook
from hft_engine.microstructure.analyser import MicrostructureAnalyser
from hft_engine.toxicity.detector import ToxicityDetector
from hft_engine.execution.simulator import ExecutionSimulator

# ── Agents ──────────────────────────────────────────────────────────
from agentic_finance.research_agent import ResearchAgent, ResearchRequest
from agentic_finance.execution_agent import ExecutionAgent, Signal
from agentic_finance.risk_manager import RiskManager
from agentic_finance.portfolio_agent import PortfolioAgent
from agentic_finance.dashboard import Dashboard

# ── Config ──────────────────────────────────────────────────────────
EPICS = [
    "IX.D.SPTRD.IFD.IP",     # US 500 (S&P 500 CFD)
    "CS.D.EURUSD.CFD.IP",    # EUR/USD
]
RESEARCH_INTERVAL = 300      # seconds (5 min)
RISK_CHECK_INTERVAL = 60     # seconds (1 min)
INITIAL_CAPITAL = 100_000.0


class ProductionRunner:
    def __init__(self) -> None:
        logger.info("Initialising production runner")
        self._running = False

        # Infrastructure
        self.message_bus = MessageBus()
        self.audit_trail = AuditTrail()

        # HFT pipeline
        self.tick_consumer = TickConsumer()
        self.order_book = OrderBook()
        self.microstructure = MicrostructureAnalyser()
        self.toxicity = ToxicityDetector()
        self.execution_sim = ExecutionSimulator()

        # Agents
        self.research_agent = ResearchAgent(self.message_bus, self.audit_trail)
        self.execution_agent = ExecutionAgent(
            self.message_bus, self.audit_trail,
            initial_capital=INITIAL_CAPITAL,
        )
        self.risk_manager = RiskManager(
            self.message_bus, self.audit_trail,
            initial_capital=INITIAL_CAPITAL,
        )
        self.portfolio_agent = PortfolioAgent(
            self.message_bus, self.audit_trail,
            total_capital=INITIAL_CAPITAL,
        )

        # Timing state
        self._last_research_time: float = 0.0
        self._last_risk_check_time: float = 0.0
        self._price_cache: list[float] = []

    # ── Setup ───────────────────────────────────────────────────────

    def setup(self) -> None:
        logger.info("Setting up production pipeline")

        # Subscribe TickConsumer to all epics
        for epic in EPICS:
            self.tick_consumer.subscribe(epic, self._on_market_data)
            logger.info("  Subscribed epic: %s", epic)

        # Wire agents to ExecutionAgent for toxicity + impact checks
        self.execution_agent.set_toxicity_detector(self.toxicity)
        self.execution_agent.set_execution_simulator(self.execution_sim)

        # Register strategies on portfolio agent
        self.portfolio_agent.register_strategy(
            "index_momentum", expected_return=0.12, expected_vol=0.20,
        )
        self.portfolio_agent.register_strategy(
            "fx_carry", expected_return=0.06, expected_vol=0.10,
        )

        # Initial capital allocation
        state = self.portfolio_agent.allocate(method="risk_parity")
        logger.info("Initial allocation: %s", state)

        # Wire bus subscriptions for inter-agent communication
        self.message_bus.subscribe("research.completed",
                                   self._on_research_completed)
        self.message_bus.subscribe("risk.kill", self._on_kill_switch)

    # ── Market data callback ────────────────────────────────────────

    def _on_market_data(self, msg: MarketDataMessage) -> None:
        """Handle an incoming market data tick from the feed."""
        if msg.message_type == "L1":
            self.order_book.apply_diff(
                "bid", msg.bid_price, msg.bid_size,
            ) if msg.bid_price > 0 else None
            self.order_book.apply_diff(
                "ask", msg.ask_price, msg.ask_size,
            ) if msg.ask_price > 0 else None

        self.microstructure.on_tick(msg)

        if msg.last_price > 0:
            self._price_cache.append(msg.last_price)
            if len(self._price_cache) > 1000:
                self._price_cache = self._price_cache[-1000:]

    # ── Agent wiring callbacks ──────────────────────────────────────

    def _on_research_completed(self, msg) -> None:
        """Convert a research result into a trading signal."""
        payload = msg.payload
        epic = payload.get("epic", "")
        output = payload.get("output", {})
        summary = payload.get("summary", "")

        if "error" in output:
            logger.warning("Research error for %s: %s", epic, output["error"])
            return

        l1 = self.order_book.l1_snapshot()
        if l1 is None:
            logger.warning("No L1 data for %s — skipping signal", epic)
            return

        price = l1.mid_price
        direction = 1.0 if output.get("regime_labels", [0])[-1] == 1 else -1.0
        confidence = min(1.0, abs(output.get("prediction_mean", 0)) / 0.02)

        signal = Signal(
            epic=epic, direction=direction, confidence=confidence,
            source=payload.get("model_type", "research"),
        )
        order = self.execution_agent.process_signal(signal, price)
        if order is not None:
            logger.info("Order created: %s %s %.0f @ %.2f",
                         order.side, order.epic, order.size, price)
        else:
            logger.info("No order for %s (veto or zero size)", epic)

    def _on_kill_switch(self, msg) -> None:
        """Handle kill switch from risk manager."""
        logger.warning("KILL SWITCH triggered: %s", msg.payload)
        self.execution_agent.kill()

    # ── Periodic tasks ──────────────────────────────────────────────

    def _run_research(self) -> None:
        """Run ML research on the latest market data."""
        if len(self._price_cache) < 50:
            logger.info("Not enough price data for research (%d points)",
                         len(self._price_cache))
            return

        import numpy as np
        prices = np.array(self._price_cache)
        epic = EPICS[0]
        request = ResearchRequest(epic=epic, model_type="regime")
        result = self.research_agent.research(request, prices)
        logger.info("Research complete: %s", result.summary[:100])

    def _check_risk(self) -> None:
        """Periodic risk assessment."""
        metrics = self.risk_manager.check_risk()
        if metrics.is_breached:
            logger.warning("Risk breached: %s", "; ".join(metrics.breaches))
            self.risk_manager.kill_if_needed()
        else:
            logger.info("Risk OK — VaR=%.4f drawdown=%.2f%% positions=%d",
                         metrics.var_parametric,
                         metrics.current_drawdown * 100,
                         metrics.n_positions)

    def _tick_loop(self) -> None:
        """Background periodic tasks: research, risk check, PnL."""
        l1 = None
        try:
            l1 = self.order_book.l1_snapshot()
        except ValueError:
            pass

        if l1 and l1.mid_price > 0:
            self.risk_manager.record_pnl((l1.mid_price - 100.0) * 10)

        now = time.time()
        if now - self._last_research_time > RESEARCH_INTERVAL:
            try:
                self._run_research()
            except Exception as e:
                logger.warning("Research task failed: %s", e)
            self._last_research_time = now

        if now - self._last_risk_check_time > RISK_CHECK_INTERVAL:
            try:
                self._check_risk()
            except Exception as e:
                logger.warning("Risk check failed: %s", e)
            self._last_risk_check_time = now

    def run(self) -> None:
        self.setup()
        self._running = True

        self.tick_consumer.connect()

        dashboard = Dashboard(
            order_book=self.order_book,
            microstructure=self.microstructure,
            toxicity=self.toxicity,
            execution_agent=self.execution_agent,
            risk_manager=self.risk_manager,
            portfolio_agent=self.portfolio_agent,
            audit_trail=self.audit_trail,
            epics=EPICS,
            refresh_sec=1.0,
            on_refresh=self._tick_loop,
        )

        # If IGStreamer unavailable, use MarketSimulator
        if self.tick_consumer._streamer is None:
            from hft_engine.market_sim.generator import MarketSimulator
            sim = MarketSimulator(
                initial_price=100.0, volatility=0.02,
                l1_freq_hz=10.0, trade_freq_hz=1.0,
                depth_levels=10, seed=42,
            )
            self.tick_consumer.connect_market_simulator(sim)
            logger.info("Running with MarketSimulator + Dashboard")
            self._sim_feed(sim, dashboard)
            return

        logger.info("Running with IG Lightstreamer + Dashboard")
        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)
        dashboard.run()

        self._cleanup()

    def _sim_feed(
        self, sim: Any, dashboard: Dashboard
    ) -> None:
        """Feed synthetic ticks from MarketSimulator while dashboard runs."""
        import threading

        def feed():
            while self._running:
                ticks = sim.generate_ticks(3.0)
                for tick in ticks:
                    if not self._running:
                        return
                    msg = MarketDataMessage(
                        epic=EPICS[0],
                        message_type=tick.tick_type,
                        bid_price=tick.bid_price,
                        ask_price=tick.ask_price,
                        bid_size=tick.bid_size,
                        ask_size=tick.ask_size,
                        last_price=tick.last_price,
                        last_size=tick.last_size,
                        timestamp_ns=int(tick.timestamp * 1e9),
                    )
                    self._on_market_data(msg)
                import time
                time.sleep(0.05)

        t = threading.Thread(target=feed, daemon=True)
        t.start()

        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)
        dashboard.run()
        self._cleanup()

    def _shutdown(self, sig, frame) -> None:
        """Graceful shutdown on signal."""
        logger.info("Shutdown signal received")
        self._running = False

    def _cleanup(self) -> None:
        """Clean up connections and print summary."""
        self.tick_consumer.disconnect()
        summary = self.audit_trail.to_json()
        n = self.audit_trail.count()
        logger.info("Shutdown complete — %d decisions recorded", n)


def main() -> NoReturn:
    runner = ProductionRunner()
    try:
        runner.run()
    except Exception as e:
        logger.exception("Fatal error in production runner: %s", e)
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
