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
from typing import Any, NoReturn

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
from agentic_finance.research_agent import ResearchAgent
from agentic_finance.model_adapters import ForecastAdapter, RegimeAdapter
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
# Below this, a research signal is logged but not turned into an order. Without
# a floor, a forecast indistinguishable from noise still trades at ~zero size.
MIN_SIGNAL_CONFIDENCE = 0.10


class ProductionRunner:
    def __init__(self) -> None:
        logger.info("Initialising production runner")
        self._running = False

        # Infrastructure
        self.message_bus = MessageBus()
        self.audit_trail = AuditTrail()

        # Epics
        self.epics = list(EPICS)

        # HFT pipeline — one book, one analyser and one price history PER EPIC.
        # Sharing them was a data-corruption defect: feeding instrument A at
        # 100/101 and B at 1/1.1 produced a crossed book (bid 100, ask 1.1) and
        # a mixed price series that research then labelled with the first EPIC.
        self.tick_consumer = TickConsumer()
        self.books: dict[str, OrderBook] = {epic: OrderBook() for epic in self.epics}
        self.analysers: dict[str, MicrostructureAnalyser] = {
            epic: MicrostructureAnalyser() for epic in self.epics
        }
        self._price_caches: dict[str, list[float]] = {epic: [] for epic in self.epics}

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

    # ── Setup ───────────────────────────────────────────────────────

    def setup(self) -> None:
        logger.info("Setting up production pipeline")

        # Risk values P&L and exposure from the execution ledger, not from a
        # price-level heuristic.
        self.risk_manager.set_ledger(self.execution_agent.ledger)

        # Subscribe TickConsumer to all epics
        for epic in EPICS:
            self.tick_consumer.subscribe(epic, self._on_market_data)
            logger.info("  Subscribed epic: %s", epic)

        # Wire agents to ExecutionAgent for toxicity + impact checks
        self.execution_agent.set_toxicity_detector(self.toxicity)
        self.execution_agent.set_execution_simulator(self.execution_sim)

        # Register the real research models. Adapters import the ML packages
        # lazily, so this stays cheap; a missing dependency fails here with a
        # clear message instead of silently producing zero predictions.
        self.research_agent.register_model("forecast", ForecastAdapter())
        self.research_agent.register_model("regime", RegimeAdapter())

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

        # Wire bus subscriptions for inter-agent communication.
        # NOTE: these are STATE notifications. The halt command is never
        # republished by the handler that consumes it, which is what caused the
        # review's recursive kill events.
        self.message_bus.subscribe("research.completed",
                                   self._on_research_completed)
        self.message_bus.subscribe("risk.halted", self._on_risk_halted)
        self.message_bus.subscribe("risk.resumed", self._on_risk_resumed)

    # ── Market data callback ────────────────────────────────────────

    def _on_market_data(self, msg: MarketDataMessage) -> None:
        """
        Route an update to its own instrument's state.

        WHY routing on msg.epic:
            The runner subscribes to several EPICs. Previously every update went
            into one book, one analyser and one price list, so two instruments
            contaminated each other (a crossed book, and a mixed research input
            labelled with whichever EPIC came first).
        """
        epic = msg.epic
        book = self.books.get(epic)
        if book is None:
            logger.debug("Ignoring update for unknown epic %s", epic)
            return

        if msg.message_type == "L1":
            # IG L1 is a snapshot of the top of book, not a depth diff.
            book.apply_l1(msg.bid_price, msg.ask_price, msg.bid_size, msg.ask_size)
            if msg.bid_price > 0 and msg.ask_price > 0:
                self.risk_manager.update_mark(epic, (msg.bid_price + msg.ask_price) / 2.0)
        elif msg.message_type == "L2":
            if msg.bid_price > 0:
                book.apply_diff("bid", msg.bid_price, msg.bid_size, 0)
            if msg.ask_price > 0:
                book.apply_diff("ask", msg.ask_price, msg.ask_size, 0)

        analyser = self.analysers.get(epic)
        if analyser is not None:
            analyser.on_tick(msg)

        if msg.last_price > 0:
            cache = self._price_caches[epic]
            cache.append(msg.last_price)
            if len(cache) > 1000:
                del cache[:-1000]

    # ── Agent wiring callbacks ──────────────────────────────────────

    def _on_research_completed(self, msg) -> None:
        """Convert a research result into a trading signal."""
        # Enforcement point 1 (runner). The execution agent also refuses signals
        # while halted, but the runner must not even ask.
        if (self.risk_manager.is_halted is True
                or self.execution_agent.is_active is False
                or self.risk_manager.can_trade() is False):
            logger.warning("Trading halted or limits breached — discarding research result")
            return

        payload = msg.payload
        epic = payload.get("epic", "")
        output = payload.get("output", {})
        summary = payload.get("summary", "")

        if "error" in output:
            logger.warning("Research error for %s: %s", epic, output["error"])
            return

        book = self.books.get(epic)
        if book is None:
            logger.warning("No book for %s — skipping signal", epic)
            return
        try:
            l1 = book.l1_snapshot()
        except ValueError:
            logger.warning("No L1 data for %s — skipping signal", epic)
            return

        price = l1.mid_price

        # The signal policy is defined once, in the research agent: direction
        # comes from a return forecast, confidence from its magnitude (halved in
        # the highest-volatility regime). A regime label is never parsed into a
        # direction, which is what previously made every signal a coin flip.
        try:
            direction = float(output.get("direction", 0.0))
            confidence = float(output.get("confidence", 0.0))
        except (TypeError, ValueError):
            logger.warning("Malformed research output for %s: %s", epic, output)
            return

        if direction == 0.0:
            logger.info("No directional view for %s (regime=%s)",
                        epic, output.get("regime_label"))
            return
        if confidence < MIN_SIGNAL_CONFIDENCE:
            logger.info("Signal for %s below confidence floor (%.3f < %.2f)",
                        epic, confidence, MIN_SIGNAL_CONFIDENCE)
            return

        signal = Signal(
            epic=epic, direction=direction, confidence=confidence,
            source=f"research:{output.get('regime_label') or 'signal'}",
        )
        order = self.execution_agent.process_signal(signal, price)
        if order is not None:
            logger.info("Order created: %s %s %.0f @ %.2f",
                         order.side, order.epic, order.size, price)
        else:
            logger.info("No order for %s (veto or zero size)", epic)

    def _on_risk_halted(self, msg) -> None:
        """Risk halted trading: stop accepting signals (idempotent)."""
        reason = msg.payload.get("reason", "risk halt") if msg.payload else "risk halt"
        logger.warning("RISK HALT: %s", reason)
        self.execution_agent.halt(reason)

    def _on_risk_resumed(self, msg) -> None:
        """Risk resumed trading: allow signals again (idempotent)."""
        logger.warning("RISK RESUME requested")
        self.execution_agent.resume()

    # ── Periodic tasks ──────────────────────────────────────────────

    def _run_research(self) -> None:
        """Run ML research on the latest market data."""
        import numpy as np
        import pandas as pd

        researched = 0
        for epic in self.epics:
            cache = self._price_caches[epic]
            if len(cache) < 50:
                continue

            # Models require a unique, increasing DatetimeIndex and positive
            # prices. The cache is a bare list of last-traded prices, so attach
            # a synthetic minute index; inference uses the return series, not
            # the calendar labels.
            index = pd.date_range(
                end=pd.Timestamp.now(tz="UTC").normalize(),
                periods=len(cache), freq="min",
            )
            prices = pd.Series(np.asarray(cache, dtype=float), index=index)

            result = self.research_agent.research_signal(epic, prices)
            logger.info("Research signal for %s: %s", epic, str(result.summary)[:160])
            researched += 1

        if researched == 0:
            logger.info(
                "Not enough price data for research (need 50 points per epic)"
            )

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
        # Mark every instrument from its own book; risk values the ledger against
        # these marks. The old code booked (mid - 100) * 10 regardless of
        # holdings, which was not P&L in any meaningful sense.
        for epic, book in self.books.items():
            try:
                l1 = book.l1_snapshot()
            except ValueError:
                continue
            if l1 and l1.mid_price > 0:
                self.risk_manager.update_mark(epic, l1.mid_price)

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
            order_book=self.books[self.epics[0]],
            microstructure=self.analysers[self.epics[0]],
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
