"""
Tests for run_production.py — the main pipeline orchestration loop.

Tests use mock components to verify the pipeline logic without real IG/Lightstreamer.
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from agentic_finance.run_production import ProductionRunner


# ── Fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def mock_pipeline():
    """Create a ProductionRunner with all components mocked."""
    runner = ProductionRunner()

    # Replace all internal components with mocks
    runner.message_bus = MagicMock()
    runner.audit_trail = MagicMock()
    runner.research_agent = MagicMock()
    runner.execution_agent = MagicMock()
    runner.risk_manager = MagicMock()
    runner.portfolio_agent = MagicMock()
    runner.tick_consumer = MagicMock()
    runner.books[runner.epics[0]] = MagicMock()
    runner.books[runner.epics[0]].l1_snapshot.return_value = MagicMock(mid_price=100.0)
    runner.analysers[runner.epics[0]] = MagicMock()
    runner.toxicity = MagicMock()

    return runner


@pytest.fixture
def runner_with_mock_connect():
    """Runner with tick_consumer.connect() mocked to test setup + tick_loop."""
    with patch.object(ProductionRunner, 'setup'):
        runner = ProductionRunner()
        runner.message_bus = MagicMock()
        runner.audit_trail = MagicMock()
        runner.research_agent = MagicMock()
        runner.execution_agent = MagicMock()
        runner.risk_manager = MagicMock()
        runner.portfolio_agent = MagicMock()
        runner.tick_consumer = MagicMock()
        runner.books[runner.epics[0]] = MagicMock()
        runner.analysers[runner.epics[0]] = MagicMock()
        runner.toxicity = MagicMock()
        runner._running = True
    return runner


# ── Initialisation tests ───────────────────────────────────────────

class TestInit:
    def test_runner_creates(self):
        runner = ProductionRunner()
        assert hasattr(runner, 'message_bus')
        assert hasattr(runner, 'audit_trail')
        assert hasattr(runner, 'research_agent')
        assert hasattr(runner, 'execution_agent')
        assert hasattr(runner, 'risk_manager')
        assert hasattr(runner, 'portfolio_agent')
        assert hasattr(runner, 'tick_consumer')
        assert hasattr(runner, 'books')
        assert hasattr(runner, 'analysers')
        assert hasattr(runner, 'toxicity')

    def test_epics_defined(self):
        runner = ProductionRunner()
        assert len(runner.epics) > 0
        assert "IX.D.SPTRD.IFD.IP" in runner.epics

    def test_initial_state(self):
        runner = ProductionRunner()
        assert runner._running is False
        assert runner._last_research_time == 0.0
        assert runner._last_risk_check_time == 0.0


# ── Setup tests ─────────────────────────────────────────────────────

class TestSetup:
    def test_setup_creates_components(self):
        with patch.object(ProductionRunner, 'setup'):
            runner = ProductionRunner()

    def test_setup_creates_agents(self):
        runner = ProductionRunner()
        runner.setup()
        assert runner.research_agent is not None
        assert runner.execution_agent is not None

    def test_setup_registers_strategies(self):
        runner = ProductionRunner()
        runner.setup()
        assert len(runner.portfolio_agent.strategies) >= 2

    def test_setup_subscribes_tick_consumer(self):
        runner = ProductionRunner()
        runner.setup()
        assert runner.tick_consumer._subscriptions  # has subscriptions


# ── Tick loop tests ─────────────────────────────────────────────────

class TestTickLoop:
    def test_tick_loop_no_l1(self, mock_pipeline):
        mock_pipeline.books[mock_pipeline.epics[0]].l1_snapshot.side_effect = ValueError("empty")
        mock_pipeline._tick_loop()  # should not raise

    def test_tick_loop_marks_prices_for_risk(self, mock_pipeline):
        mock_pipeline.books[mock_pipeline.epics[0]].l1_snapshot.return_value = MagicMock(
            mid_price=105.0,
        )
        mock_pipeline._tick_loop()
        mock_pipeline.risk_manager.update_mark.assert_called()
        args = mock_pipeline.risk_manager.update_mark.call_args[0]
        assert args[0] == mock_pipeline.epics[0]
        assert args[1] == pytest.approx(105.0)

    def test_tick_loop_runs_research(self, mock_pipeline):
        mock_pipeline.books[mock_pipeline.epics[0]].l1_snapshot.return_value = MagicMock(
            mid_price=100.0,
        )
        mock_pipeline._price_caches[mock_pipeline.epics[0]] = [100.0] * 100  # enough for the 50-point threshold
        mock_pipeline._last_research_time = -9999  # force run
        mock_pipeline._tick_loop()
        mock_pipeline.research_agent.research_signal.assert_called()

    def test_tick_loop_runs_risk_check(self, mock_pipeline):
        mock_pipeline.books[mock_pipeline.epics[0]].l1_snapshot.return_value = MagicMock(
            mid_price=100.0,
        )
        mock_pipeline._last_risk_check_time = -9999  # force run
        mock_pipeline._tick_loop()
        mock_pipeline.risk_manager.check_risk.assert_called()

    def test_tick_loop_research_catches_exception(self, mock_pipeline):
        mock_pipeline.research_agent.research_signal.side_effect = RuntimeError("fail")
        mock_pipeline.books[mock_pipeline.epics[0]].l1_snapshot.return_value = MagicMock(
            mid_price=100.0,
        )
        mock_pipeline._last_research_time = -9999
        mock_pipeline._tick_loop()  # should not raise

    def test_tick_loop_risk_catches_exception(self, mock_pipeline):
        mock_pipeline.risk_manager.check_risk.side_effect = RuntimeError("fail")
        mock_pipeline.books[mock_pipeline.epics[0]].l1_snapshot.return_value = MagicMock(
            mid_price=100.0,
        )
        mock_pipeline._last_risk_check_time = -9999
        mock_pipeline._tick_loop()  # should not raise


# ── Shutdown & cleanup tests ───────────────────────────────────────

class TestShutdown:
    def test_shutdown_sets_running_false(self, mock_pipeline):
        mock_pipeline._running = True
        mock_pipeline._shutdown(None, None)
        assert mock_pipeline._running is False

    def test_cleanup_disconnects(self, mock_pipeline):
        mock_pipeline._cleanup()
        mock_pipeline.tick_consumer.disconnect.assert_called_once()

    def test_cleanup_logs_count(self, mock_pipeline):
        mock_pipeline.audit_trail.count.return_value = 42
        mock_pipeline._cleanup()  # should not raise


# ── Simulator fallback test ─────────────────────────────────────────

class TestSimulatorFallback:
    def test_sim_feed_creates_daemon_thread(self, mock_pipeline):
        """_sim_feed should start a daemon thread and run dashboard."""
        mock_pipeline._running = True
        mock_pipeline.tick_consumer._connected = False
        mock_sim = MagicMock()
        mock_sim.generate_ticks.return_value = []

        mock_dash = MagicMock()

        with patch('threading.Thread', autospec=True) as mock_thread:
            mock_pipeline._sim_feed(mock_sim, mock_dash)
            mock_thread.assert_called_once()
            args, kwargs = mock_thread.call_args
            assert kwargs.get('daemon', False) is True

    def test_sim_feed_stops_on_running_false(self, mock_pipeline):
        mock_pipeline._running = False
        mock_sim = MagicMock()
        mock_dash = MagicMock()
        mock_pipeline._sim_feed(mock_sim, mock_dash)
        mock_sim.generate_ticks.assert_not_called()
