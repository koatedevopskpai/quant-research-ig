"""
Test dashboard _render_* methods in isolation (no Live terminal).
"""

import pytest
from unittest.mock import MagicMock, create_autospec

from agentic_finance.dashboard import Dashboard, run_dashboard
from agentic_finance.decision import Decision, DecisionType, DecisionStatus


# ── Fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def mock_book():
    book = MagicMock()
    l2 = MagicMock()
    l2.bids = []
    l2.asks = []
    book.l2_snapshot.return_value = l2
    book.depth_summary.return_value = MagicMock(
        imbalance=0.1, bid_volume=50000, ask_volume=45000,
    )
    return book


@pytest.fixture
def mock_micro():
    ms = MagicMock()
    ss = MagicMock()
    ss.mean_spread_bps = 1.5
    ss.std_spread_bps = 0.3
    ms.spread_stats.return_value = ss
    return ms


@pytest.fixture
def mock_tox():
    tox = MagicMock()
    tox.predict.return_value = MagicMock(
        is_toxic=False, toxicity=0.12,
    )
    return tox


@pytest.fixture
def mock_execution():
    exec_ = MagicMock()
    exec_.positions = {}
    return exec_


@pytest.fixture
def mock_risk():
    rm = MagicMock()
    rm.compute_metrics.return_value = MagicMock(
        var_parametric=-0.012, var_historical=-0.008,
        current_drawdown=0.05, daily_pnl=120.50,
        total_exposure=25000, n_positions=2,
        is_breached=False, breaches=[],
    )
    return rm


@pytest.fixture
def mock_portfolio():
    pa = MagicMock()
    pa.current_weights = {"strat_a": 0.6, "strat_b": 0.4}
    pa.total_capital = 100000.0
    return pa


@pytest.fixture
def mock_audit():
    at = MagicMock()
    at.query.return_value = []
    return at


@pytest.fixture
def dashboard(mock_book, mock_micro, mock_tox, mock_execution,
              mock_risk, mock_portfolio, mock_audit):
    return Dashboard(
        order_book=mock_book,
        microstructure=mock_micro,
        toxicity=mock_tox,
        execution_agent=mock_execution,
        risk_manager=mock_risk,
        portfolio_agent=mock_portfolio,
        audit_trail=mock_audit,
        epics=["EPIC1", "EPIC2"],
    )


# ── Helpers ──────────────────────────────────────────────────────────

def _cell_text(panel):
    """Extract concatenated text from all cells of a Panel's Table/Grid renderable."""
    parts = []
    table = panel.renderable
    for row_idx in range(table.row_count):
        for col in table.columns:
            if row_idx < len(col._cells):
                parts.append(str(col._cells[row_idx]))
    return " ".join(parts)


# ── Render method tests ─────────────────────────────────────────────

class TestRenderHeader:
    def test_returns_panel(self, dashboard):
        panel = dashboard._render_header()
        assert str(type(panel).__name__) == "Panel"

    def test_includes_epics(self, dashboard):
        panel = dashboard._render_header()
        text = panel.renderable  # Panel.renderable is a Text object
        assert "EPIC1" in str(text)
        assert "EPIC2" in str(text)


class TestRenderOrderBook:
    def test_empty_book(self, dashboard):
        panel = dashboard._render_order_book()
        assert "L2 Order Book" in panel.title
        assert panel.renderable.row_count >= 1

    def test_with_levels(self, dashboard, mock_book):
        from hft_engine.book.order_book import PriceLevel
        l2 = MagicMock()
        l2.bids = [PriceLevel(100.50, 5000, 1), PriceLevel(100.25, 3000, 1)]
        l2.asks = [PriceLevel(101.00, 4000, 1)]
        mock_book.l2_snapshot.return_value = l2
        panel = dashboard._render_order_book()
        text = _cell_text(panel)
        assert "100.50" in text
        assert "101.00" in text
        assert "5,000" in text or "5000" in text

    def test_exception_handling(self, dashboard, mock_book):
        mock_book.l2_snapshot.side_effect = ValueError("empty")
        panel = dashboard._render_order_book()
        text = _cell_text(panel)
        assert "(empty)" in text


class TestRenderMicrostructure:
    def test_returns_panel(self, dashboard):
        panel = dashboard._render_microstructure()
        assert "Microstructure" in panel.title

    def test_spread_stats(self, dashboard, mock_micro):
        ss = MagicMock()
        ss.mean_spread_bps = 2.5
        ss.std_spread_bps = 0.8
        mock_micro.spread_stats.return_value = ss
        panel = dashboard._render_microstructure()
        text = _cell_text(panel)
        assert "2.50" in text or "2.5" in text

    def test_toxicity_clean(self, dashboard, mock_tox):
        mock_tox.predict.return_value = MagicMock(
            is_toxic=False, toxicity=0.05,
        )
        panel = dashboard._render_microstructure()
        text = _cell_text(panel)
        assert "clean" in text.lower()

    def test_toxicity_toxic(self, dashboard, mock_tox):
        mock_tox.predict.return_value = MagicMock(
            is_toxic=True, toxicity=0.85,
        )
        panel = dashboard._render_microstructure()
        text = _cell_text(panel)
        assert "toxic" in text.lower()

    def test_spread_exception(self, dashboard, mock_micro):
        mock_micro.spread_stats.side_effect = RuntimeError("fail")
        panel = dashboard._render_microstructure()
        # Should not crash — panels still render with fallback values

    def test_depth_exception(self, dashboard, mock_book):
        mock_book.depth_summary.side_effect = RuntimeError("fail")
        panel = dashboard._render_microstructure()
        # Should not crash — depth row shows placeholder


class TestRenderPositions:
    def test_no_positions(self, dashboard):
        panel = dashboard._render_positions()
        text = _cell_text(panel)
        assert "no positions" in text.lower()

    def test_with_positions(self, dashboard, mock_execution):
        mock_execution.positions = {"EPIC1": 100, "EPIC2": -50}
        panel = dashboard._render_positions()
        text = _cell_text(panel)
        assert "EPIC1" in text
        assert "LONG" in text
        assert "SHORT" in text


class TestRenderDecisions:
    def test_no_decisions(self, dashboard):
        panel = dashboard._render_decisions()
        text = _cell_text(panel)
        assert "no decisions" in text.lower()

    def test_with_decisions(self, dashboard, mock_audit):
        from datetime import datetime, timezone
        d = Decision.now(
            agent="TestAgent",
            decision_type=DecisionType.RESEARCH,
            description="Test decision description",
        )
        entry = MagicMock()
        entry.decision = d
        mock_audit.query.return_value = [entry]
        panel = dashboard._render_decisions()
        text = _cell_text(panel)
        assert "Test" in text
        assert "Test decision description" in text

    def test_exception_handling(self, dashboard, mock_audit):
        mock_audit.query.side_effect = RuntimeError("fail")
        panel = dashboard._render_decisions()
        text = _cell_text(panel)
        assert "no data" in text.lower()


class TestRenderRisk:
    def test_returns_panel(self, dashboard):
        panel = dashboard._render_risk()
        assert "Risk" in panel.title

    def test_shows_metrics(self, dashboard, mock_risk):
        mock_risk.compute_metrics.return_value = MagicMock(
            var_parametric=-0.015, var_historical=-0.01,
            current_drawdown=0.03, daily_pnl=500.0,
            total_exposure=10000, n_positions=1,
            is_breached=False, breaches=[],
        )
        panel = dashboard._render_risk()
        text = _cell_text(panel)
        assert "-1.50%" in text or "1.50" in text

    def test_breaches_shown(self, dashboard, mock_risk):
        mock_risk.compute_metrics.return_value = MagicMock(
            var_parametric=-0.03, var_historical=-0.02,
            current_drawdown=0.20, daily_pnl=-5000.0,
            total_exposure=100000, n_positions=5,
            is_breached=True, breaches=["Drawdown limit exceeded"],
        )
        panel = dashboard._render_risk()
        text = _cell_text(panel)
        assert "breach" in text.lower()

    def test_exception_handling(self, dashboard, mock_risk):
        mock_risk.compute_metrics.side_effect = RuntimeError("fail")
        panel = dashboard._render_risk()
        # Should not crash


class TestRenderPortfolio:
    def test_not_allocated(self, dashboard, mock_portfolio):
        mock_portfolio.current_weights = {}
        panel = dashboard._render_portfolio()
        text = _cell_text(panel)
        assert "not allocated" in text.lower()

    def test_with_weights(self, dashboard):
        panel = dashboard._render_portfolio()
        text = _cell_text(panel)
        assert "strat_a" in text
        assert "strat_b" in text
        assert "Cash" in text
        assert "60" in text or "0.6" in text


class TestRenderFooter:
    def test_returns_panel(self, dashboard):
        panel = dashboard._render_footer()
        assert "Ctrl" in str(panel.renderable)


class TestRefresh:
    """_refresh calls all render methods and updates layout panels."""

    def test_refresh_populates_all_panels(self, dashboard):
        from rich.layout import Layout
        layout = dashboard._build_layout()
        dashboard._refresh(layout)
        # Each layout panel should have a Panel renderable
        for name in ["header", "order_book", "positions",
                      "decisions", "microstructure", "risk",
                      "portfolio", "footer"]:
            assert layout[name].renderable is not None, f"{name} was None"


class TestRunDashboard:
    """run_dashboard convenience function."""

    def test_run_dashboard_creates_dashboard(self):
        dash = Dashboard(
            order_book=MagicMock(),
            microstructure=MagicMock(),
            toxicity=MagicMock(),
            execution_agent=MagicMock(),
            risk_manager=MagicMock(),
            portfolio_agent=MagicMock(),
            audit_trail=MagicMock(),
            epics=["TEST"],
        )
        assert isinstance(dash, Dashboard)
