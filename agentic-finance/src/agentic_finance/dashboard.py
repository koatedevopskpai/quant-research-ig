"""
dashboard.py — Real-time terminal dashboard for the quant-research-ig platform.

USAGE:
    from agentic_finance.dashboard import Dashboard

    dash = Dashboard(order_book, microstructure, toxicity,
                     execution_agent, risk_manager, portfolio_agent,
                     audit_trail)
    dash.run()

    # From another thread:
    dash.stop()

LAYOUT:
    ┌─────────────────────────────────────────────────────────┐
    │  QUANT RESEARCH IG — LIVE DASHBOARD             14:30   │
    ├───────────────────────────┬─────────────────────────────┤
    │  L2 ORDER BOOK            │  MICROSTRUCTURE & TOXICITY  │
    │  Bids ↓        Asks ↑     │  Spread  Depth  Trade Sign  │
    │  price  size   price size │  Toxicity Score             │
    ├───────────────────────────┼─────────────────────────────┤
    │  POSITIONS & P&L          │  RISK METRICS               │
    │  Epic  Side  Size  P&L    │  VaR  Drawdown  Daily P&L   │
    ├───────────────────────────┼─────────────────────────────┤
    │  RECENT DECISIONS         │  PORTFOLIO ALLOCATION       │
    │  Time  Agent  Description │  Strategy  Weight  Capital  │
    └───────────────────────────┴─────────────────────────────┘

DEPENDENCIES:
    rich>=13.0  (install with: pip install rich)
"""

from datetime import datetime, timezone
from typing import Any, Optional

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.align import Align


class Dashboard:
    """Real-time terminal dashboard for the quant-research-ig platform."""

    def __init__(
        self,
        order_book: Any,
        microstructure: Any,
        toxicity: Any,
        execution_agent: Any,
        risk_manager: Any,
        portfolio_agent: Any,
        audit_trail: Any,
        epics: list[str] | None = None,
        refresh_sec: float = 1.0,
        on_refresh: Optional[callable] = None,
    ) -> None:
        self.order_book = order_book
        self.microstructure = microstructure
        self.toxicity = toxicity
        self.execution_agent = execution_agent
        self.risk_manager = risk_manager
        self.portfolio_agent = portfolio_agent
        self.audit_trail = audit_trail
        self.epics = epics or []
        self.refresh_sec = refresh_sec
        self.on_refresh = on_refresh
        self._running = False
        self.console = Console()

    # ── Public API ──────────────────────────────────────────────

    def run(self) -> None:
        """Enter the live dashboard display loop (blocking)."""
        self._running = True
        layout = self._build_layout()

        with Live(layout, console=self.console, screen=True,
                  refresh_per_second=1.0 / self.refresh_sec) as live:
            while self._running:
                if self.on_refresh:
                    self.on_refresh()
                self._refresh(layout)
                live.update(layout)
                import time
                try:
                    time.sleep(self.refresh_sec)
                except KeyboardInterrupt:
                    self._running = False

    def stop(self) -> None:
        """Stop the dashboard display loop."""
        self._running = False

    # ── Layout ──────────────────────────────────────────────────

    def _build_layout(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=1),
        )
        layout["body"].split_row(
            Layout(name="left"),
            Layout(name="right"),
        )
        layout["body"]["left"].split_column(
            Layout(name="order_book", ratio=2),
            Layout(name="positions", ratio=1),
            Layout(name="decisions", ratio=2),
        )
        layout["body"]["right"].split_column(
            Layout(name="microstructure", ratio=2),
            Layout(name="risk", ratio=1),
            Layout(name="portfolio", ratio=2),
        )
        return layout

    def _refresh(self, layout: Layout) -> None:
        layout["header"].update(self._render_header())
        layout["order_book"].update(self._render_order_book())
        layout["positions"].update(self._render_positions())
        layout["decisions"].update(self._render_decisions())
        layout["microstructure"].update(self._render_microstructure())
        layout["risk"].update(self._render_risk())
        layout["portfolio"].update(self._render_portfolio())
        layout["footer"].update(self._render_footer())

    # ── Header ──────────────────────────────────────────────────

    def _render_header(self) -> Panel:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        epics_str = ", ".join(self.epics) if self.epics else "no epics"
        text = Text.assemble(
            (" QUANT RESEARCH IG — LIVE DASHBOARD ", "bold white on blue"),
            "\n",
            (f" {now}  ", "white"),
            (f" [{epics_str}]", "dim white"),
        )
        return Panel(text, style="blue")

    # ── Order Book Panel ────────────────────────────────────────

    def _render_order_book(self) -> Panel:
        table = Table(show_header=True, header_style="bold", box=None, padding=(0, 1))
        table.add_column("Bids ↓", style="green", width=12)
        table.add_column("Size", style="green", justify="right", width=10)
        table.add_column("Asks ↑", style="red", width=12)
        table.add_column("Size", style="red", justify="right", width=10)

        try:
            l2 = self.order_book.l2_snapshot(max_levels=8)
        except (ValueError, Exception):
            table.add_row("(empty)", "", "(empty)", "")
            return Panel(table, title="📊 L2 Order Book", border_style="cyan")

        bids = l2.bids[:8] if l2.bids else []
        asks = l2.asks[:8] if l2.asks else []
        max_rows = max(len(bids), len(asks), 1)

        for i in range(max_rows):
            bid_row = ""
            bid_size = ""
            if i < len(bids):
                bid_row = f"{bids[i].price:.2f}" if hasattr(bids[i], 'price') else ""
                bid_size = f"{bids[i].size:,.0f}" if hasattr(bids[i], 'size') else ""
            ask_row = ""
            ask_size = ""
            if i < len(asks):
                ask_row = f"{asks[i].price:.2f}" if hasattr(asks[i], 'price') else ""
                ask_size = f"{asks[i].size:,.0f}" if hasattr(asks[i], 'size') else ""
            table.add_row(bid_row, bid_size, ask_row, ask_size)

        return Panel(table, title="📊 L2 Order Book", border_style="cyan")

    # ── Microstructure + Toxicity Panel ─────────────────────────

    def _render_microstructure(self) -> Panel:
        grid = Table.grid(padding=(0, 2))
        grid.add_column(style="bold", width=18)
        grid.add_column(style="white", width=16)

        try:
            ss = self.microstructure.spread_stats()
            grid.add_row("Spread (mean)", f"{ss.mean_spread_bps:.2f} bps" if ss.mean_spread_bps else "—")
            grid.add_row("Spread (std)", f"{ss.std_spread_bps:.2f} bps" if ss.std_spread_bps else "—")
        except Exception:
            grid.add_row("Spread", "—")

        try:
            depth = self.order_book.depth_summary()
            imb = depth.imbalance if hasattr(depth, 'imbalance') else 0.0
            grid.add_row("Depth Imbalance", f"{imb:+.3f}")
            grid.add_row("Bid Vol", f"{depth.bid_volume:,.0f}" if depth.bid_volume else "0")
            grid.add_row("Ask Vol", f"{depth.ask_volume:,.0f}" if depth.ask_volume else "0")
        except Exception:
            grid.add_row("Depth", "—")

        try:
            tox = self.toxicity.predict()
            tox_label = "TOXIC" if tox.is_toxic else "clean"
            tox_style = "red bold" if tox.is_toxic else "green"
            grid.add_row("Toxicity", f"{tox.toxicity:.3f}")
            grid.add_row("Status", Text(tox_label, style=tox_style))
        except Exception:
            grid.add_row("Toxicity", "—")

        return Panel(grid, title="🔬 Microstructure & Toxicity", border_style="magenta")

    # ── Positions & P&L Panel ───────────────────────────────────

    def _render_positions(self) -> Panel:
        table = Table(show_header=True, header_style="bold", box=None, padding=(0, 1))
        table.add_column("Epic", width=22)
        table.add_column("Side", width=6)
        table.add_column("Size", justify="right", width=8)

        positions = self.execution_agent.positions
        if not positions:
            table.add_row("(no positions)", "", "")
        else:
            for epic, size in sorted(positions.items()):
                side = "LONG" if size > 0 else "SHORT" if size < 0 else "FLAT"
                side_style = "green" if size > 0 else "red"
                table.add_row(epic[:22], Text(side, style=side_style), f"{abs(size):.0f}")

        return Panel(table, title="💰 Positions", border_style="green")

    # ── Recent Decisions Panel ──────────────────────────────────

    def _render_decisions(self) -> Panel:
        table = Table(show_header=True, header_style="bold", box=None, padding=(0, 1))
        table.add_column("Time", width=10)
        table.add_column("Agent", width=14)
        table.add_column("Decision", width=38)

        try:
            entries = self.audit_trail.query(limit=8)
        except Exception:
            table.add_row("—", "(no data)", "")
            return Panel(table, title="📋 Recent Decisions", border_style="yellow")

        if not entries:
            table.add_row("—", "(no decisions)", "")
        else:
            for entry in reversed(entries):
                d = entry.decision
                ts = d.timestamp.strftime("%H:%M:%S") if hasattr(d.timestamp, 'strftime') else str(d.timestamp)[:8]
                desc = d.description[:38]
                table.add_row(ts, d.agent[:14], desc)

        return Panel(table, title="📋 Recent Decisions", border_style="yellow")

    # ── Risk Metrics Panel ──────────────────────────────────────

    def _render_risk(self) -> Panel:
        grid = Table.grid(padding=(0, 2))
        grid.add_column(style="bold", width=16)
        grid.add_column(style="white", width=18)

        try:
            metrics = self.risk_manager.compute_metrics()
            var_style = "red bold" if metrics.var_parametric < -0.02 else "white"
            dd_style = "red bold" if metrics.current_drawdown > 0.15 else "white"

            grid.add_row("VaR (95%)", Text(f"{metrics.var_parametric:.2%}", style=var_style))
            grid.add_row("VaR Hist", f"{metrics.var_historical:.2%}")
            grid.add_row("Drawdown", Text(f"{metrics.current_drawdown:.2%}", style=dd_style))
            grid.add_row("Daily P&L", f"${metrics.daily_pnl:+,.2f}")
            grid.add_row("Exposure", f"${metrics.total_exposure:,.0f}")
            grid.add_row("Positions", str(metrics.n_positions))

            if metrics.is_breached:
                breaches = "; ".join(metrics.breaches)
                grid.add_row("⚠ BREACH", Text(breaches[:30], style="red bold"))
        except Exception:
            grid.add_row("VaR", "—")
            grid.add_row("Drawdown", "—")

        return Panel(grid, title="⚠ Risk Metrics", border_style="red")

    # ── Portfolio Allocation Panel ──────────────────────────────

    def _render_portfolio(self) -> Panel:
        table = Table(show_header=True, header_style="bold", box=None, padding=(0, 1))
        table.add_column("Strategy", width=20)
        table.add_column("Weight", justify="right", width=8)
        table.add_column("Capital", justify="right", width=12)

        weights = self.portfolio_agent.current_weights
        if not weights:
            table.add_row("(not allocated)", "", "")
        else:
            for sid, w in sorted(weights.items()):
                cap = self.portfolio_agent.total_capital * w
                table.add_row(sid[:20], f"{w:.1%}", f"${cap:,.0f}")

        cash = self.portfolio_agent.total_capital * max(0.0, 1.0 - sum(weights.values())) if weights else self.portfolio_agent.total_capital
        table.add_row("Cash", "", f"${cash:,.0f}")

        return Panel(table, title="📈 Portfolio Allocation", border_style="blue")

    # ── Footer ──────────────────────────────────────────────────

    def _render_footer(self) -> Panel:
        text = Text(" Ctrl+C to stop  |  q to quit", style="dim white")
        return Panel(text)


def run_dashboard(
    order_book=None, microstructure=None, toxicity=None,
    execution_agent=None, risk_manager=None, portfolio_agent=None,
    audit_trail=None, epics=None,
) -> Dashboard:
    """Convenience function to create and run a dashboard."""
    dash = Dashboard(
        order_book=order_book,
        microstructure=microstructure,
        toxicity=toxicity,
        execution_agent=execution_agent,
        risk_manager=risk_manager,
        portfolio_agent=portfolio_agent,
        audit_trail=audit_trail,
        epics=epics,
    )
    dash.run()
    return dash
