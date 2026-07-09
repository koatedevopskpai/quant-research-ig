# quant-methods — Quantitative Methods & Pricing Models

## Objective
Implement classical quant finance models: Black-Scholes, binomial trees,
Monte Carlo simulation, GARCH volatility modelling, and portfolio
optimisation — all backtested against real IG market data.

## Projects (Planned)
1. **Black-Scholes Pricer** — European option pricing + Greeks
2. **Binomial Tree** — American option pricing via CRR lattice
3. **Monte Carlo Engine** — Path-dependent option simulation (Asian, barrier)
4. **GARCH Volatility** — GARCH(1,1) fit to IG index returns
5. **Portfolio Optimisation** — Mean-variance, Black-Litterman, risk parity

## Key Files
| File | Purpose |
|------|---------|
| src/quant_methods/pricers/ | Option pricing implementations |
| src/quant_methods/volatility/ | Volatility models |
| src/quant_methods/portfolio/ | Portfolio optimisation |

## Status
- [ ] Project skeleton created
- [ ] Data dependency on ig-market-data established
