"""Check that the five local distributions and important imports are available."""

import importlib
import importlib.metadata
import sys


def main():
    if sys.version_info[:2] != (3, 12):
        raise SystemExit("The tested workspace baseline requires Python 3.12; run uv sync")
    for distribution, module in [
        ("ig-market-data", "ig_market_data.models"),
        ("quant-methods", "quant_methods.validation.backtester"),
        ("hft-engine", "hft_engine.book.order_book"),
        ("ml-for-finance", "ml_for_finance.forecasting.xgboost_forecaster"),
        ("agentic-finance", "agentic_finance.run_production"),
    ]:
        importlib.import_module(module)
        print(f"OK {distribution} {importlib.metadata.version(distribution)}")


if __name__ == "__main__":
    main()
