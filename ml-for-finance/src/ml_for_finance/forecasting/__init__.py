"""
forecasting/__init__.py — Forecasting model registry.
"""

from ml_for_finance.forecasting.xgboost_forecaster import XGBoostForecaster, ForecastResult

__all__ = ["XGBoostForecaster", "ForecastResult"]
