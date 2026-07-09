"""
ml_for_finance/__init__.py — ML for Finance model registry.
"""

from ml_for_finance.regime.hmm_detector import HMMRegimeDetector, RegimeResult
from ml_for_finance.forecasting.xgboost_forecaster import XGBoostForecaster, ForecastResult
from ml_for_finance.deep_learning.lstm_predictor import LSTMPredictor, LSTMResult
from ml_for_finance.sentiment.sentiment_model import SentimentAugmentedModel, SentimentResult

__all__ = [
    "HMMRegimeDetector", "RegimeResult",
    "XGBoostForecaster", "ForecastResult",
    "LSTMPredictor", "LSTMResult",
    "SentimentAugmentedModel", "SentimentResult",
]
