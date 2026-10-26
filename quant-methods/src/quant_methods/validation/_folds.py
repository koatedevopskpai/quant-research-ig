"""Shared supervised-fold validation; no model sees held-out labels during fit."""

import numpy as np


def supervised_data(x, y):
    features = np.asarray(x, dtype=float)
    targets = np.asarray(y, dtype=float)
    if features.ndim == 1:
        features = features.reshape(-1, 1)
    if features.ndim != 2 or targets.ndim != 1 or len(features) != len(targets):
        raise ValueError("x must be a feature matrix aligned with a one-dimensional y")
    if not np.isfinite(features).all() or not np.isfinite(targets).all():
        raise ValueError("x and y must be finite")
    return features, targets


def event_ends(n, label_end):
    starts = np.arange(n)
    if label_end is None:
        return starts
    ends = np.asarray(label_end)
    if ends.shape != (n,) or not np.issubdtype(ends.dtype, np.integer):
        raise ValueError("label_end must be an aligned integer vector of inclusive row positions")
    if np.any(ends < starts) or np.any(ends >= n):
        raise ValueError("Each label_end must lie between its observation and the final row")
    return ends


def sharpe(returns, periods_per_year):
    if len(returns) < 2:
        raise ValueError("At least two held-out returns are required")
    sd = returns.std(ddof=1)
    return float(returns.mean() / sd * np.sqrt(periods_per_year)) if sd > 0 else 0.0


def evaluate_fold(x, y, train, test, model_factory, returns_fn):
    if len(train) < 2 or len(test) < 2:
        raise ValueError("Purging left fewer than two training or testing observations")
    model = model_factory()
    model.fit(x[train].copy(), y[train].copy())
    predictions = np.asarray(model.predict(x[test].copy()), dtype=float)
    if predictions.shape != (len(test),) or not np.isfinite(predictions).all():
        raise ValueError("Model must produce one finite prediction per test row")
    returns = np.asarray(returns_fn(y[test].copy(), predictions.copy()), dtype=float)
    if returns.shape != (len(test),) or not np.isfinite(returns).all():
        raise ValueError("returns_fn must produce one finite strategy return per test row")
    return {
        "train_indices": train.copy(),
        "test_indices": test.copy(),
        "predictions": predictions,
        "oos_returns": returns,
    }
