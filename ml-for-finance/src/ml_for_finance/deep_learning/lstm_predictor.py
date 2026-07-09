"""
lstm_predictor.py — PyTorch LSTM for sequence-based price prediction.

WHY LSTM FOR FINANCE:
    LSTMs (Hochreiter & Schmidhuber, 1997) are designed to learn long-range
    dependencies in sequential data. For financial time series:

    1. **Memory cells** — LSTMs maintain a cell state that can carry
       information across hundreds of time steps, capturing cycles and
       trends that CNNs and basic RNNs miss.
    2. **Gating mechanism** — Input, forget, and output gates control what
       information is stored, discarded, or emitted. This is analogous to
       regime-dependent feature selection.
    3. **Sequence-to-sequence** — LSTMs naturally handle variable-length
       input sequences and can predict multiple steps ahead.

WHY LSTM vs. XGBoost:
    - XGBoost: Better for tabular features with clear predictive power.
      Faster to train, interpretable feature importance.
    - LSTM: Better when the temporal structure matters (sequences of
      returns, not just lagged values). Can learn patterns that span
      multiple time scales.

    Practical rule: Start with XGBoost. If you need sequence-level patterns
    (e.g., "the shape of the last 20 returns predicts the next day"),
    use LSTM.

ARCHITECTURE:
    Input: (batch, seq_len, n_features)
        └─ LSTM(hidden_dim, num_layers)
            └─ Dropout(p=0.2)
                └─ Linear(hidden_dim → 1)  # single-step forecast

    WHY single-step (not multi-step):
        Multi-step forecasting compounds errors. For trading, predicting
        the next day's return is sufficient; we re-train/re-predict daily.

USAGE:
    predictor = LSTMPredictor(
        seq_len=20,       # use 20 days of history
        hidden_dim=64,    # LSTM hidden size
        num_layers=2,     # stacked LSTMs
    )

    predictor.fit(prices_df)  # DataFrame with 'close' column

    result = predictor.predict(prices_df)
    print(f"Next day return: {result.forecast[-1]:.4f}")

REFERENCES:
    Hochreiter, S. & Schmidhuber, J. (1997). Long Short-Term Memory.
        Neural Computation, 9(8).
    Bao, W., Yue, J., & Rao, Y. (2017). A Deep Learning Framework for
        Financial Time Series. PLoS ONE, 12(7).
    Fischer, T. & Krauss, C. (2018). Deep Learning with Long Short-Term
        Memory Networks for Financial Market Predictions. European Journal
        of Operational Research, 270(2).
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset


@dataclass
class LSTMResult:
    """
    Output of the LSTM predictor.

    Attributes:
        forecast: Array of predicted returns (same length as input,
            with NaN for sequence-start positions where predictions
            are not available).
        actuals: Array of actual returns (for comparison).
        loss_history: Training loss per epoch (for convergence check).
    """

    forecast: np.ndarray
    actuals: np.ndarray
    loss_history: list[float]


class _LSTMModel(nn.Module):
    """
    Internal PyTorch LSTM module.

    Architecture:
        LSTM → Dropout → Linear → output

    WHY a separate internal class:
        Separates the PyTorch model definition from the training/prediction
        logic. Makes the model hot-swappable (e.g., replace with GRU or
        Transformer) without changing the trainer interface.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor, shape (batch, seq_len, input_dim).

        Returns:
            Predicted return, shape (batch, 1).
        """
        lstm_out, _ = self.lstm(x)
        # Take the last time step's output
        last_out = lstm_out[:, -1, :]
        last_out = self.dropout(last_out)
        return self.linear(last_out)


class LSTMPredictor:
    """
    PyTorch LSTM for single-step return forecasting.

    Parameters:
        seq_len: Number of past time steps to use as input.
        hidden_dim: LSTM hidden state dimension.
        num_layers: Number of stacked LSTM layers.
        dropout: Dropout probability between LSTM layers.
        learning_rate: Adam optimiser learning rate.
        epochs: Number of training epochs.
        batch_size: Training batch size.
        device: "cpu" or "cuda" (auto-detected if None).
    """

    def __init__(
        self,
        seq_len: int = 20,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        epochs: int = 100,
        batch_size: int = 64,
        device: str | None = None,
    ) -> None:
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._model: _LSTMModel | None = None
        self._scaler_mean: float = 0.0
        self._scaler_std: float = 1.0

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def _prepare_data(
        self,
        prices: pd.Series,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert price series into supervised learning sequences.

        Creates input sequences of length seq_len and target returns.

        WHY normalisation:
            LSTM training with tanh/sigmoid activations requires inputs
            in a reasonable range (typically [-3, 3]). Log returns on
            daily data are ~0.01, which is fine without scaling, but
            standardising improves convergence speed.

        Args:
            prices: Series of closing prices.

        Returns:
            Tuple of (X, y) tensors for training.
        """
        # Log returns
        log_ret = np.log(prices / prices.shift(1)).dropna().values.reshape(-1, 1)

        # Standardise
        self._scaler_mean = float(np.mean(log_ret))
        self._scaler_std = float(np.std(log_ret))
        if self._scaler_std < 1e-12:
            self._scaler_std = 1.0
        scaled = (log_ret - self._scaler_mean) / self._scaler_std

        # Create sequences
        X, y = [], []
        for i in range(len(scaled) - self.seq_len):
            X.append(scaled[i : i + self.seq_len])
            y.append(scaled[i + self.seq_len])

        return (
            torch.tensor(np.array(X), dtype=torch.float32),
            torch.tensor(np.array(y), dtype=torch.float32),
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        prices: pd.Series | pd.DataFrame,
        verbose: bool = False,
    ) -> "LSTMPredictor":
        """
        Train the LSTM model on historical price data.

        Uses an 80/20 train/validation split based on time (not random).

        Args:
            prices: Series of closing prices with datetime index (or
                    DataFrame with 'close' column).
            verbose: If True, prints epoch progress.

        Returns:
            Self (fitted model).
        """
        if isinstance(prices, pd.DataFrame):
            prices = prices["close"] if "close" in prices.columns else prices.iloc[:, 0]

        X, y = self._prepare_data(prices)
        n = len(X)
        if n < 100:
            raise ValueError(f"Need at least 100 sequences, got {n}")

        # Walk-forward split (80/20 by time)
        split = int(n * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=self.batch_size,
            shuffle=False,  # don't shuffle time series
        )

        # Initialise model
        self._model = _LSTMModel(
            input_dim=1,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self._model.parameters(), lr=self.learning_rate)

        X_val = X_val.to(self.device)
        y_val = y_val.to(self.device)

        self._loss_history: list[float] = []

        for epoch in range(self.epochs):
            self._model.train()
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                output = self._model(batch_X)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()

            # Validation loss
            self._model.eval()
            with torch.no_grad():
                val_pred = self._model(X_val)
                val_loss = float(criterion(val_pred, y_val).item())
            self._loss_history.append(val_loss)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.epochs}, Val Loss: {val_loss:.6f}")

        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        prices: pd.Series | pd.DataFrame,
    ) -> LSTMResult:
        """
        Generate in-sample return forecasts.

        For each position i, predicts the return at i using the sequence
        of seq_len returns before i. The first seq_len positions have NaN
        forecasts (insufficient history).

        Args:
            prices: Series or DataFrame of closing prices.

        Returns:
            LSTMResult with forecast and actual return arrays.
        """
        if self._model is None:
            raise RuntimeError("Model must be fitted before predict()")

        if isinstance(prices, pd.DataFrame):
            prices = prices["close"] if "close" in prices.columns else prices.iloc[:, 0]

        # Log returns
        log_ret = np.log(prices / prices.shift(1)).dropna().values.reshape(-1, 1)
        scaled = (log_ret - self._scaler_mean) / self._scaler_std

        # Input sequences
        X = torch.tensor(
            np.array([scaled[i : i + self.seq_len] for i in range(len(scaled) - self.seq_len)]),
            dtype=torch.float32,
        ).to(self.device)

        self._model.eval()
        with torch.no_grad():
            pred_scaled = self._model(X).cpu().numpy().flatten()

        # Un-scale predictions
        pred_returns = pred_scaled * self._scaler_std + self._scaler_mean

        # Full-length arrays with NaN padding.
        # log_ret has len(prices)-1 valid entries (first is NaN from
        # shift(1)). log_ret[0] maps to prices position 1.
        # pred_returns[k] predicts log_ret[seq_len + k], which maps to
        # prices[seq_len + 1 + k].
        full_pred = np.full(len(prices), np.nan)
        full_pred[self.seq_len + 1:] = pred_returns

        full_actual = np.full(len(prices), np.nan)
        full_actual[self.seq_len + 1:] = log_ret[self.seq_len:].flatten()

        return LSTMResult(
            forecast=full_pred,
            actuals=full_actual,
            loss_history=self._loss_history,
        )
