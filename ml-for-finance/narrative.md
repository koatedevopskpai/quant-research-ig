# ml-for-finance — Machine Learning for Finance

## Objective
Apply ML to financial time series: regime detection, return forecasting,
sentiment-aware models, and deep learning for alpha generation — all trained
and evaluated on IG market data.

## Projects
1. **Regime Detection** ✅ — HMM (GaussianHMM via hmmlearn) for bull/bear/range classification
2. **Return Forecasting** ✅ — XGBoost on technical lags, rolling stats, RSI, MACD, calendar features
3. **LSTM Price Predictor** ✅ — Single-step PyTorch LSTM on log-return sequences
4. **Sentiment-Augmented Model** ✅ — Ensemble blending price and NLP sub-models weighted by R²
5. **Model Interpretability** ⏳ — SHAP / LIME (future work)

## Key Files
| File | Purpose |
|------|---------|
| src/ml_for_finance/regime/hmm_detector.py | Gaussian HMM with Viterbi decoding |
| src/ml_for_finance/forecasting/xgboost_forecaster.py | XGBoost with 15+ engineered features |
| src/ml_for_finance/deep_learning/lstm_predictor.py | Single-step PyTorch LSTM |
| src/ml_for_finance/sentiment/sentiment_model.py | Ensemble: price-only + sentiment, R²-weighted blend |
| tests/test_regime.py | 7 tests: fit, predict, score, labels, multivariate |
| tests/test_forecasting.py | 5 tests: fit, feature shapes, importance, edge cases |
| tests/test_lstm.py | 4 tests: fit/predict, loss decreases, edge cases |
| tests/test_sentiment.py | 3 tests: fit/predict, edge cases |

## Architecture Decisions (Phase 3a ADRs)

### ADR-7: HMM over GMM for Regime Detection
- **Context:** Both HMM and GMM can cluster market states; GMM is simpler.
- **Decision:** Use GaussianHMM (hmmlearn) because financial regimes exhibit
  temporal persistence (bull markets don't flip to bear overnight). HMM models
  the transition probability — a bear state is more likely to follow a bear
  state.
- **Consequence:** More parameters (transition matrix + emission means/covs)
  requires longer history (≥50 obs per regime). Labels sorted by variance
  (Low Vol → High Vol) for intuitive interpretation.

### ADR-8: XGBoost over LightGBM for Return Forecasting
- **Context:** Both are gradient-boosted tree frameworks.
- **Decision:** XGBoost for its smaller dependency surface, built-in
  feature importance, and robust pandas integration. LightGBM's histogram-based
  training provides little advantage at our data volume.
- **Consequence:** Slower training on large datasets (future: evaluate
  LightGBM if latency becomes an issue). Feature engineering lives in
  `_build_features()` — lags [1,2,3,5,10,21], rolling mean/std/skew/kurt,
  RSI, MACD, day-of-week/month dummies.

### ADR-9: Single-Step LSTM over Seq2Seq
- **Context:** Standard LSTM architectures for time series include
  many-to-many (seq2seq) and many-to-one (single-step).
- **Decision:** Many-to-one single-step with MSE loss. For a trading
  application, the next-day return is the atomic forecast unit; multi-step
  forecasts compound error and add complexity without actionable benefit.
- **Consequence:** The predict() method slides a seq_len window to produce
  one forecast per day. A separate forecasting horizon extension would be
  needed for swing/positional strategies.

### ADR-10: Ensemble Blending over Stacking for Sentiment
- **Context:** Combining price-based and sentiment-based forecasts can
  follow stacking (meta-model) or blending (weighted average).
- **Decision:** R²-weighted blending. On small financial datasets,
  a meta-model risks overfitting; ex-post R² weights are transparent,
  interpretable, and monotonic in sub-model skill.
- **Consequence:** The blend weight is computed from the most recent
  `eval_window` (default 60) trading days. A sentiment sub-model with
  persistently negative R² is down-weighted toward zero.

## Status
- [x] Regime Detection — HMM with Viterbi, 7 tests, all passing
- [x] Return Forecasting — XGBoost with feature engineering, 5 tests
- [x] LSTM Predictor — PyTorch single-step, 4 tests
- [x] Sentiment-Augmented Model — R²-weighted ensemble, 3 tests
- [ ] Model Interpretability (future phase)
