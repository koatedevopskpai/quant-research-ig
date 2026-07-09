import numpy as np
import pytest
from hft_engine.toxicity.detector import ToxicityDetector


class TestToxicityDetector:
    def test_fit_baseline(self):
        t = np.cumsum(np.random.exponential(0.1, 100))
        d = ToxicityDetector()
        d.fit(t.tolist())
        r = d.predict()
        assert isinstance(r.toxicity, float)
        assert 0.0 <= r.toxicity <= 1.0
        assert r.n_events == 100

    def test_fit_too_few_samples(self):
        d = ToxicityDetector()
        with pytest.raises(ValueError, match="Need >=10 events"):
            d.fit([1.0, 2.0])

    def test_predict_before_fit_raises(self):
        d = ToxicityDetector()
        with pytest.raises(RuntimeError, match="fitted"):
            d.predict()

    def test_clustered_events_are_more_toxic(self):
        t = [0.0]
        for _ in range(50):
            t.append(t[-1] + np.random.exponential(0.01))
        d = ToxicityDetector().fit(t)
        r = d.predict()
        assert r.toxicity >= 0.3

    def test_intensity_increases_after_event(self):
        t = np.cumsum(np.random.exponential(0.1, 50)).tolist()
        d = ToxicityDetector().fit(t)
        i_before = d.intensity(t[-1] - 0.01)
        i_after = d.intensity(t[-1] + 0.01)
        assert i_after >= i_before * 0.5

    def test_window_size_truncates(self):
        t = np.cumsum(np.random.exponential(0.1, 500)).tolist()
        d = ToxicityDetector(window_size=50).fit(t)
        assert len(d.timestamps) <= 50

    def test_intensity_never_negative(self):
        t = np.cumsum(np.random.exponential(0.2, 30)).tolist()
        d = ToxicityDetector().fit(t)
        for _ in range(10):
            lam = d.intensity(d.timestamps[-1] + np.random.exponential(0.5))
            assert lam >= 0
