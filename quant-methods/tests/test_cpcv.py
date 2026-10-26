import numpy as np
import pytest
from quant_methods.validation.cpcv import CombinatorialPurgedCV


class MeanModel:
    def fit(self, x, y):
        self.mean = y.mean()
        return self

    def predict(self, x):
        return np.full(len(x), self.mean)


def strategy_returns(y, predictions):
    return np.sign(predictions) * y


def test_folds_fit_only_retained_training_rows():
    x = np.arange(60).reshape(-1, 1)
    y = np.linspace(-0.02, 0.03, 60)
    instances = []

    class RecordingModel(MeanModel):
        def fit(self, x, y):
            self.seen = x[:, 0].astype(int)
            return super().fit(x, y)

    def factory():
        model = RecordingModel()
        instances.append(model)
        return model

    ends = np.minimum(np.arange(60) + 3, 59)
    cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, embargo=2)
    result = cv.run(x, y, factory, strategy_returns, label_end=ends)
    assert len(instances) == result.n_combinations == 15
    assert len({id(m) for m in instances}) == 15
    for model, fold in zip(instances, result.fold_results, strict=True):
        np.testing.assert_array_equal(model.seen, fold["train_indices"])
        train, test = fold["train_indices"], fold["test_indices"]
        for i in train:
            assert all(ends[i] < j or i > ends[j] for j in test)
        expected = np.sign(y[train].mean()) * y[test]
        np.testing.assert_allclose(fold["oos_returns"], expected)
    assert result.is_significant is False  # overlapping folds are not IID tests


def test_label_overlap_and_embargo_have_exact_expected_indices():
    ends = np.minimum(np.arange(12) + 2, 11)
    cv = CombinatorialPurgedCV(n_groups=3, n_test_groups=1, embargo=1)
    # Middle test group spans starts 4..7 and label outcomes through 9.
    train, test = list(cv.split(12, label_end=ends))[1]
    np.testing.assert_array_equal(test, [4, 5, 6, 7])
    np.testing.assert_array_equal(train, [0, 1, 11])


def test_embargo_changes_the_data_used_to_fit_models():
    x = np.arange(60)[:, None]
    y = np.sin(np.arange(60)) * 0.01
    a = CombinatorialPurgedCV(6, 1, embargo=0).run(x, y, MeanModel, strategy_returns)
    b = CombinatorialPurgedCV(6, 1, embargo=4).run(x, y, MeanModel, strategy_returns)
    assert len(a.fold_results[0]["train_indices"]) - len(b.fold_results[0]["train_indices"]) == 4
    assert a.fold_results[0]["predictions"][0] != b.fold_results[0]["predictions"][0]


def test_scaling_pipeline_is_fitted_on_each_training_fold():
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    x = np.arange(40, dtype=float)[:, None]
    y = x[:, 0] / 100
    models = []

    def factory():
        model = make_pipeline(StandardScaler(), LinearRegression())
        models.append(model)
        return model

    result = CombinatorialPurgedCV(4, 1).run(x, y, factory, strategy_returns)
    for model, fold in zip(models, result.fold_results, strict=True):
        np.testing.assert_allclose(model[0].mean_, x[fold["train_indices"]].mean(axis=0))


def test_combination_cap_is_reproducible():
    cv = CombinatorialPurgedCV(20, 4, max_combinations=7)
    first, second = list(cv.split(200)), list(cv.split(200))
    assert len(first) == len(second) == 7
    for (_, a), (_, b) in zip(first, second, strict=True):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize(
    "kwargs", [dict(n_groups=5, n_test_groups=5), dict(embargo=-1), dict(max_combinations=0)]
)
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        CombinatorialPurgedCV(**kwargs)


def test_invalid_data_and_labels_rejected():
    cv = CombinatorialPurgedCV(5, 2)
    with pytest.raises(ValueError, match="Need >= 10"):
        list(cv.split(3))
    with pytest.raises(ValueError, match="label_end"):
        list(cv.split(20, label_end=np.zeros(20, dtype=float)))
    with pytest.raises(ValueError, match="label_end"):
        list(cv.split(20, label_end=np.zeros(20, dtype=int)))
    with pytest.raises(ValueError, match="aligned"):
        cv.run(np.zeros((20, 1)), np.zeros(19), MeanModel, strategy_returns)


def test_wrong_strategy_output_shape_rejected():
    with pytest.raises(ValueError, match="one finite strategy return"):
        CombinatorialPurgedCV(4, 1).run(np.arange(40), np.arange(40), MeanModel, lambda y, p: 0.01)


def test_quantile_level_is_used_without_claiming_significance():
    x = np.arange(60)
    y = np.linspace(-0.02, 0.03, 60)
    cv = CombinatorialPurgedCV(6, 1)
    result = cv.run(x, y, MeanModel, strategy_returns, significance_level=0.2)
    assert result.lower_quantile == pytest.approx(np.quantile(result.all_sharpes, 0.2))
    assert result.is_significant is False
