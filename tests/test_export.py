import numpy as np
import pandas as pd
import pytest

from fraud_xgb.data import AMOUNT, PCA_FEATURES, TARGET, TIME, time_split
from fraud_xgb.export import EXPORT_COLUMNS, build_score_export, write_score_export
from fraud_xgb.model import FEATURES, add_features


class _SigmoidV1Model:
    """Deterministic stand-in: fraud probability is sigmoid(V1)."""

    def predict_proba(self, X):
        p = 1.0 / (1.0 + np.exp(-X["V1"].to_numpy()))
        return np.column_stack([1 - p, p])


def _frame(n: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    df = pd.DataFrame(rng.normal(size=(n, 28)), columns=PCA_FEATURES)
    df[TIME] = np.arange(n, dtype=float)
    df[AMOUNT] = rng.uniform(1, 100, n)
    df[TARGET] = (np.arange(n) % 5 == 0).astype(int)
    return add_features(df)


def test_export_has_exact_columns_and_only_val_and_test_rows():
    df = _frame()
    split = time_split(df)
    out = build_score_export(df, split, _SigmoidV1Model())
    assert list(out.columns) == ["split", "Time", "Amount", "Class", "score"]
    assert set(out["split"]) == {"val", "test"}
    assert len(out) == int(split.val.sum() + split.test.sum())
    assert (out["Time"] > split.train_end).all()


def test_export_scores_match_model_on_val_and_test_rows():
    df = _frame()
    split = time_split(df)
    out = build_score_export(df, split, _SigmoidV1Model())
    expected = 1.0 / (1.0 + np.exp(-df.loc[split.val.to_numpy() | split.test.to_numpy(), "V1"].to_numpy()))
    np.testing.assert_allclose(out["score"].to_numpy(), expected)


def test_export_requires_feature_columns():
    df = _frame().drop(columns=["V5"])
    split = time_split(_frame())
    with pytest.raises(ValueError):
        build_score_export(df, split, _SigmoidV1Model(), features=FEATURES)


def test_write_export_rejects_wrong_layout():
    bad = pd.DataFrame({"split": ["val"], "score": [0.5]})
    with pytest.raises(ValueError):
        write_score_export(bad, "unused.csv")
    assert EXPORT_COLUMNS == ["split", "Time", "Amount", "Class", "score"]
