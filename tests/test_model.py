import numpy as np
import pandas as pd
import pytest

from fraud_xgb.data import TIME, time_split
from fraud_xgb.model import (
    BASELINE_PARAMS,
    FEATURES,
    SEARCH_SPACE,
    add_features,
    build_baseline,
    build_tuned,
    gain_importance,
    sample_configs,
    scale_pos_weight,
)


def _labelled_frame() -> pd.DataFrame:
    # Train window (Time <= 6.3) holds 6 legitimate and 1 fraud rows: ratio 6.
    # The full frame holds 6 legitimate and 4 frauds: ratio 1.5.
    return pd.DataFrame({TIME: np.arange(10.0), "Class": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]})


def test_scale_pos_weight_uses_train_rows_only():
    df = _labelled_frame()
    split = time_split(df)
    spw = scale_pos_weight(df.loc[split.train, "Class"])
    assert spw == pytest.approx(6.0)
    assert spw != pytest.approx(1.5)
    assert build_baseline(spw).get_params()["scale_pos_weight"] == pytest.approx(6.0)


def test_scale_pos_weight_rejects_single_class_or_bad_labels():
    with pytest.raises(ValueError):
        scale_pos_weight([0, 0, 0])
    with pytest.raises(ValueError):
        scale_pos_weight([0, 1, 2])


def test_baseline_and_tuned_share_fixed_settings():
    tuned = build_tuned({**BASELINE_PARAMS, "max_depth": 4}, 2.0).get_params()
    base = build_baseline(2.0).get_params()
    for model in (base, tuned):
        assert model["tree_method"] == "hist"
        assert model["eval_metric"] == "aucpr"
        assert model["random_state"] == 42
        assert model["n_jobs"] == -1
    assert tuned["max_depth"] == 4


def test_sample_configs_is_reproducible_and_within_bounds():
    first = sample_configs(30, 42)
    assert first == sample_configs(30, 42)
    assert len(first) == 30
    for config in first:
        assert set(config) == set(SEARCH_SPACE)
        assert 3 <= config["max_depth"] < 9
        assert 0.01 <= config["learning_rate"] <= 0.31
        assert 0.6 <= config["subsample"] <= 1.0
        assert 1 <= config["min_child_weight"] < 11


def test_add_features_adds_log_amount_without_changing_input():
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"Amount": rng.uniform(1, 500, 10)})
    out = add_features(df)
    assert "log_amount" not in df.columns
    np.testing.assert_allclose(out["log_amount"], np.log1p(df["Amount"]))
    assert FEATURES[-1] == "log_amount" and len(FEATURES) == 29


class _StubBooster:
    def get_score(self, importance_type):
        assert importance_type == "gain"
        return {"f0": 3.0, "f2": 1.0}


class _StubModel:
    def get_booster(self):
        return _StubBooster()


def test_gain_importance_maps_keys_and_shares():
    table = gain_importance(_StubModel(), feature_names=["a", "b", "c"])
    assert list(table["feature"]) == ["a", "c", "b"]
    np.testing.assert_allclose(table["gain"], [3.0, 1.0, 0.0])
    np.testing.assert_allclose(table["gain_share"], [0.75, 0.25, 0.0])
