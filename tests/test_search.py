import inspect

import pandas as pd
import pytest

from fraud_xgb.model import PARAM_NAMES, sample_configs, search, select_best


def _results(test_pr_auc_values) -> pd.DataFrame:
    configs = sample_configs(4, 42)
    rows = []
    val_values = [0.30, 0.55, 0.40, 0.55]
    for trial, (config, val, test_value) in enumerate(zip(configs, val_values, test_pr_auc_values)):
        rows.append(
            {"trial": trial, **config, "val_pr_auc": val, "test_pr_auc": test_value}
        )
    return pd.DataFrame(rows)


def test_select_best_picks_highest_validation_pr_auc():
    results = _results([0.0, 0.0, 0.0, 0.0])
    assert select_best(results) == results.loc[1, PARAM_NAMES].to_dict()


def test_select_best_ignores_test_column():
    # Garbage test scores that favour trial 0 (validation loser) must not change the choice.
    expected = _results([0.0, 0.0, 0.0, 0.0]).loc[1, PARAM_NAMES].to_dict()
    assert select_best(_results([0.99, 0.0, 0.0, 0.0])) == expected
    assert select_best(_results([0.0, 0.99, 0.5, 0.2])) == expected


def test_select_best_breaks_ties_by_earliest_trial():
    results = _results([0.0, 0.0, 0.0, 0.0])
    results.loc[3, "val_pr_auc"] = 0.55
    best = select_best(results)
    assert best == results.loc[1, PARAM_NAMES].to_dict()


def test_select_best_requires_validation_column():
    results = _results([0.0, 0.0, 0.0, 0.0]).drop(columns=["val_pr_auc"])
    with pytest.raises(ValueError):
        select_best(results)


def test_search_takes_no_test_inputs():
    names = inspect.signature(search).parameters
    assert not any("test" in name for name in names)


def test_search_rejects_validation_without_frauds():
    X = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
    with pytest.raises(ValueError):
        search(X, [0, 1, 0, 1], X, [0, 0, 0, 0], n_iter=1)
