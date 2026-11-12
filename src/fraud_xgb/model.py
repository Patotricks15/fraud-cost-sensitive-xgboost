"""Full-tree boosting baseline, random search on validation, and gain importance."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.model_selection import ParameterSampler
from xgboost import XGBClassifier

from .data import AMOUNT, PCA_FEATURES
from .metrics import pr_auc

RANDOM_STATE = 42
N_SEARCH_CONFIGS = 30
FEATURES = [*PCA_FEATURES, "log_amount"]

BASELINE_PARAMS = {
    "max_depth": 6,
    "learning_rate": 0.1,
    "n_estimators": 300,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "min_child_weight": 1,
    "reg_lambda": 1.0,
}

SEARCH_SPACE = {
    "max_depth": randint(3, 9),
    "learning_rate": loguniform(0.01, 0.3),
    "n_estimators": randint(100, 601),
    "subsample": uniform(0.6, 0.4),
    "colsample_bytree": uniform(0.6, 0.4),
    "min_child_weight": randint(1, 11),
    "reg_lambda": loguniform(0.1, 10.0),
}
PARAM_NAMES = list(SEARCH_SPACE)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["log_amount"] = np.log1p(out[AMOUNT])
    return out


def _python_scalar(value):
    return value.item() if hasattr(value, "item") else value


def scale_pos_weight(y_train) -> float:
    """Negative-to-positive ratio. Pass training labels only."""
    y = np.asarray(y_train)
    if not np.isin(y, [0, 1]).all():
        raise ValueError("labels must be 0 or 1")
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("training labels must contain both classes")
    return n_neg / n_pos


def build_xgb(params: dict, spw: float) -> XGBClassifier:
    return XGBClassifier(
        **params,
        scale_pos_weight=spw,
        objective="binary:logistic",
        tree_method="hist",
        eval_metric="aucpr",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def build_baseline(spw: float) -> XGBClassifier:
    return build_xgb(BASELINE_PARAMS, spw)


def build_tuned(params: dict, spw: float) -> XGBClassifier:
    return build_xgb(params, spw)


def sample_configs(n_iter: int = N_SEARCH_CONFIGS, random_state: int = RANDOM_STATE) -> list[dict]:
    """Reproducible random configurations drawn from SEARCH_SPACE."""
    if n_iter < 1:
        raise ValueError("n_iter must be at least 1")
    sampler = ParameterSampler(SEARCH_SPACE, n_iter=n_iter, random_state=random_state)
    return [{k: _python_scalar(v) for k, v in params.items()} for params in sampler]


def search(
    X_train: pd.DataFrame,
    y_train,
    X_val: pd.DataFrame,
    y_val,
    n_iter: int = N_SEARCH_CONFIGS,
    random_state: int = RANDOM_STATE,
) -> tuple[dict, pd.DataFrame]:
    """Fit each sampled config on train, score validation PR-AUC, return the best config.

    Only training and validation data are accepted, so the test window cannot be used.
    """
    y_val_arr = np.asarray(y_val)
    if y_val_arr.sum() == 0:
        raise ValueError("validation labels contain no frauds; PR-AUC is undefined")
    spw = scale_pos_weight(y_train)
    rows = []
    for trial, params in enumerate(sample_configs(n_iter, random_state)):
        model = build_xgb(params, spw).fit(X_train, y_train)
        scores = model.predict_proba(X_val)[:, 1]
        rows.append({"trial": trial, **params, "val_pr_auc": pr_auc(y_val_arr, scores)})
    results = pd.DataFrame(rows)
    return select_best(results), results


def select_best(results: pd.DataFrame) -> dict:
    """Return the parameters with the highest validation PR-AUC. Ties go to the earliest trial."""
    missing = {"val_pr_auc", *PARAM_NAMES} - set(results.columns)
    if missing:
        raise ValueError(f"search results are missing columns: {sorted(missing)}")
    # Select as a one-row frame so per-column dtypes survive; a row Series would upcast ints to float.
    best = results.loc[[results["val_pr_auc"].idxmax()]].to_dict("records")[0]
    return {name: _python_scalar(best[name]) for name in PARAM_NAMES}


def gain_importance(model: XGBClassifier, feature_names: list[str] = FEATURES) -> pd.DataFrame:
    """Total gain per feature from the fitted booster, sorted by gain."""
    raw = model.get_booster().get_score(importance_type="gain")
    gains: dict[str, float] = {}
    for key, value in raw.items():
        if key in feature_names:
            name = key
        elif re.fullmatch(r"f\d+", key) and int(key[1:]) < len(feature_names):
            name = feature_names[int(key[1:])]
        else:
            raise ValueError(f"unexpected feature key in booster: {key}")
        gains[name] = float(value)
    table = pd.DataFrame(
        {"feature": feature_names, "gain": [gains.get(name, 0.0) for name in feature_names]}
    )
    total = table["gain"].sum()
    table["gain_share"] = table["gain"] / total if total > 0 else 0.0
    return table.sort_values("gain", ascending=False, kind="mergesort").reset_index(drop=True)
