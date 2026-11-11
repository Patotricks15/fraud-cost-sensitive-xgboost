"""Score export for FD05: validation and test rows scored by the tuned model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data import AMOUNT, TARGET, TIME, TimeSplit
from .model import FEATURES

EXPORT_COLUMNS = ["split", TIME, AMOUNT, TARGET, "score"]


def build_score_export(
    df: pd.DataFrame,
    split: TimeSplit,
    model,
    features: list[str] = FEATURES,
) -> pd.DataFrame:
    """Score validation and test rows. Training rows are never exported."""
    required = {TIME, AMOUNT, TARGET, *features}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"frame is missing columns: {sorted(missing)}")
    parts = []
    for name, mask in (("val", split.val), ("test", split.test)):
        rows = df.loc[mask.to_numpy()]
        part = rows[[TIME, AMOUNT, TARGET]].copy()
        part["score"] = model.predict_proba(rows[features])[:, 1]
        part.insert(0, "split", name)
        parts.append(part)
    out = pd.concat(parts, ignore_index=True)
    if not np.isfinite(out["score"]).all() or not out["score"].between(0, 1).all():
        raise ValueError("scores must be finite probabilities in [0, 1]")
    return out[EXPORT_COLUMNS]


def write_score_export(frame: pd.DataFrame, path: str | Path) -> Path:
    """Write the export CSV after checking its exact column layout."""
    if list(frame.columns) != EXPORT_COLUMNS:
        raise ValueError(f"export must have columns {EXPORT_COLUMNS}")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    return out
