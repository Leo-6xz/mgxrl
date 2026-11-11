"""Windowed feature extraction at the secondary-control timescale.

Pointwise features (one row = one instant) are what the published detector
uses. They work at 2 us because the classifier can memorise absolute drift
offsets. At a 10 ms control rate on held-out runs they degrade badly, because
Ramp/SlowRamp/Stealth are defined by *rates and correlations over time*, not by
instantaneous values.

The windowed features below are the fix, and they also make the detector's
output physically interpretable -- which matters downstream, since the SHAP
attributions the LLM verbalises are computed over these names.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _slope(a: np.ndarray) -> np.ndarray:
    """Least-squares slope along axis 0 of a (window, features) array."""
    n = a.shape[0]
    x = np.arange(n) - (n - 1) / 2.0
    return (x[:, None] * a).sum(0) / (x @ x)


def window_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    window: int = 20,
    stride: int = 5,
    group_cols=("scenario", "run_id"),
) -> pd.DataFrame:
    """Rolling-window aggregation. Windows never straddle runs."""
    out_rows, meta_rows = [], []
    for keys, g in df.groupby(list(group_cols), sort=False):
        arr = g[feature_cols].to_numpy(np.float32)
        if len(arr) < window:
            continue
        y_bin = g["y_bin"].to_numpy() if "y_bin" in g else np.zeros(len(g), int)
        y_mul = g["y_multi"].to_numpy() if "y_multi" in g else np.zeros(len(g), int)
        for s in range(0, len(arr) - window + 1, stride):
            w = arr[s:s + window]
            out_rows.append(np.concatenate([
                w.mean(0), w.std(0), _slope(w),
                w.max(0) - w.min(0),
                np.abs(np.diff(w, axis=0)).mean(0),
            ]))
            meta_rows.append({
                # A window is "attacked" if any sample in it is attacked.
                "y_bin": int(y_bin[s:s + window].max()),
                "y_multi": int(np.bincount(y_mul[s:s + window]).argmax()),
                "scenario": keys[0] if isinstance(keys, tuple) else keys,
                "run_id": keys[1] if isinstance(keys, tuple) and len(keys) > 1 else 0,
                "window_start": int(s),
            })

    stats = ["mean", "std", "slope", "range", "meanabsdiff"]
    names = [f"{c}__{s}" for s in stats for c in feature_cols]
    X = pd.DataFrame(np.asarray(out_rows), columns=names)
    return pd.concat([X, pd.DataFrame(meta_rows)], axis=1)


def physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derived quantities with direct physical meaning.

    These exist so the explanation layer has something an operator recognises.
    "q_sharing_dispersion rose by 0.04" is actionable; "feature_143 had SHAP
    value 0.02" is not.
    """
    out = pd.DataFrame(index=df.index)
    for tag, label in (("p_dg", "p"), ("q_dg", "q"), ("f_dg", "f")):
        cols = [c for c in df.columns if c.startswith(tag)]
        if not cols:
            continue
        a = df[cols].to_numpy(np.float32)
        out[f"{label}_mean"] = a.mean(1)
        out[f"{label}_dispersion"] = a.std(1)          # power-sharing quality
        out[f"{label}_maxdev"] = np.abs(a - a.mean(1, keepdims=True)).max(1)
    fcols = [c for c in df.columns if c.startswith("f_dg")]
    if fcols:
        out["f_error"] = df[fcols].mean(1) - 50.0      # secondary restoration error
    vcols = [c for c in df.columns if c.lower().startswith("v") and df[c].dtype.kind == "f"]
    if vcols:
        v = df[vcols].to_numpy(np.float32)
        out["v_mean"] = v.mean(1)
        out["v_min"] = v.min(1)
        out["v_spread"] = v.max(1) - v.min(1)
    return out
