"""Detector training with all three split protocols side by side."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from .dataio import SCENARIOS
from .splits import SPLITTERS, leakage_report


class SplitCoverageError(RuntimeError):
    """A split left the training set without a class the task requires."""

@dataclass
class DetectorConfig:
    num_leaves: int = 63
    learning_rate: float = 0.05
    n_estimators: int = 400
    feature_fraction: float = 0.9
    bagging_fraction: float = 0.8
    bagging_freq: int = 5
    min_child_samples: int = 50
    n_jobs: int = -1
    seed: int = 0


def _fit(X_tr, y_tr, X_va, y_va, cfg, objective, n_class=None):
    """Fit with early stopping, guarding against class-coverage mismatch.

    LightGBM fits a LabelEncoder on y_train and then transforms y_valid, so a
    class present in validation but absent from training raises
    "y contains previously unseen labels" from deep inside sklearn -- a message
    that says nothing about the actual cause. That cause is always the split:
    contiguous-block splitting on a short recording can strand every attacked
    block outside the training set.
    """
    import lightgbm as lgb

    tr_classes = set(np.unique(y_tr).tolist())
    if len(tr_classes) < 2:
        raise SplitCoverageError(
            f"training split contains only class {sorted(tr_classes)}; "
            f"this protocol cannot be evaluated on this data"
        )
    uncovered = set(np.unique(y_va).tolist()) - tr_classes
    if uncovered:
        keep = ~np.isin(y_va, list(uncovered))
        print(f"    note: validation classes {sorted(uncovered)} absent from "
              f"training; dropping {int((~keep).sum())} validation row(s)")
        X_va, y_va = X_va[keep], y_va[keep]
        if len(y_va) == 0 or len(set(np.unique(y_va).tolist())) < 2:
            raise SplitCoverageError(
                "validation split empty or single-class after removing "
                "classes unseen in training"
            )

    params = dict(
        num_leaves=cfg.num_leaves, learning_rate=cfg.learning_rate,
        n_estimators=cfg.n_estimators, colsample_bytree=cfg.feature_fraction,
        subsample=cfg.bagging_fraction, subsample_freq=cfg.bagging_freq,
        min_child_samples=cfg.min_child_samples, n_jobs=cfg.n_jobs,
        random_state=cfg.seed, verbose=-1,
    )
    if objective == "multiclass":
        model = lgb.LGBMClassifier(objective="multiclass", num_class=n_class, **params)
    else:
        model = lgb.LGBMClassifier(objective="binary", **params)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
              callbacks=[lgb.early_stopping(50, verbose=False)])
    return model


def evaluate_protocols(df, feature_cols, cfg=None,
                       protocols=("random", "block", "scenario"), repeats=1):
    """Run each protocol `repeats` times with different split seeds."""
    if repeats > 1:
        cfg = cfg or DetectorConfig()
        runs = []
        for r in range(repeats):
            c = DetectorConfig(**{**asdict(cfg), "seed": cfg.seed + r})
            res, _ = _evaluate_once(df, feature_cols, c, protocols)
            runs.append(res)
        return _aggregate_repeats(runs), {}
    return _evaluate_once(df, feature_cols, cfg, protocols)


def _aggregate_repeats(runs):
    out = {}
    for proto in runs[0]:
        if "error" in runs[0][proto]:
            out[proto] = runs[0][proto]
            continue
        entry = {"leakage": runs[0][proto]["leakage"], "n_repeats": len(runs)}
        for task in ("binary", "multiclass"):
            vals = [r[proto][task]["macro_f1"] for r in runs]
            per_class = {}
            for k in runs[0][proto][task]["per_class_f1"]:
                v = [r[proto][task]["per_class_f1"][k] for r in runs]
                per_class[k] = float(np.mean(v))
                per_class[k + "_std"] = float(np.std(v))
            entry[task] = {
                "macro_f1": float(np.mean(vals)),
                "macro_f1_std": float(np.std(vals)),
                "accuracy": float(np.mean([r[proto][task]["accuracy"] for r in runs])),
                "per_class_f1": per_class,
            }
        out[proto] = entry
    return out


def _evaluate_once(df, feature_cols, cfg=None, protocols=("random", "block", "scenario")):
    from sklearn.metrics import f1_score, accuracy_score, classification_report

    cfg = cfg or DetectorConfig()
    results, models = {}, {}

    for proto in protocols:
        try:
            splits = SPLITTERS[proto](df, seed=cfg.seed)
        except Exception as e:
            results[proto] = {"error": str(e)}
            continue

        Xs = {k: df.loc[v, feature_cols].to_numpy(np.float32) for k, v in splits.items()}
        entry = {"leakage": leakage_report(df, splits, feature_cols)}

        for task, col in (("binary", "y_bin"), ("multiclass", "y_multi")):
            y = {k: df.loc[v, col].to_numpy() for k, v in splits.items()}
            n_class = int(df[col].max()) + 1
            try:
                m = _fit(Xs["train"], y["train"], Xs["val"], y["val"], cfg,
                         "multiclass" if task == "multiclass" else "binary",
                         n_class if task == "multiclass" else None)
            except SplitCoverageError as e:
                print(f"  {proto}/{task}: SKIPPED -- {e}")
                entry[task] = {"error": str(e), "accuracy": float("nan"),
                               "macro_f1": float("nan"), "per_class_f1": {}}
                continue
            pred = m.predict(Xs["test"])
            per_class = f1_score(y["test"], pred, average=None,
                                 labels=list(range(n_class)), zero_division=0)
            entry[task] = {
                "accuracy": float(accuracy_score(y["test"], pred)),
                "macro_f1": float(f1_score(y["test"], pred, average="macro", zero_division=0)),
                "per_class_f1": {
                    (SCENARIOS[i] if task == "multiclass" and i < len(SCENARIOS) else str(i)):
                    float(v) for i, v in enumerate(per_class)
                },
                "report": classification_report(y["test"], pred, zero_division=0,
                                                output_dict=True),
            }
            models[(proto, task)] = m
        results[proto] = entry
    return results, models


def summarise(results) -> pd.DataFrame:
    rows = []
    for proto, r in results.items():
        if "error" in r:
            rows.append({"protocol": proto, "note": r["error"]})
            continue
        if "error" in r.get("binary", {}) or "error" in r.get("multiclass", {}):
            rows.append({"protocol": proto,
                         "note": r.get("binary", {}).get("error")
                                 or r["multiclass"]["error"]})
            continue
        rows.append({
            "protocol": proto,
            "n_repeats": r.get("n_repeats", 1),
            "binary_macro_f1": round(r["binary"]["macro_f1"], 4),
            "binary_std": round(r["binary"].get("macro_f1_std", 0.0), 4),
            "multi_macro_f1": round(r["multiclass"]["macro_f1"], 4),
            "multi_std": round(r["multiclass"].get("macro_f1_std", 0.0), 4),
            "stealth_f1": round(r["multiclass"]["per_class_f1"].get("Stealth", float("nan")), 4),
            "slowramp_f1": round(r["multiclass"]["per_class_f1"].get("SlowRamp", float("nan")), 4),
            "nn_dist_per_dim": round(r["leakage"].get("median_nn_distance_per_dim", float("nan")), 5),
        })
    return pd.DataFrame(rows)


def save(results, path):
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=float)
