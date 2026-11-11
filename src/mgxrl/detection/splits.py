"""Leakage-corrected train/val/test splitting."""

from __future__ import annotations

import numpy as np
import pandas as pd


def random_split(df, test_size=0.15, val_size=0.15, seed=0):
    """Reproduces the published protocol. For comparison only."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    n_test, n_val = int(len(df) * test_size), int(len(df) * val_size)
    return {
        "test": df.index[idx[:n_test]],
        "val": df.index[idx[n_test:n_test + n_val]],
        "train": df.index[idx[n_test + n_val:]],
    }


def block_split(df, n_blocks=40, test_size=0.15, val_size=0.15,
                purge_blocks=1, seed=0, group_cols=("scenario", "run_id")):
    """Contiguous-block split with purging."""
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["_block"] = -1

    bid = 0
    for _, g in df.groupby(list(group_cols), sort=False):
        edges = np.linspace(0, len(g), n_blocks + 1).astype(int)
        for b in range(n_blocks):
            df.loc[g.index[edges[b]:edges[b + 1]], "_block"] = bid
            bid += 1

    blocks = np.array(sorted(df["_block"].unique()))
    perm = rng.permutation(blocks)
    n_test, n_val = int(len(blocks) * test_size), int(len(blocks) * val_size)
    test_b = set(perm[:n_test].tolist())
    val_b = set(perm[n_test:n_test + n_val].tolist())

    purged = set()
    for b in test_b | val_b:
        for d in range(1, purge_blocks + 1):
            purged.update({b - d, b + d})
    train_b = set(blocks.tolist()) - test_b - val_b - purged

    sel = lambda s: df.index[df["_block"].isin(s)]
    return {"train": sel(train_b), "val": sel(val_b), "test": sel(test_b)}


def scenario_split(df, test_frac=0.2, val_frac=0.2, seed=0,
                   test_runs=None, val_runs=None):
    """Hold out whole simulation runs. The strictest and most honest protocol."""
    rng = np.random.default_rng(seed)
    parts = {"train": [], "val": [], "test": []}
    for scen, g in df.groupby("scenario", sort=False):
        runs = np.array(sorted(g["run_id"].unique()))

        n_te = test_runs if test_runs is not None else max(1, int(round(len(runs) * test_frac)))
        n_va = val_runs if val_runs is not None else max(1, int(round(len(runs) * val_frac)))

        if len(runs) < n_te + n_va + 1:
            raise ValueError(
                f"Scenario '{scen}' has {len(runs)} run(s); scenario-level "
                f"splitting needs >= {n_te + n_va + 1}. Use block_split "
                f"and state the limitation."
            )
        perm = rng.permutation(runs)
        assign = {"test": perm[:n_te], "val": perm[n_te:n_te + n_va],
                  "train": perm[n_te + n_va:]}
        for k, rs in assign.items():
            parts[k].append(g.index[g["run_id"].isin(rs)])
    return {k: np.concatenate(v) for k, v in parts.items()}


SPLITTERS = {"random": random_split, "block": block_split, "scenario": scenario_split}


def leakage_report(df, splits, feature_cols, n_probe=2000, seed=0):
    """Nearest-neighbour probe: how close is the average test row to its nearest
    training row? A near-zero distance means the split leaked."""
    from sklearn.neighbors import NearestNeighbors

    rng = np.random.default_rng(seed)
    tr = df.loc[splits["train"], feature_cols].to_numpy(np.float64)
    te = df.loc[splits["test"], feature_cols].to_numpy(np.float64)
    if len(tr) == 0 or len(te) == 0:
        return {}

    # Keep only columns that are finite everywhere, then rows that are finite.
    # Doing this here rather than upstream keeps the probe usable on raw
    # exports without silently imputing values into a security dataset.
    good_col = np.isfinite(tr).all(0) & np.isfinite(te).all(0)
    n_dropped = int((~good_col).sum())
    tr, te = tr[:, good_col], te[:, good_col]
    tr = tr[np.isfinite(tr).all(1)]
    te = te[np.isfinite(te).all(1)]
    if len(tr) == 0 or len(te) == 0 or tr.shape[1] == 0:
        return {"error": "no finite data for the leakage probe"}

    mu, sd = tr.mean(0), tr.std(0) + 1e-9
    tr, te = (tr - mu) / sd, (te - mu) / sd
    tr_s = tr[rng.choice(len(tr), min(n_probe * 5, len(tr)), replace=False)]
    te_s = te[rng.choice(len(te), min(n_probe, len(te)), replace=False)]
    d, _ = NearestNeighbors(n_neighbors=1).fit(tr_s).kneighbors(te_s)
    # Normalize by sqrt(dimension). Raw nearest-neighbor distances grow with

    dim = tr.shape[1]
    return {
        "median_nn_distance": float(np.median(d)),
        "median_nn_distance_per_dim": float(np.median(d) / np.sqrt(dim)),
        "frac_below_0.01": float((d < 0.01).mean()),
        "n_train": int(len(tr)), "n_test": int(len(te)),
        "n_features": int(dim), "n_features_dropped_nonfinite": n_dropped,
    }
