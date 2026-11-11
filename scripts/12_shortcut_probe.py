#!/usr/bin/env python3
"""SHORTCUT PROBE: classify the release using NO physics at all.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from mgxrl.detection.dataio import LoadConfig, load_all, feature_columns, SCENARIOS
from mgxrl.detection.splits import SPLITTERS
from mgxrl.detection.train_lgbm import DetectorConfig, _fit, SplitCoverageError


def build_indicator_features(df, feature_cols, ceiling=1.0e7, mode="binary"):
    """Artifact-only features.
    """
    X = pd.DataFrame(index=df.index)
    kept = []
    for c in feature_cols:
        col = pd.to_numeric(df[c], errors="coerce")
        div = (col.abs() > ceiling)
        nan = col.isna()
        if div.sum() or nan.sum():
            X[f"{c}__diverged"] = div.astype("int8")
            X[f"{c}__missing"] = nan.astype("int8")
            if mode == "magnitude":
                lg = np.zeros(len(col))
                vals = col.abs().to_numpy(dtype="float64", na_value=0.0)
                m = div.to_numpy() & np.isfinite(vals)
                with np.errstate(divide="ignore"):
                    lg[m] = np.log10(np.maximum(vals[m], 1.0))
                X[f"{c}__log10mag"] = lg
            kept.append(c)
    return X, kept


def conditional_report(X, y_file, pred, test_idx):
    """Accuracy split by whether any artifact flag fired.
    """
    flag_cols = [c for c in X.columns if c.endswith(("__diverged", "__missing"))]
    any_flag = X.iloc[test_idx][flag_cols].to_numpy().sum(axis=1) > 0
    out = {"frac_rows_flagged": float(any_flag.mean())}
    yt = y_file[test_idx]
    if any_flag.sum():
        out["accuracy_on_flagged"] = float((pred[any_flag] == yt[any_flag]).mean())
    if (~any_flag).sum():
        out["accuracy_on_unflagged"] = float((pred[~any_flag] == yt[~any_flag]).mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--decimate", type=int, default=500)
    ap.add_argument("--ceiling", type=float, default=1.0e7)
    ap.add_argument("--mode", default="binary", choices=["binary", "magnitude"],
                    help="binary = flags only (strictest, understates); "
                         "magnitude = also log10|x| on diverged samples, which "
                         "is still artifact-only and is what a detector sees")
    ap.add_argument("--compare-real", type=float, default=None,
                    help="macro-F1 of the real detector under the same protocol, "
                         "to report what fraction of above-chance performance the "
                         "artifact explains")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    # plausible_max=0 disables masking: we need the raw magnitudes here in
    # order to derive the indicators from them.
    df = load_all(LoadConfig(root=a.root, decimate=a.decimate, plausible_max=0.0),
                  verbose=False)
    feats = feature_columns(df)

    X, kept = build_indicator_features(df, feats, a.ceiling, a.mode)
    print(f"Loaded {len(df):,} rows. Channels showing corruption: {kept}")
    if X.shape[1] == 0:
        print("\nNo corruption indicators exist -- the release is clean on this "
              "measure and no shortcut is available. Nothing further to report.")
        return 0
    print(f"Indicator features: {X.shape[1]} (all binary, zero physical content)\n")

    # Per-scenario signature, for the paper's table
    sig = pd.DataFrame({
        "scenario": df["scenario"],
        **{c: X[c] for c in X.columns},
    }).groupby("scenario").mean()
    print("Fraction of samples flagged, by scenario:")
    print(sig.round(4).to_string())
    print()

    y = df["y_multi"].to_numpy()

    y_file = df["scenario"].map({s: i for i, s in enumerate(SCENARIOS)}).to_numpy()

    cfg = DetectorConfig()
    results = {}
    for proto in ("random", "block"):
        try:
            splits = SPLITTERS[proto](df, seed=0)
        except Exception as e:
            results[proto] = {"error": str(e)}
            continue
        Xs = {k: X.loc[v].to_numpy(np.float32) for k, v in splits.items()}
        ys = {k: y_file[df.index.get_indexer(v)] for k, v in splits.items()}
        try:
            m = _fit(Xs["train"], ys["train"], Xs["val"], ys["val"],
                     cfg, "multiclass", n_class=len(SCENARIOS))
        except SplitCoverageError as e:
            results[proto] = {"error": str(e)}
            print(f"  {proto}: SKIPPED -- {e}")
            continue

        from sklearn.metrics import f1_score, accuracy_score
        pred = m.predict(Xs["test"])
        per_class = f1_score(ys["test"], pred, average=None,
                             labels=list(range(len(SCENARIOS))), zero_division=0)
        test_pos = df.index.get_indexer(splits["test"])
        results[proto] = {
            "accuracy": float(accuracy_score(ys["test"], pred)),
            "macro_f1": float(f1_score(ys["test"], pred, average="macro",
                                       zero_division=0)),
            "per_class_f1": {SCENARIOS[i]: float(v) for i, v in enumerate(per_class)},
            **conditional_report(X, y_file, pred, test_pos),
        }

    chance = 1.0 / len(SCENARIOS)
    print(f"\n{'protocol':<10}{'accuracy':>10}{'macro F1':>10}{'flagged':>10}"
          f"{'acc|flagged':>13}   (chance = {chance:.3f})")
    for proto, r in results.items():
        if "error" in r:
            print(f"{proto:<10}  {r['error']}")
            continue
        print(f"{proto:<10}{r['accuracy']:>10.4f}{r['macro_f1']:>10.4f}"
              f"{r.get('frac_rows_flagged', float('nan')):>10.3f}"
              f"{r.get('accuracy_on_flagged', float('nan')):>13.4f}")
    print("\n  'flagged' = fraction of test rows where any artifact flag fired.")
    print("  Rows with no flag carry no information, so the model guesses on them")
    print("  and the overall figure is diluted. Read acc|flagged for the strength")
    print("  of the fingerprint and the overall for its reach.")

    best = max((r.get("macro_f1", 0) for r in results.values()), default=0)
    print()
    if a.compare_real is not None:
        num, den = best - chance, a.compare_real - chance
        if den > 0:
            print(f"Artifact accounts for {100 * num / den:.0f}% of the real "
                  f"detector's above-chance macro-F1 "
                  f"({a.compare_real:.4f}) under the same protocol.")
            print("Treat that as a LOWER bound: this probe sees only corruption")
            print("features, never the full column set the detector was given.\n")
    flagged_acc = max((r.get("accuracy_on_flagged", 0) for r in results.values()
                       if "error" not in r), default=0)
    if flagged_acc > 0.5:
        print(f"On rows where an artifact flag fires, the source file is identified")
        print(f"{flagged_acc:.1%} of the time against {chance:.1%} by chance. The overall")
        print("macro-F1 is lower only because unflagged rows carry no signal -- it")
        print("measures the fingerprint's REACH, not its STRENGTH. Report both.\n")

    if best > 3 * chance:
        print(f"VERDICT: a shortcut exists. Data-quality artifacts alone identify")
        print(f"the source file at macro-F1 {best:.3f}, versus {chance:.3f} by chance.")
        print("Any accuracy reported on this release -- published or your own --")
        print("must be shown NOT to rest on this before it can be read as evidence")
        print("about attack physics. The clean way to do that is to exclude the")
        print("affected channels entirely and re-report.")
    elif best > 1.5 * chance:
        print(f"VERDICT: a weak shortcut ({best:.3f} vs {chance:.3f} chance).")
        print("Report it and show your headline result survives excluding the")
        print("affected channels.")
    else:
        print(f"VERDICT: no meaningful shortcut ({best:.3f} vs {chance:.3f} chance).")
        print("The artifacts are not class-specific; this line of concern is closed.")

    json.dump({"results": results, "channels": kept,
               "signature": sig.round(6).to_dict()},
              open(os.path.join(a.out, "shortcut_probe.json"), "w"), indent=2)
    print(f"\nWrote {a.out}/shortcut_probe.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
