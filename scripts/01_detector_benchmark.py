#!/usr/bin/env python3
"""Stage 1 deliverable: the leakage-corrected detector table.

We run this FIRST, before any RL work.

    python scripts/01_detector_benchmark.py --root /path/to/dataport --out results/
"""
import argparse, json, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
from mgxrl.detection.dataio import (LoadConfig, load_all, feature_columns,
                                    sanitize)
from mgxrl.detection.features import window_features, physics_features
from mgxrl.detection.train_lgbm import DetectorConfig, evaluate_protocols, summarise, save


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="folder containing the 7 scenario dirs")
    ap.add_argument("--out", default="results")
    ap.add_argument("--decimate", type=int, default=500,
                    help="2us -> 1ms. NOTE: 5000 (10ms) leaves only 700 rows "
                         "on the 1-second Dataport release -- see RUNBOOK 1.0")
    ap.add_argument("--windowed", action="store_true", help="use windowed features")
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument("--trim", type=float, default=0.0,
                    help="seconds of startup transient to discard; the release "
                         "starts from ZERO initial conditions, so early samples "
                         "are an energization event. Measure with "
                         "scripts/10_inspect_release.py")
    ap.add_argument("--max-magnitude", type=float, default=1.0e7,
                    help="values beyond this are masked to NaN as solver "
                         "divergence; 0 disables the check")
    ap.add_argument("--drop-corrupt", action="store_true",
                    help="drop ENTIRELY any channel containing a non-physical or "
                         "missing value. The clean comparison: if performance "
                         "collapses toward chance without them, the release's "
                         "apparent separability was solver divergence, not attack "
                         "physics (see scripts/12_shortcut_probe.py)")
    ap.add_argument("--exclude", default="",
                    help="comma-separated channels to drop, e.g. v1,v3,i1")
    ap.add_argument("--repeats", type=int, default=1,
                    help="split seeds per protocol; use >=3 to get error bars")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    print(f"Loading from {a.root} (decimate={a.decimate}) ...")
    df = load_all(LoadConfig(root=a.root, decimate=a.decimate,
                             max_files_per_scenario=a.max_files,
                             trim_startup_s=a.trim,
                             plausible_max=a.max_magnitude),
                  verbose=True)
    print(f"  {len(df):,} rows x {df.shape[1]} cols")
    print(df.groupby("scenario")["y_bin"].agg(["size", "mean"]))

    runs = df.groupby("scenario")["run_id"].nunique()
    protocols = ["random", "block"] + (["scenario"] if runs.min() >= 3 else [])
    if "scenario" not in protocols:
        print(f"\n  NOTE: only {runs.min()} run(s) per scenario -- scenario-level "
              f"splitting unavailable. Use block-level and state this limitation "
              f"explicitly in the paper.\n")

    feats = feature_columns(df)

    # --- channel exclusion -------------------------------------------------
    dropped = []
    if a.exclude:
        want = {c.strip().lower() for c in a.exclude.split(",") if c.strip()}
        dropped += [c for c in feats if c.lower() in want]
    if a.drop_corrupt:
        # A channel is corrupt if it holds ANY NaN after sanitization -- which
        # is exactly where a non-physical value was masked, or where the export
        # already had one. Dropping the whole channel, not the affected rows,
        # is deliberate: the missingness PATTERN is itself a per-file
        # fingerprint, so masking rows leaves the shortcut intact.
        dropped += [c for c in feats if df[c].isna().any() and c not in dropped]
    if dropped:
        print(f"\n  dropping {len(dropped)} corrupt channel(s): {sorted(set(dropped))}")
        print("  Remaining features carry no solver-divergence signature.")
        feats = [c for c in feats if c not in set(dropped)]
        print(f"  {len(feats)} feature(s) remain\n")

    # Compute ONCE. Calling physics_features twice (the second time on the
    # already-augmented frame) regenerates the same names and puts duplicates
    # into `feats`, which silently doubles those columns in X.
    phys = physics_features(df)
    phys = phys[[c for c in phys.columns if c not in df.columns]]
    df = pd.concat([df, phys], axis=1)
    feats = feats + list(phys.columns)

    if a.windowed:
        print("Building windowed features ...")
        df = window_features(df, feats, window=20, stride=5)
        feats = [c for c in df.columns if "__" in c]

    print(f"Evaluating protocols: {protocols}")
    results, models = evaluate_protocols(df, feats, DetectorConfig(),
                                        tuple(protocols), repeats=a.repeats)
    table = summarise(results)
    print("\n" + table.to_string(index=False))

    save(results, os.path.join(a.out, "detector_protocols.json"))
    table.to_csv(os.path.join(a.out, "detector_protocols.csv"), index=False)
    print(f"\nWrote {a.out}/detector_protocols.{{json,csv}}")


if __name__ == "__main__":
    main()
