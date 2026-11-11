"""
14_preonset_identity_probe.py

Precondition test for the differential comparison.

The proposed rebuild of 06_validate_against_dataset.py subtracts the Normal file
from each attacked file over matched windows. That subtraction only removes the
energization transient if the two runs FOLLOW THE SAME TRAJECTORY BEFORE ONSET.
This script measures whether they do, before any of it is built.

Three outcomes:

  IDENTICAL   pre-onset difference at or near float noise. The two runs are the
              same solve up to the attack. The differential is exact and can be
              taken per timestep, not per window mean.

  SEPARABLE   pre-onset difference is non-zero but much smaller than the
              post-onset difference. The differential is usable; the pre-onset
              residual is the noise floor and every Tier B effect must clear it.

  CONFOUNDED  pre-onset difference is the same order as the post-onset
              difference. The runs do not share a startup, so attacked-minus-
              Normal measures run-to-run variation, not the attack. No
              single-run comparison on this release is defensible -> fifth
              forensic finding.

Usage (PowerShell, one line):

  python scripts\14_preonset_identity_probe.py --dataset C:\research\mgxrl\Dataset

Reads nothing from the package; only pandas + numpy, so was run against
the release without touching plant code.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# file discovery
# --------------------------------------------------------------------------

def scenario_name(path: str) -> str:
    """Scenario label from the deepest folder, falling back to the filename."""
    parent = os.path.basename(os.path.dirname(path))
    stem = os.path.splitext(os.path.basename(path))[0]
    # prefer whichever of the two actually names a scenario
    for cand in (parent, stem):
        low = cand.lower()
        if any(k in low for k in ("normal", "attack", "ramp", "dos", "sinus",
                                  "stealth", "additive", "slow")):
            return cand
    return f"{parent}/{stem}" if parent else stem


def find_files(dataset_dir: str) -> dict[str, str]:
    paths = sorted(glob.glob(os.path.join(dataset_dir, "**", "*.csv"),
                             recursive=True))
    if not paths:
        sys.exit(f"no CSVs found under {dataset_dir}")
    out: dict[str, str] = {}
    for p in paths:
        name = scenario_name(p)
        # if two CSVs collapse to the same label, disambiguate by filename
        if name in out:
            name = f"{name}::{os.path.basename(p)}"
        out[name] = p
    return out


def pick_normal(files: dict[str, str]) -> str:
    cands = [k for k in files if "normal" in k.lower()]
    if len(cands) == 1:
        return cands[0]
    if not cands:
        sys.exit("no file whose scenario label contains 'normal'; "
                 f"found: {sorted(files)}")
    sys.exit(f"ambiguous Normal file, candidates: {cands}")


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def time_column(df: pd.DataFrame) -> str:
    for c in df.columns:
        if "time" in c.strip().lower():
            return c
    # fall back: first numeric column that is strictly increasing
    for c in df.columns:
        v = pd.to_numeric(df[c], errors="coerce").to_numpy()
        if np.all(np.isfinite(v)) and np.all(np.diff(v) > 0):
            return c
    sys.exit(f"could not identify a time column among {list(df.columns)[:8]}")


def load(path: str) -> tuple[np.ndarray, pd.DataFrame]:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    tcol = time_column(df)
    t = pd.to_numeric(df[tcol], errors="coerce").to_numpy(dtype=float)
    data = df.drop(columns=[tcol]).apply(pd.to_numeric, errors="coerce")
    return t, data


def on_grid(t: np.ndarray, data: pd.DataFrame, grid: np.ndarray,
            cols: list[str]) -> np.ndarray:
    """Interpolate selected columns onto a common time grid. Non-finite samples
    are dropped per column before interpolation, so a diverged column does not
    poison a clean one."""
    out = np.full((grid.size, len(cols)), np.nan)
    for j, c in enumerate(cols):
        v = data[c].to_numpy(dtype=float)
        ok = np.isfinite(v) & np.isfinite(t)
        if ok.sum() < 2:
            continue
        out[:, j] = np.interp(grid, t[ok], v[ok], left=np.nan, right=np.nan)
    return out


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=r"C:\research\mgxrl\Dataset")
    ap.add_argument("--onset", type=float, default=0.5005,
                    help="attack onset in seconds (release: 0.5005)")
    ap.add_argument("--trim", type=float, default=0.15,
                    help="startup seconds discarded before the pre-onset window")
    ap.add_argument("--post", type=float, default=0.30,
                    help="length of the post-onset window in seconds")
    ap.add_argument("--dt", type=float, default=1e-4,
                    help="common comparison grid step")
    ap.add_argument("--cols", default="",
                    help="comma-separated columns to compare; default is every "
                         "column finite in both files over both windows")
    ap.add_argument("--out", default="results/preonset_identity.csv")
    args = ap.parse_args()

    files = find_files(args.dataset)
    norm_key = pick_normal(files)
    print(f"Normal reference : {norm_key}  ({files[norm_key]})")
    print(f"pre-onset window : [{args.trim:.4f}, {args.onset:.4f}) s")
    print(f"post-onset window: [{args.onset:.4f}, {args.onset + args.post:.4f}) s")
    print()

    t_n, d_n = load(files[norm_key])

    grid = np.arange(args.trim, args.onset + args.post, args.dt)
    pre = grid < args.onset
    post = ~pre
    if pre.sum() < 10 or post.sum() < 10:
        sys.exit("windows too short; check --trim/--onset/--post")

    rows = []
    for key, path in files.items():
        if key == norm_key:
            continue
        t_a, d_a = load(path)

        shared = [c for c in d_n.columns if c in d_a.columns]
        if args.cols:
            want = [c.strip() for c in args.cols.split(",")]
            shared = [c for c in shared if c in want]
        if not shared:
            print(f"{key}: no shared columns, skipped")
            continue

        A = on_grid(t_a, d_a, grid, shared)
        N = on_grid(t_n, d_n, grid, shared)

        n_dropped = 0
        for j, c in enumerate(shared):
            a, n = A[:, j], N[:, j]
            good = np.isfinite(a) & np.isfinite(n)
            if good[pre].sum() < 10 or good[post].sum() < 10:
                n_dropped += 1
                continue

            d = a - n
            pre_rms = float(np.sqrt(np.mean(d[pre & good] ** 2)))
            pre_max = float(np.max(np.abs(d[pre & good])))
            post_rms = float(np.sqrt(np.mean(d[post & good] ** 2)))
            # scale of the signal's own motion over the pre-onset window
            drift = float(np.ptp(n[pre & good]))
            exact = float(np.mean(d[pre & good] == 0.0))
            sep = post_rms / pre_rms if pre_rms > 0 else np.inf

            rows.append(dict(scenario=key, column=c,
                             pre_rms=pre_rms, pre_max=pre_max,
                             post_rms=post_rms, separation=sep,
                             normal_drift_pre=drift, exact_match_frac=exact,
                             n_pre=int(good[pre].sum()),
                             n_post=int(good[post].sum())))
        if n_dropped:
            print(f"{key}: {n_dropped}/{len(shared)} columns dropped "
                  f"(non-finite in one or both windows)")

    if not rows:
        sys.exit("nothing comparable; every column was dropped")

    res = pd.DataFrame(rows)

    def verdict(s: float) -> str:
        if s >= 100.0:
            return "IDENTICAL"
        if s >= 10.0:
            return "SEPARABLE"
        if s >= 3.0:
            return "MARGINAL"
        return "CONFOUNDED"

    res["verdict"] = res["separation"].map(verdict)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    res.to_csv(args.out, index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda v: f"{v:.4g}")

    print("\n--- per column ---")
    print(res[["scenario", "column", "pre_rms", "post_rms", "separation",
               "normal_drift_pre", "exact_match_frac", "verdict"]]
          .to_string(index=False))

    print("\n--- per scenario (worst column governs) ---")
    agg = (res.groupby("scenario")
              .agg(min_separation=("separation", "min"),
                   median_separation=("separation", "median"),
                   max_pre_rms=("pre_rms", "max"),
                   mean_exact=("exact_match_frac", "mean"))
              .reset_index())
    agg["verdict"] = agg["min_separation"].map(verdict)
    print(agg.to_string(index=False))

    print(f"\nwritten: {args.out}")
    print("\nreading: IDENTICAL or SEPARABLE on the columns you intend to use "
          "(P_DG*, f_DG*) means the differential rebuild of 06 is sound and "
          "pre_rms is the floor every Tier B effect must clear. CONFOUNDED on "
          "those columns means attacked-minus-Normal measures run-to-run "
          "variation and no comparison on this release is defensible.")


if __name__ == "__main__":
    main()
