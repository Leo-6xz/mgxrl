"""Loading and decimation for the IEEE Dataport CSVs (doi:10.21227/bmtr-fr76)."""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

SCENARIOS = ["Normal", "Additive", "Ramp", "SlowRamp", "Sinusoid", "Stealth", "DoS"]
CLASS_INDEX = {name: i for i, name in enumerate(SCENARIOS)}

DATASET_DT = 2e-6           # s, native sampling interval
SECONDARY_DT = 0.010        # s, control-relevant timescale


@dataclass
class LoadConfig:
    root: str
    decimate: int = 5000        # 2 us -> 10 ms. See note below.
    anti_alias: bool = True     # block-average instead of subsampling
    max_files_per_scenario: int | None = None
    dtype: str = "float32"
    plausible_max: float = 1.0e7
    """Values beyond this are masked to NaN as solver divergence. 0 disables."""

    trim_startup_s: float = 0.0
    """Seconds to discard from the start of every run."""


PLAUSIBLE_MAX = 1.0e7
"""Generous ceiling for volts / amps / watts / Hz in a 380 V microgrid."""


def sanitize(df: pd.DataFrame, verbose: bool = True,
             plausible_max: float = PLAUSIBLE_MAX) -> pd.DataFrame:
    """Coercing feature columns to finite numerics and reporting what was wrong."""
    meta = {"scenario", "source_file", "run_id", "phase"}
    num_cols = [c for c in df.columns if c not in meta]

    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
    n_inf = int(np.isinf(df[num_cols].to_numpy(dtype="float64", na_value=np.nan)).sum())
    if n_inf:
        df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)

    # --- quarantine non-physical magnitudes --------------------------------
    # Masked to NaN rather than dropped: LightGBM handles NaN natively, the
    # leakage probe excludes non-finite columns, and the rows stay aligned so
    # the rest of the channels remain usable. Masking is also honest -- the
    # samples are recorded as missing, not silently replaced with a plausible
    # number that the model would then treat as a measurement.
    if plausible_max and plausible_max > 0:
        drop_meta = {"time", "t", "label", "attack_state", "y_bin", "y_multi"}
        check = [c for c in num_cols if c not in drop_meta]
        mag = df[check].abs()
        mask = mag > plausible_max
        n_bad = int(mask.to_numpy().sum())
        if n_bad:
            per_col = mask.sum()
            per_col = per_col[per_col > 0].sort_values(ascending=False)
            if verbose:
                print(f"  sanitize: {n_bad:,} value(s) exceed the physical ceiling "
                      f"({plausible_max:g}) -- masked to NaN")
                for c, k in per_col.head(8).items():
                    print(f"    {c:<20} {k:>9,} samples ({100.0 * k / len(df):5.1f}%)"
                          f"  max |x| = {float(mag[c].max()):.4g}")
                print("    These are solver divergence, not measurements. Report the")
                print("    affected channel and its extent; do not")
                print("    present a detector metric that depends on it.")
            df[check] = df[check].mask(mask)

    nan_counts = df[num_cols].isna().sum()
    bad = nan_counts[nan_counts > 0]

    if verbose and (len(bad) or n_inf):
        print(f"  sanitize: {n_inf} infinite value(s); "
              f"{len(bad)} column(s) contain NaN")
        for c, k in bad.sort_values(ascending=False).head(10).items():
            pct = 100.0 * k / len(df)
            print(f"    {c:<20} {k:>8,} NaN ({pct:5.1f}%)")

    # Drop columns that are entirely empty -- they carry no information and
    # would otherwise poison every distance computation downstream.
    empty = [c for c in num_cols if df[c].isna().all()]
    if empty:
        if verbose:
            print(f"  sanitize: dropping all-NaN column(s): {empty}")
        df = df.drop(columns=empty)

    return df


def redundancy_report(df: pd.DataFrame, cols: list[str], tol: float = 1e-9,
                      verbose: bool = True) -> dict:
    """Group numerically identical feature columns."""
    groups, seen = {}, {}
    for c in cols:
        x = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype="float64")
        key = None
        for k, y in seen.items():
            if y.shape == x.shape:
                scale = max(float(np.nanmax(np.abs(y))), 1e-12)
                if np.nanmax(np.abs(x - y)) / scale < tol:
                    key = k
                    break
        if key is None:
            seen[c] = x
            groups[c] = [c]
        else:
            groups[key].append(c)

    dupes = {k: v for k, v in groups.items() if len(v) > 1}
    if verbose and dupes:
        n_unique = len(groups)
        print(f"  redundancy: {len(cols)} columns -> {n_unique} unique signals")
        for k, v in list(dupes.items())[:6]:
            print(f"    {k} == {', '.join(v[1:6])}{' ...' if len(v) > 7 else ''}"
                  f"  ({len(v)} identical)")
    return {"n_columns": len(cols), "n_unique": len(groups),
            "duplicate_groups": {k: v for k, v in dupes.items()}}


def _canonical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names across folders (the release is not fully consistent)."""
    ren = {}
    for c in df.columns:
        k = c.strip().lower().replace(" ", "_")
        k = k.replace("pdg", "p_dg").replace("qdg", "q_dg").replace("fdg", "f_dg")
        k = k.replace("p__dg", "p_dg").replace("q__dg", "q_dg").replace("f__dg", "f_dg")
        ren[c] = k
    return df.rename(columns=ren)


def classify_ac_columns(df: pd.DataFrame, cols: list[str],
                        ratio_threshold: float = 0.3,
                        min_sign_changes: int = 4) -> list[str]:
    """Identify columns holding instantaneous alternating waveforms.

    """
    ac = []
    for c in cols:
        x = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype="float64")
        x = x[np.isfinite(x)]
        if len(x) < 50:
            continue
        rms = float(np.sqrt(np.mean(x ** 2)))
        if rms <= 0:
            continue
        sign_changes = int(np.count_nonzero(np.diff(np.signbit(x))))
        if sign_changes < min_sign_changes:
            continue
        if abs(float(np.mean(x))) / rms < ratio_threshold:
            ac.append(c)
    return ac


def decimate_block(df: pd.DataFrame, factor: int, anti_alias: bool = True,
                   ac_cols: list[str] | None = None) -> pd.DataFrame:
    """Decimate from 2 us to the control timescale."""
    if factor <= 1:
        return df
    n = (len(df) // factor) * factor
    if n == 0:
        return df.iloc[:0]
    d = df.iloc[:n]
    g = np.arange(n) // factor
    if anti_alias:
        num = d.select_dtypes(include=[np.number])
        if ac_cols is None:
            ac_cols = classify_ac_columns(
                d, [c for c in num.columns if c not in ("time", "label")])
        ac = [c for c in ac_cols if c in num.columns]
        dc = [c for c in num.columns if c not in ac]

        out = num[dc].groupby(g).mean()
        if ac:
            rms = num[ac].pow(2).groupby(g).mean().pow(0.5)
            out = pd.concat([out, rms], axis=1)[list(num.columns)]
        # Labels must be aggregated by max, never by mean: a block that contains
        # any attacked sample is an attacked block.
        for lab in ("label", "attack_state"):
            if lab in d.columns:
                # fillna(0) before the int cast: a NaN label would otherwise
                # cast to a garbage integer and silently corrupt the ground truth
                out[lab] = (d[lab].groupby(g).max()
                            .fillna(0).clip(0, 1).astype("int8"))
    else:
        out = d.iloc[::factor].reset_index(drop=True)
    return out.reset_index(drop=True)


def load_scenario(root: str, scenario: str, cfg: LoadConfig) -> pd.DataFrame:
    folder = os.path.join(root, scenario)
    files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if cfg.max_files_per_scenario:
        files = files[: cfg.max_files_per_scenario]
    if not files:
        raise FileNotFoundError(f"No CSVs under {folder}")

    frames, ac_cols = [], None
    for run_id, path in enumerate(files):
        df = _canonical_columns(pd.read_csv(path))
        if cfg.trim_startup_s > 0:
            tcol = "time" if "time" in df.columns else df.columns[0]
            n_before = len(df)
            df = df[pd.to_numeric(df[tcol], errors="coerce") >= cfg.trim_startup_s]
            df = df.reset_index(drop=True)
            if run_id == 0 and len(df) < n_before:
                print(f"  {scenario}: trimmed {n_before - len(df):,} startup "
                      f"samples (< {cfg.trim_startup_s}s)")
        if cfg.decimate > 1 and cfg.anti_alias and ac_cols is None:
            ac_cols = classify_ac_columns(
                df, [c for c in df.columns if c not in ("time", "label")])
            if ac_cols:
                print(f"  {scenario}: AC columns aggregated by RMS: {ac_cols}")
            else:
                print(f"  {scenario}: no AC columns detected; all means")
        df = decimate_block(df, cfg.decimate, cfg.anti_alias, ac_cols)
        df["scenario"] = scenario
        df["y_multi"] = CLASS_INDEX[scenario]
        # Before onset the sample is genuinely normal, even inside an attack
        # folder -- this is the mixed 0/1 labelling the author confirmed.
        if "label" in df.columns:
            df["y_bin"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).clip(0, 1).astype("int8")
            # phase: 0 = pre-attack, 1 = attack, 2 = post-attack recovery.
            y = df["y_bin"].to_numpy()
            phase = np.zeros(len(y), dtype="int8")
            if y.max() > 0:
                first, last = int(np.argmax(y > 0)), int(len(y) - np.argmax(y[::-1] > 0))
                phase[first:last] = 1
                phase[last:] = 2
            df["phase"] = phase
            df.loc[df["y_bin"] == 0, "y_multi"] = 0
        df["run_id"] = run_id
        df["source_file"] = os.path.basename(path)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_all(cfg: LoadConfig, verbose: bool = True) -> pd.DataFrame:
    df = pd.concat(
        [load_scenario(cfg.root, s, cfg) for s in SCENARIOS], ignore_index=True
    )
    return sanitize(df, verbose=verbose, plausible_max=cfg.plausible_max)


def feature_columns(df: pd.DataFrame) -> list[str]:
    drop = {"time", "t", "label", "attack_state", "y_bin", "y_multi", "phase",
            "scenario", "run_id", "source_file", "attack_type", "block_id"}
    return [c for c in df.columns
            if c not in drop and pd.api.types.is_numeric_dtype(df[c])]
