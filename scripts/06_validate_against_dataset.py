#!/usr/bin/env python3
"""validates the rebuilt plant's PRIMARY layer against the IEEE Dataport
release. The secondary layer is explicitly out of scope, and this file says why.

Usage (PowerShell, one line):

  python scripts\\06_validate_against_dataset.py --root C:\\research\\mgxrl\\Dataset
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
from mgxrl.detection.dataio import LoadConfig, load_scenario, SCENARIOS
from mgxrl.plant.microgrid import Microgrid, MicrogridParams
from mgxrl.plant.attacks import ATTACK_REGISTRY

N_DG = 10
CHANNELS = ("P", "Q", "f")
# Classes whose injection is zero-mean within the window have no meaningful
# signed level; direction is not reported for them.
OSCILLATORY = ("Sinusoid", "Stealth")


# --------------------------------------------------------------------------
# release side
# --------------------------------------------------------------------------

def load_release(root, scenario, decimate):
    cfg = LoadConfig(root=root, decimate=decimate, max_files_per_scenario=1)
    df = load_scenario(root, scenario, cfg)
    cmap = {c.strip().lower(): c for c in df.columns}
    t = pd.to_numeric(df[cmap.get("time", df.columns[0])],
                      errors="coerce").to_numpy(dtype=float)
    sig = {}
    for ch in CHANNELS:
        for i in range(1, N_DG + 1):
            key = f"{ch.lower()}_dg{i}"
            if key not in cmap:
                raise KeyError(f"{scenario}: missing {ch}_DG{i}")
            sig[f"{ch}{i}"] = pd.to_numeric(df[cmap[key]],
                                            errors="coerce").to_numpy(float)
    lab = None
    for k in ("y_bin", "label"):
        if k in cmap:
            lab = pd.to_numeric(df[cmap[k]], errors="coerce").to_numpy(float)
            break
    return t, sig, lab


def label_window(t, lab, default=(0.5005, 0.801)):
    if lab is None or not np.any(lab > 0):
        return default
    idx = np.flatnonzero(lab > 0)
    return float(t[idx[0]]), float(t[idx[-1]])


def regrid(t, y, grid):
    ok = np.isfinite(t) & np.isfinite(y)
    return (np.interp(grid, t[ok], y[ok], left=np.nan, right=np.nan)
            if ok.sum() >= 2 else np.full(grid.size, np.nan))


# --------------------------------------------------------------------------
# plant side
# --------------------------------------------------------------------------

def plant_run(scenario, t_start, t_end, attacked, dt, seed=0):
    """Deterministic trajectory, noise off. The attacked and unattacked runs
    share a seed and a settle, so with noisy=False they are identical until the
    injector fires -- the same property the release has for five classes, which
    is what makes both differentials exact rather than approximate."""
    mg = Microgrid(MicrogridParams(seed=seed, dt=dt, control_dt=dt))
    if attacked:
        targets = (0,) if scenario != "Stealth" else (0, 1, 2)
        mg.attach_attack(ATTACK_REGISTRY[scenario](targets=targets,
                                                   t_start=t_start))
    mg.reset(settle_time=3.0)
    n = int(round(t_end / dt)) + 1
    T = np.empty(n)
    rec = {f"{ch}{i}": np.empty(n) for ch in CHANNELS for i in range(1, N_DG + 1)}
    for k in range(n):
        m = mg.measure(noisy=False)
        T[k] = m["t"]
        for i in range(1, N_DG + 1):
            rec[f"P{i}"][k] = m["P"][i - 1]
            rec[f"Q{i}"][k] = m["Q"][i - 1]
            rec[f"f{i}"][k] = m["f"][i - 1]
        mg.step()
    return T, rec


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def end_level(x, frac=0.25):
    x = x[np.isfinite(x)]
    if x.size < 4:
        return float("nan")
    return float(np.mean(x[-max(2, int(x.size * frac)):]))


def rms(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.sqrt(np.mean(x ** 2))) if x.size else float("nan")


def corr(a, b):
    n = min(a.size, b.size)
    if n < 8:
        return float("nan")
    g = np.linspace(0, 1, n)
    a = np.interp(g, np.linspace(0, 1, a.size), a)
    b = np.interp(g, np.linspace(0, 1, b.size), b)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 8 or np.std(a[ok]) < 1e-15 or np.std(b[ok]) < 1e-15:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--decimate", type=int, default=100,
                    help="100 gives ~0.2 ms sampling, needed to resolve a 50 ms "
                         "primary window")
    ap.add_argument("--primary", type=float, default=0.05,
                    help="length of the droop-dominated comparison window, s")
    ap.add_argument("--trim", type=float, default=0.15)
    ap.add_argument("--dt", type=float, default=1e-3)
    ap.add_argument("--gate", type=float, default=10.0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    t_n, sig_n, _ = load_release(a.root, "Normal", a.decimate)

    rows, gates, per_dg = [], [], []
    for scen in SCENARIOS:
        if scen == "Normal":
            continue
        t_a, sig_a, lab_a = load_release(a.root, scen, a.decimate)
        onset, offset = label_window(t_a, lab_a)

        g_pre = np.arange(a.trim, onset, a.dt)
        d_pre = regrid(t_a, sig_a["P1"], g_pre) - regrid(t_n, sig_n["P1"], g_pre)
        g_full = np.arange(onset, offset, a.dt)
        d_post = regrid(t_a, sig_a["P1"], g_full) - regrid(t_n, sig_n["P1"], g_full)
        pre_rms, post_rms = rms(d_pre), rms(d_post)
        ratio = post_rms / pre_rms if pre_rms > 0 else np.inf
        gates.append({"attack": scen, "pre_rms_W": pre_rms,
                      "post_rms_W": post_rms, "ratio": ratio,
                      "shares_startup": bool(ratio >= a.gate)})
        if ratio < a.gate:
            rows.append({"attack": scen, "window": "-",
                         "verdict": "NOT COMPARABLE -- file does not share a "
                                    "startup with Normal"})
            continue

        tp, run_a = plant_run(scen, onset, offset, True, a.dt)
        _, run_b = plant_run(scen, onset, offset, False, a.dt)

        for wname, w_end in (("primary", onset + a.primary), ("full", offset)):
            grid = np.arange(onset, w_end, a.dt)
            rel = {k: regrid(t_a, sig_a[k], grid) - regrid(t_n, sig_n[k], grid)
                   for k in sig_a}
            msk = (tp >= onset) & (tp < w_end)
            pl = {k: (run_a[k] - run_b[k])[msk] for k in run_a}

            osc = scen in OSCILLATORY
            dP_r, dP_p = end_level(rel["P1"]), end_level(pl["P1"])

            # FREQUENCY IS COMPARED ON THE FLEET MEAN, NOT ON DG1.
            fr = np.nanmean([rel[f"f{i}"] for i in range(1, N_DG + 1)], axis=0)
            fp = np.nanmean([pl[f"f{i}"] for i in range(1, N_DG + 1)], axis=0)
            df_r, df_p = end_level(fr), end_level(fp)

            # HAS THE ATTACK DEVELOPED IN THIS WINDOW?
            dev = rms(rel["P1"]) / post_rms if post_rms > 0 else np.nan
            undeveloped = (wname == "primary") and (dev < 0.05)

            dir_P = None if osc else bool(np.sign(dP_r) == np.sign(dP_p))
            dir_f = None if osc else bool(np.sign(df_r) == np.sign(df_p))
            cP, cf = corr(rel["P1"], pl["P1"]), corr(fr, fp)

            if undeveloped:
                verdict = (f"NOT DEVELOPED -- only {dev * 100:.1f}% of the "
                           f"full-window response; no test either way")
                dir_P = dir_f = None
                cP = cf = float("nan")
            elif osc:
                verdict = "shape only (zero-mean injection; direction undefined)"
            elif dir_P and dir_f:
                verdict = "direction agrees on P and f"
            elif dir_f:
                verdict = "frequency agrees, power direction differs"
            elif dir_P:
                verdict = "power agrees, frequency direction differs"
            else:
                verdict = "direction differs on both"
            if wname == "full":
                verdict += " [plant secondary active -- not a primary test]"

            rows.append({
                "attack": scen, "window": wname,
                "len_s": round(w_end - onset, 4),
                "developed": round(dev, 4),
                "rmsP_release_W": round(rms(rel["P1"]), 4),
                "rmsP_plant_pu": round(rms(pl["P1"]), 6),
                "dP1_release_W": round(dP_r, 4),
                "dP1_plant_pu": round(dP_p, 6),
                "dfmean_release_Hz": round(df_r, 7),
                "dfmean_plant_Hz": round(df_p, 7),
                "dir_P": dir_P, "dir_f": dir_f,
                "corr_P1": round(cP, 4), "corr_f1": round(cf, 4),
                "verdict": verdict,
            })

            if wname == "primary":
                for i in range(1, N_DG + 1):
                    per_dg.append({"attack": scen, "dg": i,
                                   "dP_release_W": end_level(rel[f"P{i}"]),
                                   "dP_plant_pu": end_level(pl[f"P{i}"])})

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 240)
    print("\n--- gate: does the attacked file share a startup with Normal? ---")
    print(pd.DataFrame(gates).to_string(index=False))
    print("\n--- Tier C, primary layer ---")
    print(df.to_string(index=False))

    df.to_csv(os.path.join(a.out, "fidelity_tierC.csv"), index=False)
    pd.DataFrame(gates).to_csv(os.path.join(a.out, "fidelity_tierC_gate.csv"),
                               index=False)
    pd.DataFrame(per_dg).to_csv(os.path.join(a.out, "fidelity_tierC_per_dg.csv"),
                                index=False)

    print(f"""
Scope of this table
  The PRIMARY rows are the result. Over {a.primary * 1000:.0f} ms both systems
  are droop-dominated: the plant's power filter is 16 ms and its consensus
  timescale is ~250 ms, so its secondary has moved only a few percent.
  The FULL rows are shown for contrast and are NOT a primary-layer test --
  the plant's consensus is active across them and the release's is not.

  The consensus layer is not validated here and cannot be, from this
  release. Its secondary restoration runs beyond one second (confirmed by
  an author of the release) and the record is one second long, so the
  attack's effect on secondary control never develops inside the data.
  State that as a scope limit of the reference, not as an untested part
  of the plant.

  Direction is not reported for {', '.join(OSCILLATORY)}: their injection is
  zero-mean inside the window, so the sign of an end level is a statement
  about phase, not about the attack.

  Per-DG differentials are written to fidelity_tierC_per_dg.csv as
  description. They are not scored: the plant's per-DG footprint is shaped
  by its consensus graph, which is out of scope above.

Wrote {a.out}/fidelity_tierC.csv, _gate.csv, _per_dg.csv""")


if __name__ == "__main__":
    main()
