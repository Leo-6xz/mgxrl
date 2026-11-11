"""
17_settling_timescale.py

Does the release's secondary layer settle faster than the plant's, and if so, is
that a wrong consensus gain or a different communication graph?

This script changes nothing. It runs the plant unmodified and reads the release.

Usage (PowerShell, one line):

  python scripts\\17_settling_timescale.py --root C:\\research\\mgxrl\\Dataset
"""

from __future__ import annotations

import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
from mgxrl.detection.dataio import LoadConfig, load_scenario
from mgxrl.plant.microgrid import Microgrid, MicrogridParams, ring_adjacency
from mgxrl.plant.attacks import ATTACK_REGISTRY

N_DG = 10


# --------------------------------------------------------------------------
# sources
# --------------------------------------------------------------------------

def release_diff(root, scenario, decimate, onset, window, dt):
    def grab(scen):
        cfg = LoadConfig(root=root, decimate=decimate, max_files_per_scenario=1)
        df = load_scenario(root, scen, cfg)
        cmap = {c.strip().lower(): c for c in df.columns}
        t = pd.to_numeric(df[cmap.get("time", df.columns[0])],
                          errors="coerce").to_numpy(dtype=float)
        P = np.column_stack([pd.to_numeric(df[cmap[f"p_dg{i}"]], errors="coerce")
                             for i in range(1, N_DG + 1)]).astype(float)
        return t, P
    ta, Pa = grab(scenario)
    tn, Pn = grab("Normal")
    grid = np.arange(onset, onset + window, dt)
    rg = lambda t, M: np.column_stack(
        [np.interp(grid, t, M[:, j]) for j in range(M.shape[1])])
    return grid - onset, rg(ta, Pa) - rg(tn, Pn)


def plant_diff(scenario, onset, span, dt):
    def run(attacked):
        mg = Microgrid(MicrogridParams(seed=0, control_dt=dt))
        if attacked:
            targets = (0,) if scenario != "Stealth" else (0, 1, 2)
            mg.attach_attack(ATTACK_REGISTRY[scenario](targets=targets,
                                                       t_start=onset))
        mg.reset(settle_time=3.0)
        n = int(round((onset + span) / dt)) + 1
        T = np.empty(n); P = np.empty((n, N_DG))
        for k in range(n):
            m = mg.measure(noisy=False)
            T[k] = m["t"]; P[k] = m["P"][:N_DG]
            mg.step()
        return T, P
    ta, Pa = run(True)
    _, Pb = run(False)
    keep = ta >= onset
    return ta[keep] - onset, (Pa - Pb)[keep]


# --------------------------------------------------------------------------
# measurement
# --------------------------------------------------------------------------

def dispersion(D, target=0):
    """Cross-node dispersion of |dP| among the unattacked DGs, per timestep."""
    others = np.abs(np.delete(D, target, axis=1))
    mu = others.mean(axis=1)
    sd = others.std(axis=1)
    out = np.full(mu.shape, np.nan)
    ok = mu > 0
    out[ok] = sd[ok] / mu[ok]
    return out


def timescale(t, y, window):
    """Time to reach 63% of the final level, with a plateau test.

    Deliberately not a curve fit. A fit imposes first-order shape and will
    return a confident tau for a response that is still ramping.
    """
    n = len(y)
    k_end = max(3, int(n * 0.2))
    final = float(np.nanmean(y[-k_end:]))
    prev = float(np.nanmean(y[-2 * k_end:-k_end])) if n > 2 * k_end else np.nan
    plateaued = (abs(final) > 0 and abs(final - prev) / abs(final) < 0.05)
    if abs(final) <= 0:
        return np.nan, False, final
    hit = np.flatnonzero(np.abs(y) >= 0.63 * abs(final))
    if hit.size == 0:
        return np.nan, plateaued, final
    tau = float(t[hit[0]])
    return tau, plateaued, final


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--attack", default="Additive",
                    help="only a step-like class gives a meaningful tau63")
    ap.add_argument("--decimate", type=int, default=500)
    ap.add_argument("--dt", type=float, default=1e-3)
    ap.add_argument("--onset", type=float, default=0.5005)
    ap.add_argument("--window", type=float, default=0.3)
    ap.add_argument("--span", type=float, default=3.0,
                    help="plant run length after onset")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    p = MicrogridParams()
    adj = ring_adjacency(N_DG)
    lam = np.sort(np.linalg.eigvalsh(np.diag(adj.sum(1)) - adj))
    lam2 = float(lam[1])
    c_f = float(getattr(p, "c_f", np.nan))
    tau_pred = 1.0 / (c_f * lam2) if c_f == c_f and c_f > 0 else np.nan
    print(f"plant ring: lambda2 = {lam2:.4f}, c_f = {c_f:g}, "
          f"predicted slow mode tau = {tau_pred:.4f} s")

    rows, traces = [], {}
    tr, Dr = release_diff(a.root, a.attack, a.decimate, a.onset, a.window, a.dt)
    tp, Dp = plant_diff(a.attack, a.onset, a.span, a.dt)
    win = tp <= a.window

    for label, t, D in (("release", tr, Dr),
                        ("plant (window)", tp[win], Dp[win]),
                        ("plant (full run)", tp, Dp)):
        disp = dispersion(D)
        mean9 = np.abs(np.delete(D, 0, axis=1)).mean(axis=1)
        k_e = max(3, int(len(t) * 0.10))
        k_l = max(3, int(len(t) * 0.20))
        tau, flat, final = timescale(t, mean9, a.window)
        rows.append({
            "source": label,
            "D_early": float(np.nanmean(disp[:k_e])),
            "D_late": float(np.nanmean(disp[-k_l:])),
            "D_ratio": float(np.nanmean(disp[:k_e]) / np.nanmean(disp[-k_l:]))
                       if np.nanmean(disp[-k_l:]) > 0 else np.inf,
            "tau63_s": tau,
            "plateaued": flat,
            "final_mean9": final,
            "span_s": float(t[-1]),
        })
        traces[label] = (t, disp, mean9)

    res = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda v: f"{v:.5g}")
    print(f"\n--- {a.attack} ---")
    print(res.to_string(index=False))

    out = []
    for label, (t, disp, mean9) in traces.items():
        out.append(pd.DataFrame({"source": label, "t_since_onset": t,
                                 "dispersion": disp, "mean_abs_dP_others": mean9}))
    pd.concat(out).to_csv(os.path.join(a.out, "settling_timescale.csv"),
                          index=False)

    print("\nHow to read D_ratio, which is the discriminator:")
    print("  D_ratio >> 1 in the release  -> H1. Dispersion was there and")
    print("     decayed: same graph, faster gain. c_f is too small and the")
    print("     ratio of the two tau63 values is roughly the correction.")
    print("  D_ratio ~ 1 with D_early already small -> H2. There was never any")
    print("     distance structure: the release's comms graph is denser than a")
    print("     ring. Do NOT retune c_f to chase this -- a denser graph is a")
    print("     topology assumption, and changing it changes what per-source")
    print("     trust gating means, since the connectivity floor of 3 is")
    print("     defined against the ring's degree of 2.")
    print("  tau63 below ~3 ms in the release is at the decimation limit and")
    print("     should be reported as 'faster than resolution', not as a value.")
    print("\nEither way this is one measurement, one class, one run per source.")
    print("It does not on its own justify a retrain; it sizes the question.")
    print(f"\nWrote {a.out}/settling_timescale.csv")


if __name__ == "__main__":
    main()
