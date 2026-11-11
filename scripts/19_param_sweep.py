"""
19_param_sweep.py

Which parameter actually sets the plant's settling floor?

NUMERICAL GUARD
dt is reduced to keep the fastest represented mode at 0.5 * dt, estimated from
c_f * lambda_max, c_v * lambda_max, k_p, k_q and 1/tau_f. At the defaults this
leaves dt at 1 ms and changes nothing. Every point is checked for divergence,
and all measurements are resampled to a common grid before tau63 is read.

Usage (PowerShell, one line each):

  python scripts\\19_param_sweep.py --root C:\\research\\mgxrl\\Dataset --param k_p --values 5,10,20,40,80
  python scripts\\19_param_sweep.py --root C:\\research\\mgxrl\\Dataset --param tau_f --values 0.016,0.008,0.004,0.002
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


def timescale(t, y, frac=0.63):
    n = len(y)
    k = max(3, int(n * 0.2))
    final = float(np.nanmean(y[-k:]))
    prev = float(np.nanmean(y[-2 * k:-k])) if n > 2 * k else np.nan
    if not np.isfinite(final) or abs(final) <= 0:
        return np.nan, False, final
    flat = abs(final - prev) / abs(final) < 0.05 if np.isfinite(prev) else False
    hit = np.flatnonzero(np.abs(y) >= frac * abs(final))
    return (float(t[hit[0]]) if hit.size else np.nan), flat, final


def release_tau(root, attack, decimate, onset, window, grid_dt):
    def grab(scen):
        cfg = LoadConfig(root=root, decimate=decimate, max_files_per_scenario=1)
        df = load_scenario(root, scen, cfg)
        cmap = {c.strip().lower(): c for c in df.columns}
        t = pd.to_numeric(df[cmap.get("time", df.columns[0])],
                          errors="coerce").to_numpy(dtype=float)
        P = np.column_stack([pd.to_numeric(df[cmap[f"p_dg{i}"]], errors="coerce")
                             for i in range(1, N_DG + 1)]).astype(float)
        return t, P
    ta, Pa = grab(attack)
    tn, Pn = grab("Normal")
    g = np.arange(onset, onset + window, grid_dt)
    rg = lambda t, M: np.column_stack(
        [np.interp(g, t, M[:, j]) for j in range(M.shape[1])])
    D = rg(ta, Pa) - rg(tn, Pn)
    return timescale(g - onset, np.abs(np.delete(D, 0, axis=1)).mean(axis=1))


def safe_dt(p, lam_max, cap=1e-3):
    stiff = max(p.c_f * lam_max, p.c_v * lam_max, p.k_p, p.k_q,
                1.0 / max(p.tau_f, 1e-9))
    return min(cap, 0.5 / stiff)


def plant_point(param, value, attack, onset, span, grid_dt, lam_max):
    probe = MicrogridParams(**{param: value})
    dt = safe_dt(probe, lam_max)

    def run(attacked):
        p = MicrogridParams(seed=0, dt=dt, control_dt=dt, **{param: value})
        mg = Microgrid(p)
        if attacked:
            targets = (0,) if attack != "Stealth" else (0, 1, 2)
            mg.attach_attack(ATTACK_REGISTRY[attack](targets=targets,
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
    D = Pa - Pb
    diverged = (not np.all(np.isfinite(D))) or float(np.nanmax(np.abs(D))) > 1e3
    keep = ta >= onset
    t = ta[keep] - onset
    y = np.abs(np.delete(D[keep], 0, axis=1)).mean(axis=1)
    g = np.arange(0.0, span, grid_dt)
    tau, flat, final = timescale(g, np.interp(g, t, y))
    return {param: value, "dt_s": dt, "tau63_s": tau, "plateaued": flat,
            "final_mean9_pu": final, "diverged": diverged}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--param", default="k_p")
    ap.add_argument("--values", default="5,10,20,40,80")
    ap.add_argument("--attack", default="Additive")
    ap.add_argument("--decimate", type=int, default=500)
    ap.add_argument("--onset", type=float, default=0.5005)
    ap.add_argument("--window", type=float, default=0.3)
    ap.add_argument("--span", type=float, default=3.0)
    ap.add_argument("--grid-dt", type=float, default=1e-3)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    if not hasattr(MicrogridParams(), a.param):
        sys.exit(f"MicrogridParams has no field {a.param!r}")

    adj = ring_adjacency(N_DG)
    lam_max = float(np.sort(np.linalg.eigvalsh(np.diag(adj.sum(1)) - adj))[-1])
    tau_r, flat_r, final_r = release_tau(a.root, a.attack, a.decimate,
                                         a.onset, a.window, a.grid_dt)
    base = getattr(MicrogridParams(), a.param)
    print(f"sweeping {a.param} (default {base:g}) against release tau63 = "
          f"{tau_r:.4f} s (plateaued={flat_r})")

    vals = [float(v) for v in a.values.split(",")]
    df = pd.DataFrame([plant_point(a.param, v, a.attack, a.onset, a.span,
                                   a.grid_dt, lam_max) for v in vals])
    df["tau_x_release"] = df["tau63_s"] / tau_r if tau_r == tau_r else np.nan

    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda v: f"{v:.5g}")
    print(df.to_string(index=False))

    ok = df[(~df.diverged) & np.isfinite(df.tau63_s) & (df.tau63_s > 0)]
    if len(ok) >= 3:
        x, y = np.log(ok[a.param].to_numpy()), np.log(ok.tau63_s.to_numpy())
        print(f"\nlog-log slope, all points: {np.polyfit(x, y, 1)[0]:.3f}")
        if len(ok) >= 4:
            print(f"log-log slope, top three: "
                  f"{np.polyfit(x[-3:], y[-3:], 1)[0]:.3f}")
        print("  A slope holding near -1 through the TOP of the range means")
        print("  this parameter sets the timescale. A slope decaying toward 0")
        print("  means it does not, and the floor is elsewhere.")
        hit = ok[ok.tau63_s <= tau_r]
        if len(hit):
            print(f"  reaches the release target at {a.param} = "
                  f"{hit.iloc[0][a.param]:g}")
        else:
            print(f"  does NOT reach the release target anywhere in this range")

    df.to_csv(os.path.join(a.out, f"param_sweep_{a.param}.csv"), index=False)

    print("\nCautions:")
    print("  * A gain that matches tau63 is not thereby correct. k_p, k_q and")
    print("    tau_f are all in the same loop, and several combinations will")
    print("    hit 0.051 s. Matching one number is not identification.")
    print(f"\nWrote {a.out}/param_sweep_{a.param}.csv")


if __name__ == "__main__":
    main()
