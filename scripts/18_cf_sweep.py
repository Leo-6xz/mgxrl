"""
18_cf_sweep.py

Is c_f the knob that sets the plant's settling time, or is something else?

WHY THIS EXISTS
Script 17 measured tau63 of the nine unattacked DGs' Additive differential at
0.7825 s in the plant against 0.051 s in the release, a factor of ~15. The
obvious reading is that the consensus gain c_f = 10 is too small. But the
analytic slow mode 1/(c_f * lambda2) = 0.2618 s underestimates the plant's OWN
measured timescale by 3x, which means the consensus layer is not acting alone:
it is in series with the droop law and the tau_f = 16 ms power filter, and those
do not speed up when c_f rises.

So there are two possibilities and they have opposite costs:

  SCALES     tau63 falls as 1/c_f across the sweep. c_f is the knob, the value
             that lands on 0.051 s is roughly 10 * (0.7825 / 0.051) ~ 150, and
             fixing it is a config change followed by a retrain.

  SATURATES  tau63 falls and then flattens at some floor. That floor is the
             droop/filter timescale, c_f is NOT the knob, and no achievable
             gain reproduces the release. The 15x gap then becomes a documented
             modelling difference between two secondary layers rather than a
             parameter error, and nothing already banked is invalidated.

Read the log-log slope in the output. Near -1 is SCALES. Trending to 0 is
SATURATES.

NUMERICAL GUARD
Forward Euler on the consensus mode is stable only while dt * c_f * lambda_max
< 2, and lambda_max of a 10-ring Laplacian is 4, so c_f = 500 is the limit at
dt = 1 ms. A sweep that ignored this would produce a beautiful 1/c_f curve that
turned into numerical garbage at the top end and looked like saturation. dt is
therefore reduced automatically to keep the product at 0.5, and every run is
checked for divergence and reported rather than silently averaged. Measurements
are resampled to a common 1 ms grid so the timescale is not read off different
sampling rates.

This changes nothing in the repo. It constructs MicrogridParams locally per
point and leaves configs/microgrid.yaml alone.

Usage (PowerShell, one line):

  python scripts\\18_cf_sweep.py --root C:\\research\\mgxrl\\Dataset
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
# shared measurement (same definition as script 17)
# --------------------------------------------------------------------------

def timescale(t, y, frac=0.63):
    """Time to reach `frac` of the final level, with a plateau test.

    Not a curve fit, on purpose: a first-order fit returns a confident tau for
    a response that has not settled, which is exactly the failure mode this
    project keeps finding in its own metrics.
    """
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
    y = np.abs(np.delete(D, 0, axis=1)).mean(axis=1)
    return timescale(g - onset, y)


# --------------------------------------------------------------------------
# plant point
# --------------------------------------------------------------------------

def plant_point(c_f, attack, onset, span, grid_dt, lam_max):
    # Euler stability: keep dt * c_f * lambda_max at 0.5, capped at 1 ms.
    dt = min(1e-3, 0.5 / max(c_f * lam_max, 1e-9))

    def run(attacked):
        p = MicrogridParams(seed=0, c_f=c_f, dt=dt, control_dt=dt)
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
    # resample to a common grid so tau is not read off different rates
    g = np.arange(0.0, span, grid_dt)
    yg = np.interp(g, t, y)
    tau, flat, final = timescale(g, yg)
    return {"c_f": c_f, "dt_s": dt, "tau63_s": tau, "plateaued": flat,
            "final_mean9_pu": final, "diverged": diverged}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--attack", default="Additive")
    ap.add_argument("--values", default="10,20,50,100,200,300")
    ap.add_argument("--decimate", type=int, default=500)
    ap.add_argument("--onset", type=float, default=0.5005)
    ap.add_argument("--window", type=float, default=0.3)
    ap.add_argument("--span", type=float, default=3.0)
    ap.add_argument("--grid-dt", type=float, default=1e-3)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    adj = ring_adjacency(N_DG)
    lap = np.diag(adj.sum(1)) - adj
    lam = np.sort(np.linalg.eigvalsh(lap))
    lam2, lam_max = float(lam[1]), float(lam[-1])

    tau_r, flat_r, final_r = release_tau(a.root, a.attack, a.decimate,
                                         a.onset, a.window, a.grid_dt)
    print(f"ring: lambda2 = {lam2:.4f}, lambda_max = {lam_max:.4f}")
    print(f"release target: tau63 = {tau_r:.4f} s "
          f"(plateaued={flat_r}, final={final_r:.4f} W)")
    print(f"NOTE: the release is read at {a.window:.2f} s of window, so its "
          f"tau63 is only trustworthy well inside that.\n")

    vals = [float(v) for v in a.values.split(",")]
    rows = [plant_point(c, a.attack, a.onset, a.span, a.grid_dt, lam_max)
            for c in vals]
    df = pd.DataFrame(rows)
    df["tau_x_release"] = df["tau63_s"] / tau_r if tau_r and tau_r == tau_r else np.nan
    df["analytic_1_over_cf_lam2"] = 1.0 / (df["c_f"] * lam2)

    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda v: f"{v:.5g}")
    print(df.to_string(index=False))

    ok = df[(~df.diverged) & np.isfinite(df.tau63_s) & (df.tau63_s > 0)]
    if len(ok) >= 3:
        s = np.polyfit(np.log(ok.c_f), np.log(ok.tau63_s), 1)[0]
        print(f"\nlog-log slope of tau63 vs c_f over the non-diverged points: "
              f"{s:.3f}")
        print("  near -1  -> SCALES. c_f is the knob.")
        print("  near  0  -> SATURATES. the floor is the droop/tau_f loop.")
        # local slope at the top of the sweep is the one that decides it
        if len(ok) >= 4:
            s_hi = np.polyfit(np.log(ok.c_f.iloc[-3:]),
                              np.log(ok.tau63_s.iloc[-3:]), 1)[0]
            print(f"  slope over the top three points alone: {s_hi:.3f} "
                  f"-- this is the one that decides it, since saturation shows "
                  f"up at the high end first.")
    else:
        print("\nnot enough non-diverged points to fit a slope")

    df.to_csv(os.path.join(a.out, "cf_sweep.csv"), index=False)

    print("\nBefore acting on this:")
    print("  * One seed, one class, one run per point. A gain that lands on")
    print("    the release's tau63 for Additive is not thereby validated for")
    print("    Ramp, Sinusoid or Stealth, and the ranking of severities across")
    print("    classes can move when c_f moves.")
    print("  * c_v is untouched at 10. The Additive injector corrupts zeta as")
    print("    well as xi, so the voltage consensus has its own timescale that")
    print("    this sweep does not measure.")
    print("  * If it SCALES, the retrain cost is real: raising c_f shortens")
    print("    the window in which a source trusted at onset has already")
    print("    injected a corrupted value, which is the mechanism the hedging")
    print("    finding rests on. Rerun 13_observability_probe.py at the new")
    print("    gain BEFORE retraining -- if the innovation magnitudes move,")
    print("    the observation itself has changed and old checkpoints are not")
    print("    comparable to new ones.")
    print(f"\nWrote {a.out}/cf_sweep.csv")


if __name__ == "__main__":
    main()
