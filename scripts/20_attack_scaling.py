"""
20_attack_scaling.py

Is the steady-state effect of a consensus bias equal to the bias divided by the
droop coefficient, or does it only look that way at one point?

THE CLAIM UNDER TEST
At equilibrium every DG runs at the same frequency, so w_i = w_j gives

    dP_i - dP_j = (xi_i - xi_j) / m_droop

which contains no control gain. That is why c_f over 30x (script 18) and k_p
over 8x (script 19) leave final_mean9_pu pinned at 0.10003 pu, and why raising
k_p costs a factor of four in settling for no rejection at all. At bias_xi =
0.3 rad/s and m_droop = 1.5708, the predicted cross-node spread is 0.191 pu
against a measured 0.200.

One point is not a law. If the relation is real, the effect must scale LINEARLY
with the injected bias and the slope must go as 1/m_droop.

WHAT IS AND IS NOT PREDICTED
Linearity is the strong prediction and it is what this script tests. The exact
coefficient is weaker: the bias enters the transmitted copy seen by the target's
neighbours, and the consensus layer then redistributes it over the graph, so how
much of the injected bias ends up as an xi SPREAD depends on the topology. Read
a slope of 1.0 in log-log as confirmation of the mechanism. Do not read a
coefficient that misses 1/(2*m_droop) by some factor as refutation -- that
factor is the graph's, and it is constant.

The 1/m_droop half of the claim needs no new code. m_droop is a MicrogridParams
field, so script 19 tests it directly:

  python scripts\\19_param_sweep.py --root <DATASET> --param m_droop --values 0.785,1.571,3.142 --span 20

Additive is the default because it is the only class with a constant post-onset
injection, so it is the only one with a steady state to measure. This script
changes nothing in the repo; it sets the attribute on a locally constructed
injector.

Usage (PowerShell, one line):

  python scripts\\20_attack_scaling.py --values 0.05,0.1,0.2,0.3,0.6,1.0 --span 20
"""

from __future__ import annotations

import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
from mgxrl.plant.microgrid import Microgrid, MicrogridParams
from mgxrl.plant.attacks import ATTACK_REGISTRY

N_DG = 10


def settled(t, y, tail=0.2):
    """Final level plus a plateau test. Every conclusion here is a STEADY-STATE
    claim, so a row that has not settled contributes nothing and must be
    visibly excluded rather than averaged in."""
    n = len(y)
    k = max(3, int(n * tail))
    final = float(np.nanmean(y[-k:]))
    prev = float(np.nanmean(y[-2 * k:-k])) if n > 2 * k else np.nan
    if not np.isfinite(final) or abs(final) <= 0:
        return final, False
    flat = abs(final - prev) / abs(final) < 0.02 if np.isfinite(prev) else False
    return final, flat


def point(attack, attr, value, onset, span, dt, seed=0):
    def run(attacked):
        mg = Microgrid(MicrogridParams(seed=seed, dt=dt, control_dt=dt))
        if attacked:
            targets = (0,) if attack != "Stealth" else (0, 1, 2)
            inj = ATTACK_REGISTRY[attack](targets=targets, t_start=onset)
            if not hasattr(inj, attr):
                sys.exit(f"{attack} injector has no attribute {attr!r}")
            setattr(inj, attr, value)
            inj.reset()
            mg.attach_attack(inj)
        mg.reset(settle_time=3.0)
        n = int(round((onset + span) / dt)) + 1
        T = np.empty(n); P = np.empty((n, N_DG)); F = np.empty((n, N_DG))
        for k in range(n):
            m = mg.measure(noisy=False)
            T[k] = m["t"]; P[k] = m["P"][:N_DG]; F[k] = m["f"][:N_DG]
            mg.step()
        return T, P, F

    ta, Pa, Fa = run(True)
    _, Pb, Fb = run(False)
    dP, dF = Pa - Pb, Fa - Fb
    diverged = (not np.all(np.isfinite(dP))) or float(np.nanmax(np.abs(dP))) > 1e3
    keep = ta >= onset
    t = ta[keep] - onset

    mean9 = np.abs(np.delete(dP[keep], 0, axis=1)).mean(axis=1)
    spread = dP[keep].max(axis=1) - dP[keep].min(axis=1)
    fspread = dF[keep].max(axis=1) - dF[keep].min(axis=1)

    m9, flat9 = settled(t, mean9)
    sp, flatsp = settled(t, spread)
    fs, _ = settled(t, fspread)
    return {attr: value, "mean9_pu": m9, "spread_pu": sp,
            "f_spread_Hz": fs, "plateaued": bool(flat9 and flatsp),
            "diverged": diverged}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--attack", default="Additive")
    ap.add_argument("--attr", default="bias_xi")
    ap.add_argument("--values", default="0.05,0.1,0.2,0.3,0.6,1.0")
    ap.add_argument("--onset", type=float, default=0.5005)
    ap.add_argument("--span", type=float, default=20.0)
    ap.add_argument("--dt", type=float, default=1e-3)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    m = MicrogridParams().m_droop
    vals = [float(v) for v in a.values.split(",")]
    df = pd.DataFrame([point(a.attack, a.attr, v, a.onset, a.span, a.dt)
                       for v in vals])
    df["predicted_spread_pu"] = df[a.attr] / m
    df["measured_over_predicted"] = df["spread_pu"] / df["predicted_spread_pu"]

    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda v: f"{v:.5g}")
    print(f"m_droop = {m:.4f} rad/s per pu; predicted spread = {a.attr}/m_droop\n")
    print(df.to_string(index=False))

    ok = df[(~df.diverged) & df.plateaued & (df.spread_pu > 0)]
    if len(ok) >= 3:
        x, y = np.log(ok[a.attr].to_numpy()), np.log(ok.spread_pu.to_numpy())
        slope = float(np.polyfit(x, y, 1)[0])
        ratios = ok.measured_over_predicted.to_numpy()
        print(f"\nlog-log slope of spread vs {a.attr}: {slope:.4f}")
        print(f"measured/predicted: mean {ratios.mean():.4f}, "
              f"spread across points {ratios.max() - ratios.min():.4f}")
        print("\n  slope ~ 1.0 and a CONSTANT ratio -> the relation holds and")
        print("     the ratio is the graph's redistribution factor, not an")
        print("     error")
        print("  slope departing from 1.0 at the top -> the plant is leaving")
        print("     its linear range at large bias. Range should be reported")
        print("     which the relation was verified, not the whole sweep.")
        print("  a DRIFTING ratio with slope 1.0 -> something else scales with")
        print("     the bias too; the single-point agreement was luck.")
    else:
        print("\nfewer than three settled, non-diverged points -- extend --span")

    excl = df[~df.plateaued | df.diverged]
    if len(excl):
        print(f"\nexcluded from the fit ({len(excl)} rows, not settled or "
              f"diverged): {list(excl[a.attr])}")

    df.to_csv(os.path.join(a.out, f"attack_scaling_{a.attr}.csv"), index=False)
    print(f"\nWrote {a.out}/attack_scaling_{a.attr}.csv")
    print("\nThen the other half of the claim, no new code:")
    print("  scripts\\19_param_sweep.py --param m_droop "
          "--values 0.785,1.571,3.142 --span 20")


if __name__ == "__main__":
    main()
