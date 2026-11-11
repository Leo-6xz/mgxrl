"""
21_topology_scaling.py

Is the 1.6 a property of a 10-ring, or a universal constant?

WHERE IT CAME FROM
Script 20 measured, over a 20x range of bias_xi with zero drift:

    spread_pu = 1.6000 * bias_xi / m_droop

Slope 1.0000 in log-log, ratio constant to five significant figures. A
redistribution factor arising from a ring Laplacian's eigenvectors would be
irrational; 1.6 = 8/5 exactly is not that kind of number, which is why it is
worth a run rather than a sentence.

THE HYPOTHESIS WORTH TESTING
For a ring of n nodes with the attack on one node's outgoing links, exactly two
nodes receive a corrupted value and n-2 do not. That suggests

    ratio(n) = 2 * (n - 2) / n

which gives 1.6000 at n = 10, and predicts 1.3333 at n = 6 and 1.7143 at n = 14.
It approaches 2 as n grows. This is a guess with one confirming point, which is
the weakest kind of hypothesis, so the sweep is the test rather than the
confirmation.

Three outcomes:
  * ratio tracks 2(n-2)/n      -> the constant is combinatorial and derivable,
                                  and the relation can be stated in closed form
                                  for any ring.
  * ratio is 1.6 at every n    -> it is universal and the ring size is
                                  irrelevant.
  * ratio moves but fits neither -> reported as measured at n = 10; hence the
                                  topology dependence is unresolved.

Script 19 cannot do this: it hardcodes ten DGs in its measurement arrays, so a
six-node run would be silently truncated rather than failing. This one sizes
everything from n_dg.

Everything else is held fixed: bias_xi, m_droop, the gains, the per-bus load and
setpoints. Only the ring size changes.

Usage (PowerShell, one line):

  python scripts\\21_topology_scaling.py --values 6,8,10,12,14 --span 20
"""

from __future__ import annotations

import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
from mgxrl.plant.microgrid import Microgrid, MicrogridParams, ring_adjacency
from mgxrl.plant.attacks import ATTACK_REGISTRY


def settled(y, tail=0.2):
    n = len(y)
    k = max(3, int(n * tail))
    final = float(np.nanmean(y[-k:]))
    prev = float(np.nanmean(y[-2 * k:-k])) if n > 2 * k else np.nan
    if not np.isfinite(final) or abs(final) <= 0:
        return final, False
    flat = abs(final - prev) / abs(final) < 0.02 if np.isfinite(prev) else False
    return final, flat


def point(n_dg, bias, attack, onset, span, dt, seed=0):
    def run(attacked):
        p = MicrogridParams(seed=seed, n_dg=n_dg, dt=dt, control_dt=dt)
        mg = Microgrid(p)
        if attacked:
            inj = ATTACK_REGISTRY[attack](targets=(0,), t_start=onset)
            setattr(inj, "bias_xi", bias)
            inj.reset()
            mg.attach_attack(inj)
        mg.reset(settle_time=3.0)
        steps = int(round((onset + span) / dt)) + 1
        T = np.empty(steps)
        P = np.empty((steps, n_dg)); F = np.empty((steps, n_dg))
        for k in range(steps):
            m = mg.measure(noisy=False)
            T[k] = m["t"]
            P[k] = np.asarray(m["P"])[:n_dg]
            F[k] = np.asarray(m["f"])[:n_dg]
            mg.step()
        return T, P, F

    ta, Pa, Fa = run(True)
    _, Pb, Fb = run(False)
    dP, dF = Pa - Pb, Fa - Fb
    diverged = (not np.all(np.isfinite(dP))) or float(np.nanmax(np.abs(dP))) > 1e3
    keep = ta >= onset

    spread, flat = settled((dP[keep].max(axis=1) - dP[keep].min(axis=1)))
    mean_others, _ = settled(np.abs(np.delete(dP[keep], 0, axis=1)).mean(axis=1))
    fspread, _ = settled(dF[keep].max(axis=1) - dF[keep].min(axis=1))

    adj = ring_adjacency(n_dg)
    lam = np.sort(np.linalg.eigvalsh(np.diag(adj.sum(1)) - adj))
    return {"n_dg": n_dg, "spread_pu": spread, "mean_others_pu": mean_others,
            "f_spread_Hz": fspread, "plateaued": bool(flat),
            "diverged": diverged, "lambda2": float(lam[1])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--attack", default="Additive")
    ap.add_argument("--values", default="6,8,10,12,14")
    ap.add_argument("--bias", type=float, default=0.3)
    ap.add_argument("--onset", type=float, default=0.5005)
    ap.add_argument("--span", type=float, default=20.0)
    ap.add_argument("--dt", type=float, default=1e-3)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    m = MicrogridParams().m_droop
    ns = [int(v) for v in a.values.split(",")]
    df = pd.DataFrame([point(n, a.bias, a.attack, a.onset, a.span, a.dt)
                       for n in ns])
    df["ratio"] = df.spread_pu / (a.bias / m)
    df["pred_2(n-2)/n"] = 2.0 * (df.n_dg - 2) / df.n_dg
    df["err_vs_pred"] = df.ratio - df["pred_2(n-2)/n"]
    df["err_vs_const"] = df.ratio - 1.6

    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", lambda v: f"{v:.6g}")
    print(f"bias_xi = {a.bias:g}, m_droop = {m:.4f}, "
          f"bias/m_droop = {a.bias / m:.6f} pu\n")
    print(df.to_string(index=False))

    ok = df[(~df.diverged) & df.plateaued]
    if len(ok) >= 2:
        e_pred = float(np.max(np.abs(ok.err_vs_pred)))
        e_const = float(np.max(np.abs(ok.err_vs_const)))
        print(f"\nmax |error| vs 2(n-2)/n : {e_pred:.6g}")
        print(f"max |error| vs constant 1.6: {e_const:.6g}")
        if e_pred < 0.01 and e_pred < e_const:
            print("  -> combinatorial. State it in closed form for any ring.")
        elif e_const < 0.01 and e_const < e_pred:
            print("  -> universal at 1.6. Ring size is irrelevant and the")
            print("     derivation is not the two-neighbour counting argument.")
        else:
            print("  -> neither. Report the measured value at n = 10 and leave")
            print("     the topology dependence open; do not fit a curve to")
            print("     three points to make it look resolved.")
    excl = df[~df.plateaued | df.diverged]
    if len(excl):
        print(f"\nexcluded (not settled or diverged): {list(excl.n_dg)}")

    print("\nCheck f_spread_Hz stays at float noise at every n. It is the")
    print("equilibrium premise -- equal frequencies across all buses -- that")
    print("the whole relation is derived from. If it grows with n, the larger")
    print("rings have not synchronized in the span and their ratios are not")
    print("steady-state values.")

    df.to_csv(os.path.join(a.out, "topology_scaling.csv"), index=False)
    print(f"\nWrote {a.out}/topology_scaling.csv")


if __name__ == "__main__":
    main()
