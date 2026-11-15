#!/usr/bin/env python3
"""Non-learned baselines against the trust-weighted agent, same protocol.

Runs the nominal controller, W-MSR (F=1), the innovation-threshold rule and
oracle isolation through `sweep` on the evaluation seeds, checks the nominal arm
reproduces the archived one, and compares every arm with the archived agent
sweeps (results/eval3_s*) seed by seed. No training and no torch needed.

    python scripts\\23_baselines.py --out results\\baselines
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from mgxrl.envs.resilient_env import ResilientMicrogridEnv
from mgxrl.eval.benchmarks import sweep, paired_delta, false_trigger_rate
from mgxrl.eval.baselines import (use_wmsr, calibrate_threshold, ThresholdPolicy,
                                  OraclePolicy, wmsr_edges_kept)

METRICS = ["return", "f_rmse", "p_sharing_error", "v_violation_rate"]


def summary(arms):
    rows = []
    for name, df in arms.items():
        g = df.groupby("attack")
        s = g[METRICS].mean()
        s["collapsed"] = g["collapsed"].sum()
        s["controller"] = name
        rows.append(s.reset_index())
    return pd.concat(rows, ignore_index=True)


def paired_wins(a, b, label):
    """Per class: mean (b - a) return, paired t, and seeds where b beats a."""
    m = a.merge(b, on=["attack", "seed"], suffixes=("_a", "_b"))
    d = m["return_b"] - m["return_a"]
    g = d.groupby(m["attack"])
    se = g.std() / np.sqrt(g.count())
    return pd.DataFrame({"comparison": label, "delta_return": g.mean(),
                         "t": g.mean() / se.replace(0, np.nan),
                         "n_better": (d > 0).groupby(m["attack"]).sum().astype(int),
                         "n": g.count()}).reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--calib-seeds", type=int, default=20,
                    help="attack-free calibration episodes, seeds 500.. (disjoint from evaluation)")
    ap.add_argument("--margin", type=float, default=1.5,
                    help="threshold = margin x largest attack-free |innovation|")
    ap.add_argument("--agent-dirs", nargs="*",
                    default=["results/eval3_s0", "results/eval3_s1", "results/eval3_s2"])
    ap.add_argument("--out", default="results/baselines")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    env = ResilientMicrogridEnv()

    print("Nominal ...")
    nom = sweep(env, None, n_seeds=a.seeds, verbose=True)
    ref = os.path.join(a.agent_dirs[0], "sweep_nominal.csv") if a.agent_dirs else None
    repro = None
    if ref and os.path.exists(ref):
        old = pd.read_csv(ref)
        m = nom.merge(old, on=["attack", "seed"], suffixes=("", "_old"))
        repro = float((m["return"] - m["return_old"]).abs().max())
        print(f"  nominal reproduces {ref}: max |return diff| = {repro:.3g}")

    print("W-MSR (F=1) ...")
    with use_wmsr(env, 1):
        wm = sweep(env, None, n_seeds=a.seeds, verbose=True)

    print("Calibrating threshold ...")
    tq_xi, tq_zeta = calibrate_threshold(env, range(500, 500 + a.calib_seeds))
    tau = (a.margin * tq_xi, a.margin * tq_zeta)
    print(f"  max attack-free |innovation|: xi {tq_xi:.4g}, zeta {tq_zeta:.4g}; "
          f"thresholds {tau[0]:.4g}, {tau[1]:.4g}")
    thr_pol = ThresholdPolicy(env, *tau)
    print("Threshold rule ...")
    thr = sweep(env, thr_pol, n_seeds=a.seeds, verbose=True)

    print("Oracle isolation ...")
    ora = sweep(env, OraclePolicy(env), n_seeds=a.seeds, verbose=True)

    arms = {"nominal": nom, "wmsr": wm, "threshold": thr, "oracle": ora}
    agents = {}
    for d in a.agent_dirs:
        f = os.path.join(d, "sweep_agent.csv")
        if os.path.exists(f):
            agents[os.path.basename(d.rstrip("/\\"))] = pd.read_csv(f)
    for k, v in agents.items():
        arms[f"agent_{k}"] = v

    for k, v in arms.items():
        if not k.startswith("agent_"):
            v.to_csv(os.path.join(a.out, f"sweep_{k}.csv"), index=False)

    summ = summary(arms)
    summ.to_csv(os.path.join(a.out, "summary.csv"), index=False)

    comps = [paired_wins(nom, arms[k], f"{k} vs nominal") for k in arms if k != "nominal"]
    for k, ag in agents.items():
        for b in ("wmsr", "threshold", "oracle"):
            comps.append(paired_wins(arms[b], ag, f"agent_{k} vs {b}"))
    comp = pd.concat(comps, ignore_index=True)
    comp.to_csv(os.path.join(a.out, "paired.csv"), index=False)

    print("\nNormal-operation cost ...")
    diag = {"threshold_tau_xi": tau[0], "threshold_tau_zeta": tau[1],
            "calib_max_innov_xi": tq_xi, "calib_max_innov_zeta": tq_zeta,
            "margin": a.margin, "nominal_repro_max_abs_return_diff": repro,
            **wmsr_edges_kept(env, n_seeds=a.seeds)}
    ft = false_trigger_rate(env, thr_pol, n_seeds=a.seeds)
    diag["threshold_mean_connectivity_loss"] = ft["mean_connectivity_loss"]
    diag["threshold_frac_steps_any_isolated"] = ft["frac_steps_any_isolated"]
    json.dump(diag, open(os.path.join(a.out, "diagnostics.json"), "w"), indent=2)

    pd.set_option("display.width", 200)
    print("\nMean return by class and controller:")
    print(summ.pivot(index="attack", columns="controller", values="return").round(0).to_string())
    print("\nCollapsed episodes (of", a.seeds, "):")
    print(summ.pivot(index="attack", columns="controller", values="collapsed").to_string())
    print("\nPaired return comparisons:")
    print(comp.round(2).to_string(index=False))
    print("\nDiagnostics:", json.dumps(diag, indent=2))


if __name__ == "__main__":
    main()
