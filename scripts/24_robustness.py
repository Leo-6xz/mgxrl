#!/usr/bin/env python3
"""Robustness of the source-level trust rule: no training, no torch needed.

Parts (run one or several):
  comms     delay and packet loss on the consensus links; threshold calibrated
            under the same channel ("matched") and on ideal links ("ideal")
  adaptive  an attacker who knows the threshold and stays under it
  graphs    denser consensus graphs (circulant k = 2, 3), where W-MSR is
            2- and 3-robust; nominal, W-MSR and threshold
  ringsize  rings of 6 and 14 units: nominal, threshold, oracle
  latency   time from onset to isolation, and false isolations, for the rule
  radial    transfer to a radial feeder with a chain consensus graph

    python scripts\\24_robustness.py --part comms
    python scripts\\24_robustness.py --part adaptive latency graphs ringsize
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from mgxrl.envs.resilient_env import ResilientMicrogridEnv
from mgxrl.plant.microgrid import MicrogridParams, CommChannel
from mgxrl.plant.attacks import ThresholdAware, NoAttack
from mgxrl.eval.benchmarks import sweep, rollout, episode_metrics, false_trigger_rate
from mgxrl.eval.baselines import (use_wmsr, calibrate_threshold, ThresholdPolicy,
                                  OraclePolicy, wmsr_edges_kept, circulant_adjacency,
                                  set_comm_graph, r_robustness, isolation_latency)

MARGIN = 1.5
CALIB = range(500, 520)
ATTACKS = ["Normal", "Additive", "Ramp", "SlowRamp", "Sinusoid", "Stealth", "DoS"]


def paired(base, arm, label):
    m = base.merge(arm, on=["attack", "seed"], suffixes=("_b", "_a"))
    d = m["return_a"] - m["return_b"]
    g = d.groupby(m["attack"])
    se = g.std() / np.sqrt(g.count())
    return pd.DataFrame({"arm": label, "mean_return": m.groupby("attack")["return_a"].mean(),
                         "delta_vs_nominal": g.mean(), "t": g.mean() / se.replace(0, np.nan),
                         "n_better": (d > 0).groupby(m["attack"]).sum().astype(int),
                         "collapsed": m.groupby("attack")["collapsed_a"].sum().astype(int)})


def threshold_arm(env, seeds, tau=None):
    if tau is None:
        tx, tz = calibrate_threshold(env, CALIB)
        tau = (MARGIN * tx, MARGIN * tz)
    pol = ThresholdPolicy(env, *tau)
    df = sweep(env, pol, n_seeds=seeds)
    ft = false_trigger_rate(env, pol, n_seeds=seeds)
    return df, pol, tau, ft


def part_comms(out, seeds):
    env = ResilientMicrogridEnv()
    tx, tz = calibrate_threshold(env, CALIB)
    tau_ideal = (MARGIN * tx, MARGIN * tz)
    rows, diag = [], {}
    for delay, loss in [(10, 0.0), (50, 0.0), (0, 0.2), (0, 0.5), (20, 0.2)]:
        tag = f"d{delay}ms_l{loss:g}"
        env.mg.channel = CommChannel(delay_steps=delay, loss_p=loss, seed=7)
        nom = sweep(env, None, n_seeds=seeds)
        mat, _, tau_m, ft_m = threshold_arm(env, seeds)
        idl, _, _, ft_i = threshold_arm(env, seeds, tau_ideal)
        for k, v in (("nominal", nom), ("threshold_matched", mat), ("threshold_ideal", idl)):
            v.to_csv(os.path.join(out, f"sweep_{tag}_{k}.csv"), index=False)
        for lab, arm in (("threshold_matched", mat), ("threshold_ideal", idl)):
            p = paired(nom, arm, lab); p["channel"] = tag; rows.append(p.reset_index())
        diag[tag] = {"tau_matched": tau_m, "tau_ideal": tau_ideal,
                     "normal_loss_matched": ft_m["mean_connectivity_loss"],
                     "normal_any_isolated_matched": ft_m["frac_steps_any_isolated"],
                     "normal_loss_ideal": ft_i["mean_connectivity_loss"],
                     "normal_any_isolated_ideal": ft_i["frac_steps_any_isolated"]}
        print(tag, json.dumps(diag[tag]), flush=True)
    env.mg.channel = None
    res = pd.concat(rows, ignore_index=True)
    res.to_csv(os.path.join(out, "comms_paired.csv"), index=False)
    json.dump(diag, open(os.path.join(out, "comms_diag.json"), "w"), indent=2)
    print(res.round(2).to_string(index=False), flush=True)


def part_adaptive(out, seeds):
    env = ResilientMicrogridEnv()
    tx, tz = calibrate_threshold(env, CALIB)
    tau = (MARGIN * tx, MARGIN * tz)
    pol = ThresholdPolicy(env, *tau)
    rows = []
    for alpha in (0.5, 0.9, 1.2):
        for s in range(seeds):
            rng = np.random.default_rng(1000 + s)
            tgt = (int(rng.integers(env.n)),)
            t0 = float(rng.uniform(0.5, env.cfg.episode_seconds * 0.6))
            for arm, p in (("nominal", None), ("threshold", pol)):
                inj = ThresholdAware(targets=tgt, t_start=t0, adj=env.mg.adj,
                                     tau_xi=tau[0], tau_zeta=tau[1],
                                     xi_limit=env.pp.xi_limit, zeta_limit=env.pp.zeta_limit,
                                     alpha=alpha, rng=rng)
                inj.reset()
                df, info = rollout(env, p, injector=inj, seed=s)
                a = np.stack(df["action_norm"].to_numpy())
                rows.append({"alpha": alpha, "seed": s, "arm": arm, **episode_metrics(df), **info,
                             "ever_isolated": bool((a < -0.5).any())})
        for arm, p in (("nominal", None),):
            pass
    df = pd.DataFrame(rows)
    ref = sweep(env, None, n_seeds=seeds, attacks=["Normal", "Additive"])
    df.to_csv(os.path.join(out, "adaptive.csv"), index=False)
    ref.to_csv(os.path.join(out, "adaptive_reference.csv"), index=False)
    s = df.groupby(["alpha", "arm"])[["return", "p_sharing_error", "f_rmse", "ever_isolated"]].mean()
    print(s.round(4).to_string(), flush=True)
    print(ref.groupby("attack")[["return", "p_sharing_error", "f_rmse"]].mean().round(4).to_string(), flush=True)


def part_graphs(out, seeds):
    rows, diag = [], {}
    for k in (2, 3):
        env = ResilientMicrogridEnv()
        A = circulant_adjacency(env.n, k)
        set_comm_graph(env, A)
        tag = f"circ{k}"
        nom = sweep(env, None, n_seeds=seeds)
        with use_wmsr(env, 1):
            wm = sweep(env, None, n_seeds=seeds)
        thr, _, tau, ft = threshold_arm(env, seeds)
        for lab, v in (("nominal", nom), ("wmsr", wm), ("threshold", thr)):
            v.to_csv(os.path.join(out, f"sweep_{tag}_{lab}.csv"), index=False)
        for lab, arm in (("wmsr", wm), ("threshold", thr)):
            p = paired(nom, arm, lab); p["graph"] = tag; rows.append(p.reset_index())
        diag[tag] = {"degree": 2 * k, "r_robust": r_robustness(A), "tau": tau,
                     "threshold_normal_loss": ft["mean_connectivity_loss"],
                     **wmsr_edges_kept(env, n_seeds=seeds)}
        print(tag, json.dumps(diag[tag]), flush=True)
    res = pd.concat(rows, ignore_index=True)
    res.to_csv(os.path.join(out, "graphs_paired.csv"), index=False)
    json.dump(diag, open(os.path.join(out, "graphs_diag.json"), "w"), indent=2)
    print(res.round(2).to_string(index=False), flush=True)


def part_ringsize(out, seeds):
    rows, diag = [], {}
    for n in (6, 14):
        env = ResilientMicrogridEnv(plant_params=MicrogridParams(n_dg=n))
        nom = sweep(env, None, n_seeds=seeds)
        thr, _, tau, ft = threshold_arm(env, seeds)
        ora = sweep(env, OraclePolicy(env), n_seeds=seeds)
        for lab, v in (("nominal", nom), ("threshold", thr), ("oracle", ora)):
            v.to_csv(os.path.join(out, f"sweep_n{n}_{lab}.csv"), index=False)
        for lab, arm in (("threshold", thr), ("oracle", ora)):
            p = paired(nom, arm, lab); p["n_dg"] = n; rows.append(p.reset_index())
        diag[f"n{n}"] = {"tau": tau, "threshold_normal_loss": ft["mean_connectivity_loss"]}
        print(n, json.dumps(diag[f"n{n}"]), flush=True)
    res = pd.concat(rows, ignore_index=True)
    res.to_csv(os.path.join(out, "ringsize_paired.csv"), index=False)
    json.dump(diag, open(os.path.join(out, "ringsize_diag.json"), "w"), indent=2)
    print(res.round(2).to_string(index=False), flush=True)


def part_latency(out, seeds):
    env = ResilientMicrogridEnv()
    tx, tz = calibrate_threshold(env, CALIB)
    pol = ThresholdPolicy(env, MARGIN * tx, MARGIN * tz)
    lat = isolation_latency(env, pol, ATTACKS[1:], n_seeds=seeds)
    ora = isolation_latency(env, OraclePolicy(env), ATTACKS[1:], n_seeds=seeds)
    lat.to_csv(os.path.join(out, "latency_threshold.csv"), index=False)
    ora.to_csv(os.path.join(out, "latency_oracle.csv"), index=False)
    for lab, d in (("threshold", lat), ("oracle", ora)):
        print(lab, flush=True)
        print(d.groupby("attack").agg(isolated=("isolated", "mean"),
                                      latency_ms_median=("latency_s", lambda x: 1000 * x.median()),
                                      latency_ms_max=("latency_s", lambda x: 1000 * x.max()),
                                      false_iso_steps=("false_isolation_steps", "sum")).round(1).to_string(),
              flush=True)


def part_radial(out, seeds):
    """Transfer test: a radial feeder (open electrical ring) with a chain
    consensus graph, the sparsest connected configuration (1-robust; the two
    end units have a single neighbour)."""
    from mgxrl.plant.microgrid import ring_adjacency
    env = ResilientMicrogridEnv(plant_params=MicrogridParams(elec_topology="radial"))
    A = ring_adjacency(env.n)
    A[0, env.n - 1] = A[env.n - 1, 0] = 0.0
    set_comm_graph(env, A)
    nom = sweep(env, None, n_seeds=seeds)
    with use_wmsr(env, 1):
        wm = sweep(env, None, n_seeds=seeds)
    thr, _, tau, ft = threshold_arm(env, seeds)
    ora = sweep(env, OraclePolicy(env), n_seeds=seeds)
    rows = []
    for lab, v in (("nominal", nom), ("wmsr", wm), ("threshold", thr), ("oracle", ora)):
        v.to_csv(os.path.join(out, f"sweep_radial_{lab}.csv"), index=False)
        if lab != "nominal":
            p = paired(nom, v, lab); p["system"] = "radial"; rows.append(p.reset_index())
    diag = {"r_robust": r_robustness(A), "tau": tau,
            "threshold_normal_loss": ft["mean_connectivity_loss"],
            **wmsr_edges_kept(env, n_seeds=seeds)}
    res = pd.concat(rows, ignore_index=True)
    res.to_csv(os.path.join(out, "radial_paired.csv"), index=False)
    json.dump(diag, open(os.path.join(out, "radial_diag.json"), "w"), indent=2)
    print("radial", json.dumps(diag), flush=True)
    print(res.round(2).to_string(index=False), flush=True)


PARTS = {"radial": part_radial, "comms": part_comms, "adaptive": part_adaptive, "graphs": part_graphs,
         "ringsize": part_ringsize, "latency": part_latency}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", nargs="+", choices=list(PARTS), required=True)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--out", default="results/robustness")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for p in a.part:
        print(f"=== {p} ===", flush=True)
        PARTS[p](a.out, a.seeds)


if __name__ == "__main__":
    main()
