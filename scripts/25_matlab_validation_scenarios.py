#!/usr/bin/env python3
"""Scenarios and Python reference results for the MATLAB validation of the
innovation-threshold rule.

Writes, under --out:
  scenarios.csv        attack class, seed, targets (1-based), onset and the
                       sampled severity parameters, exactly as the Python sweep
                       draws them (same keying, seed -> rng 1000 + seed)
  thresholds.json      the rule's thresholds, calibrated on the Python plant
  python_metrics.csv   nominal and threshold arms on the Python plant under the
                       SAME conditions MATLAB uses: nominal loads, no load
                       steps, no measurement noise

    python scripts\\25_matlab_validation_scenarios.py
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from mgxrl.envs.resilient_env import ResilientMicrogridEnv, EnvConfig
from mgxrl.plant.microgrid import MicrogridParams
from mgxrl.eval.benchmarks import make_injector
from mgxrl.eval.baselines import calibrate_threshold, ThresholdPolicy

CLASSES = ["Additive", "Ramp", "SlowRamp", "Sinusoid", "Stealth", "DoS"]
PARAMS = ["bias_xi", "bias_zeta", "slope", "amp_xi", "amp_zeta", "freq", "amp", "alpha"]


def episode_metrics(t, f, P, onset):
    """Identical definition in validate_threshold_rule.m."""
    post = t >= onset
    fdev = f[post] - 50.0
    share = P[post].std(axis=1, ddof=1) / (np.abs(P[post]).mean(axis=1) + 1e-9)
    return {"f_rms_mHz": 1e3 * float(np.sqrt((fdev ** 2).mean())),
            "f_max_mHz": 1e3 * float(np.abs(fdev).max()),
            "p_share": float(share.mean())}


def run(env, policy, inj, seed):
    obs, _ = env.reset(seed=seed, options={"injector": inj})
    T, F, P, TR = [], [], [], []
    for _ in range(env.max_steps):
        a = np.zeros(env.n_actions, np.float32) if policy is None else policy.predict(obs)[0]
        obs, _, term, trunc, info = env.step(a)
        m = info["raw_measurement"]
        T.append(m["t"]); F.append(m["f"]); P.append(m["P"])
        TR.append(np.clip(1.0 + np.asarray(info["action_norm"])[: env.n], 0, 1))
        if term or trunc:
            break
    return np.array(T), np.array(F), np.array(P), np.array(TR), bool(term)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out", default="results/matlab_validation")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    # Thresholds from the standard (noisy, load-stepped) environment, as in the paper.
    std_env = ResilientMicrogridEnv()
    tx, tz = calibrate_threshold(std_env, range(500, 520))
    tau = {"tau_xi": 1.5 * tx, "tau_zeta": 1.5 * tz, "margin": 1.5,
           "xi_limit": std_env.pp.xi_limit, "zeta_limit": std_env.pp.zeta_limit,
           "rate_max": std_env.cfg.residual_rate_max,
           "min_trusted": std_env.cfg.min_trusted_links, "trust_ok": std_env.cfg.trust_ok}
    json.dump(tau, open(os.path.join(a.out, "thresholds.json"), "w"), indent=2)

    # Like-for-like Python plant: what the MATLAB reference model simulates.
    pp = MicrogridParams(noise_v=0.0, noise_p=0.0, noise_q=0.0, noise_f=0.0)
    env = ResilientMicrogridEnv(config=EnvConfig(randomize_load=False, n_load_steps=0),
                                plant_params=pp)
    pol = ThresholdPolicy(env, tau["tau_xi"], tau["tau_zeta"])

    scen, rows = [], []
    for name in CLASSES:
        for s in range(a.seeds):
            inj = make_injector(env, name, np.random.default_rng(1000 + s))
            rec = {"attack": name, "seed": s, "t_start": inj.t_start,
                   "targets": ";".join(str(j + 1) for j in inj.targets)}
            for k in PARAMS:
                rec[k] = getattr(inj, k, np.nan)
            scen.append(rec)
            for arm, p in (("nominal", None), ("threshold", pol)):
                inj2 = make_injector(env, name, np.random.default_rng(1000 + s))
                t, f, P, tr, col = run(env, p, inj2, s)
                iso = (tr[:, list(inj2.targets)] < 0.5).all(axis=1) & (t >= inj2.t_start)
                rows.append({"attack": name, "seed": s, "arm": arm, "collapsed": col,
                             **episode_metrics(t, f, P, inj2.t_start),
                             "latency_ms": 1e3 * (t[iso][0] - inj2.t_start) if iso.any() else np.nan})
        print(f"  {name} done", flush=True)
    pd.DataFrame(scen).to_csv(os.path.join(a.out, "scenarios.csv"), index=False)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "python_metrics.csv"), index=False)
    print(json.dumps(tau, indent=2))


if __name__ == "__main__":
    main()
