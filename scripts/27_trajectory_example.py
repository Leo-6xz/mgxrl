"""Closed-loop trajectories for one representative episode (paper figure).

Runs one paired episode (default: Ramp, seed 1) under the nominal consensus
controller, receiver-side W-MSR (F = 1) and the calibrated innovation-threshold
rule, and records per-DG frequency, active power, the trust weights applied by
the threshold rule, and the fraction of edges W-MSR keeps.

Output: results/baselines/trajectory_<attack>_s<seed>.npz
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from mgxrl.envs.resilient_env import ResilientMicrogridEnv
from mgxrl.eval.baselines import ThresholdPolicy, calibrate_threshold, use_wmsr
from mgxrl.eval.benchmarks import make_injector


def run(env, attack, seed, policy=None):
    rng = np.random.default_rng(1000 + seed)
    inj = make_injector(env, attack, rng)
    obs, _ = env.reset(seed=seed, options={"injector": inj})
    rec = {k: [] for k in ("t", "f", "P", "w_xi", "kept_xi")}
    for _ in range(env.max_steps):
        a = (np.zeros(env.n_actions, np.float32) if policy is None
             else policy.predict(obs, deterministic=True)[0])
        obs, _, term, trunc, info = env.step(a)
        m = info["raw_measurement"]
        rec["t"].append(m["t"]); rec["f"].append(np.array(m["f"])); rec["P"].append(np.array(m["P"]))
        rec["w_xi"].append(np.clip(1.0 + info["action_norm"][: env.n], 0.0, 1.0))
        rec["kept_xi"].append(getattr(env.mg, "wmsr_kept_xi", 1.0))
        if term or trunc:
            break
    out = {k: np.array(v) for k, v in rec.items()}
    out["targets"] = np.array(sorted(getattr(inj, "targets", [])))
    out["onset"] = float(getattr(inj, "t_start", getattr(inj, "onset", np.nan)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attack", default="Ramp")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--margin", type=float, default=1.5)
    a = ap.parse_args()
    env = ResilientMicrogridEnv()
    tx, tz = calibrate_threshold(env, seeds=range(500, 520))
    pol = ThresholdPolicy(env, a.margin * tx, a.margin * tz)
    res = {"nominal": run(env, a.attack, a.seed)}
    with use_wmsr(env, 1):
        res["wmsr"] = run(env, a.attack, a.seed)
    res["threshold"] = run(env, a.attack, a.seed, pol)
    flat = {f"{arm}_{k}": v for arm, d in res.items() for k, v in d.items()}
    os.makedirs("results/baselines", exist_ok=True)
    path = f"results/baselines/trajectory_{a.attack}_s{a.seed}.npz"
    np.savez_compressed(path, **flat)
    for arm, d in res.items():
        fdev = 1e3 * np.sqrt(((d["f"] - 50.0) ** 2).mean())
        print(f"{arm:10s} f_rms {fdev:8.2f} mHz  targets {d['targets']}  onset {d['onset']:.3f}  steps {len(d['t'])}")
    print("wrote", path)


if __name__ == "__main__":
    main()
