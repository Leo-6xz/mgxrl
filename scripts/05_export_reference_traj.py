#!/usr/bin/env python3
"""Export reduced-order plant trajectories for MATLAB fidelity comparison.

Column layout consumed by matlab/validate_fidelity.m:
    t, V(1..10), f(1..10), P(1..10), Q(1..10)

Trajectories are logged with noise DISABLED. Fidelity is a property of the
deterministic dynamics; comparing noisy realizations just measures the RNG and
adds a floor to every RMSE that has nothing to do with the model.
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from mgxrl.plant.microgrid import Microgrid, MicrogridParams
from mgxrl.plant.attacks import ATTACK_REGISTRY


def export(name, T=12.0, t_start=1.5, settle=3.0, seed=0):
    targets = (0,) if name != "Stealth" else (0, 1, 2)
    inj = ATTACK_REGISTRY[name](targets=targets, t_start=t_start)
    if name == "Stealth":
        inj.freq = 0.7
    mg = Microgrid(MicrogridParams(seed=seed))
    mg.attach_attack(inj)
    mg.reset(settle_time=settle)          # deterministic: no load randomisation

    rows, n = [], int(round(T / mg.p.control_dt)) + 1
    for _ in range(n):
        m = mg.measure(noisy=False)
        rows.append(np.concatenate([[m["t"]], m["V"], m["f"], m["P"], m["Q"]]))
        mg.step()
    return np.asarray(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--attack", default=None, help="single class, default all")
    ap.add_argument("--seconds", type=float, default=12.0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    names = [a.attack] if a.attack else list(ATTACK_REGISTRY)
    for name in names:
        arr = export(name, T=a.seconds)
        path = os.path.join(a.out, f"reduced_{name}.csv")
        np.savetxt(path, arr, delimiter=",")
        print(f"{name:<10} {arr.shape}  -> {path}")


if __name__ == "__main__":
    main()
