#!/usr/bin/env python3
"""Generate a labeled detector corpus from the reconstructed plant.

WHY THIS EXISTS
The IEEE Dataport release contains 500,001 samples per scenario, which sounds
substantial. It is 1.00 second of simulated time, one trajectory per class --
7 seconds in total. The 3.5 million rows are that same 7 seconds sampled every
2 microseconds.

Sampling rate does not create information. At any physically meaningful rate the
release is tiny:

    2 us   3,500,007 rows   <- the published protocol
    1 ms       7,000 rows
    10 ms        700 rows   <- ~31 attacked samples per attack class

An intrusion detector cannot be honestly evaluated on ~31 attacked samples per
class drawn from a single trajectory, at any sampling rate. Nor can a
scenario-level split be performed: holding out a run requires more than one run.
This is the structural reason the published 99.72% is optimistic, and it is not
fixable by better modeling.

WHAT THIS DOES
Generates many independent trajectories from the reconstructed plant, with
randomised attack onset, severity, target DGs, load profile, and benign load
steps. Output is written in the EXACT folder and column format of the Dataport
release, so `load_all()`, `01_detector_benchmark.py`, and every split protocol
work unchanged -- and `scenario_split` becomes available for the first time.

USAGE
    python scripts/09_generate_detector_dataset.py --out data/synthetic --runs 40
    python scripts/01_detector_benchmark.py --root data/synthetic --decimate 1
"""
import argparse
import os
import sys
import zlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from mgxrl.plant.microgrid import Microgrid, MicrogridParams
from mgxrl.plant.attacks import ATTACK_REGISTRY

HEADER = (["time", "V1", "V2", "V3", "I1", "I2", "I3"]
          + [f"{q}DG{i+1}" for i in range(10) for q in ("P", "Q", "f")]
          + ["label"])


def generate_run(scenario, seed, seconds, log_dt, rng):
    """One independent trajectory. Everything that can vary, varies."""
    n_dg = 10
    targets = tuple(rng.choice(
        n_dg, size=1 if scenario != "Stealth" else int(rng.integers(2, 5)),
        replace=False).tolist())
    t_start = float(rng.uniform(0.25, 0.70) * seconds) if scenario != "Normal" else 0.0

    inj = ATTACK_REGISTRY[scenario](targets=targets, t_start=t_start)

    # Severity randomization. Without it the detector learns one amplitude and
    # fails on anything else -- and the same fixed onset would let it learn the
    # clock instead of the signal.
    if scenario != "Normal":
        scale = float(rng.uniform(0.4, 1.8))
        for attr in ("bias_xi", "bias_zeta", "slope", "amp", "amp_xi", "amp_zeta"):
            if hasattr(inj, attr):
                setattr(inj, attr, getattr(inj, attr) * scale)
        if hasattr(inj, "freq"):
            inj.freq = float(inj.freq * rng.uniform(0.5, 2.0))
    inj.reset()

    pp = MicrogridParams(seed=seed, control_dt=log_dt)
    mg = Microgrid(pp)
    mg.attach_attack(inj)
    mg.reset(load_scale=rng.uniform(0.85, 1.15, size=n_dg), settle_time=3.0)

    # Benign load steps. These are what stop the detector learning
    # "any transient == attack", and they are why its false-positive rate on
    # ordinary switching is measurable at all.
    n_steps = int(rng.integers(1, 4))
    step_times = np.sort(rng.uniform(0.1, seconds * 0.9, size=n_steps))
    scale = mg.load_scale.copy()

    rows, n_log = [], int(round(seconds / log_dt))
    si = 0
    for _ in range(n_log):
        t = mg.state.t
        while si < len(step_times) and t >= step_times[si]:
            bus = int(rng.integers(0, n_dg))
            scale[bus] *= 1.0 + rng.uniform(-0.20, 0.20)
            mg.apply_load_step(scale)
            si += 1

        m = mg.measure(noisy=True)
        y_bin, _ = inj.label(t)
        rows.append(np.concatenate([mg.to_dataset_row(m), [y_bin]]))
        mg.step()
        if m["collapsed"]:
            break

    return pd.DataFrame(rows, columns=HEADER)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--runs", type=int, default=40, help="runs per scenario")
    ap.add_argument("--seconds", type=float, default=8.0, help="per run")
    ap.add_argument("--log-dt", type=float, default=0.001, help="1 ms = 1 kHz")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    total_rows = 0

    print(f"Generating {a.runs} runs x 7 scenarios x {a.seconds}s @ {1/a.log_dt:.0f} Hz")
    print(f"  -> {a.runs * 7 * a.seconds:.0f} s of simulated time "
          f"({a.runs * 7 * a.seconds / 7:.0f}x the Dataport release)\n")

    for scenario in ATTACK_REGISTRY:
        folder = os.path.join(a.out, scenario)
        os.makedirs(folder, exist_ok=True)
        n_attacked = 0
        for r in range(a.runs):
            # zlib.crc32, not hash(): str hashing is salted per process unless
            # PYTHONHASHSEED is set, so the previous line produced a DIFFERENT
            # corpus on every invocation. The headline synthetic-corpus result
            # was not reproducible, including by you.
            df = generate_run(scenario,
                              seed=a.seed + 1000 * r + zlib.crc32(scenario.encode()) % 97,
                              seconds=a.seconds, log_dt=a.log_dt, rng=rng)
            df.to_csv(os.path.join(folder, f"run_{r:03d}.csv"), index=False)
            n_attacked += int(df["label"].sum())
            total_rows += len(df)
        print(f"  {scenario:<10} {a.runs} runs, {n_attacked:>7,} attacked rows")

    print(f"\nTotal: {total_rows:,} rows in {a.out}/")
    print("\nNext:")
    print(f"  python scripts/01_detector_benchmark.py --root {a.out} --decimate 1 --windowed")
    print("\nWith multiple runs per scenario, `scenario_split` now works -- that is")
    print("the honest protocol, and it was impossible on the release itself.")


if __name__ == "__main__":
    main()
