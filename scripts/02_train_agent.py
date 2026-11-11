#!/usr/bin/env python3
"""Stage 2: this trains the residual PI-RL agent. ~35 min per seed at 8 workers."""
import argparse, os, pickle, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from mgxrl.agents.train_ppo import TrainConfig, train


def load_detector(path: str):
    """Load a trained classifier and check it before 35 minutes of training.

    Every check here exists because the failure it catches is silent at
    runtime. The env now raises on a bad predict_proba rather than returning
    zeros, but a crash 30 seconds into a SubprocVecEnv worker is a worse place
    to find out than here.
    """
    if not os.path.exists(path):
        raise SystemExit(f"detector not found: {path}")
    try:
        import joblib
        det = joblib.load(path)
    except Exception:
        with open(path, "rb") as fh:
            det = pickle.load(fh)

    if not hasattr(det, "predict_proba"):
        raise SystemExit(
            f"{path} has no predict_proba. The env calls the sklearn API; a "
            f"raw lightgbm.Booster exposes predict() instead and needs "
            f"wrapping."
        )

    n_feat = getattr(det, "n_features_in_", None)
    print(f"[detector] {path}  expects "
          f"{n_feat if n_feat is not None else 'unknown'} features")
    classes = getattr(det, "classes_", None)
    if classes is not None and len(classes) != 7:
        raise SystemExit(
            f"detector has {len(classes)} classes, expected 7 "
            f"(Normal + 6 attack classes). obs_dim assumes 7."
        )
    return det, n_feat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=2_000_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--algo", default="ppo", choices=["ppo", "sac"])
    ap.add_argument("--outdir", default="runs/ppo")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--detector", default=None,
                    help="Path to a trained 7-class detector. Supplying one "
                         "sets EnvConfig.include_detector and appends the "
                         "class posteriors to the observation, taking obs_dim "
                         "from 320 to 327. That INVALIDATES every existing "
                         "checkpoint -- runs/ppo2_s0..2 were all trained at "
                         "320 -- so this starts a new run, it does not "
                         "continue an old one. Train the detector on the "
                         "SYNTHETIC corpus (scripts/09), not on the release: "
                         "the release fills V1-V3 with three-phase voltage at "
                         "one point while to_dataset_row writes three DG "
                         "voltages there, so a release-trained booster reads "
                         "mismatched columns.")
    ap.add_argument("--w-action", type=float, default=None,
                    help="Weight on connectivity loss, mean(1 - trust) -- the "
                         "price of standing distrust during healthy operation, "
                         "and the axis of the nominal-cost / attack-protection "
                         "frontier. Default (None) uses RewardWeights' 1.0. "
                         "Sweep 1, 3, 10 at equal timesteps and plot "
                         "mean_connectivity_loss from false_trigger.json "
                         "against the Additive and Ramp deltas. Note p_normal "
                         "does NOT control this: 0.159 at p=1/7 vs 0.196 at "
                         "p=0.5, both inside the 0.159-0.258 seed spread.")
    a = ap.parse_args()
    if a.w_action is not None:
        print(f"Reward override: w_action = {a.w_action}")

    detector = None
    if a.detector:
        detector, n_feat = load_detector(a.detector)
        # to_dataset_row returns 37 columns (time + 3 V + 3 I + 10 x P,Q,f);
        # the env drops the time column and passes 36. If the detector was
        # trained on engineered features rather than raw columns, this will not
        # match, and the mismatch has to be fixed by putting the SAME transform
        # from mgxrl/detection/features.py in front of predict_proba inside
        # ResilientMicrogridEnv._detector_features -- not by reshaping here.
        if n_feat is not None and n_feat != 36:
            raise SystemExit(
                f"detector expects {n_feat} features but the env passes 36 raw "
                f"columns from to_dataset_row. Wire the features.py transform "
                f"into _detector_features first; feeding it raw columns would "
                f"train the agent on a detector reading the wrong inputs."
            )

    train(TrainConfig(total_timesteps=a.timesteps, n_envs=a.n_envs,
                      algo=a.algo, outdir=a.outdir, seed=a.seed,
                      w_action=a.w_action),
          detector=detector)


if __name__ == "__main__":
    main()
