#!/usr/bin/env python3
"""Calibrate reward reference scales, and verify every attack class is learnable.

Two checks, both of which must pass before training:

  CALIBRATION   ref_x := 90th percentile of metric x over Normal episodes.
                Makes each term ~1.0 during healthy operation, so all terms are
                commensurable and alive_bonus has a principled value.

  SEPARATION    every attack class must degrade the undefended return by a
                clear margin, and no reward term may be saturated. A class with
                no margin gives the agent no gradient; it will be ignored during
                training while the aggregate reward curve looks perfectly
                healthy. That failure is invisible unless you check for it here.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from mgxrl.envs.resilient_env import ResilientMicrogridEnv, EnvConfig, RewardWeights
from mgxrl.plant.attacks import ATTACK_REGISTRY, NoAttack

METRICS = ("f_rmse", "v_rmse", "p_sharing_error", "q_sharing_error",
           "xi_spread_sustained", "zeta_spread_sustained")

REF_ATTR = {
    "f_rmse": "ref_f_rmse",
    "v_rmse": "ref_v_rmse",
    "p_sharing_error": "ref_p_share",
    "q_sharing_error": "ref_q_share",
    "xi_spread_sustained": "ref_xi_spread",
    "zeta_spread_sustained": "ref_zeta_spread",
}

_missing = [a for a in REF_ATTR.values() if not hasattr(RewardWeights(), a)]
assert not _missing, (
    f"REF_ATTR names fields that RewardWeights does not have: {_missing}. "
    f"Fix the mapping rather than the assertion -- a wrong name here means "
    f"the calibration silently fails to reach the reward."
)

def calibrate(env, n_episodes=12, percentile=90):
    acc = {m: [] for m in METRICS}
    for s in range(n_episodes):
        for row in env.run_nominal(injector=NoAttack(), seed=s):
            for m in METRICS:
                acc[m].append(row[m])
    return {m: float(np.percentile(acc[m], percentile)) for m in METRICS}


def separation(env, n_seeds=10):
    out = {}
    for name in ATTACK_REGISTRY:
        rets, terms = [], None
        for s in range(n_seeds):
            targets = (0,) if name != "Stealth" else (0, 1, 2)
            inj = ATTACK_REGISTRY[name](targets=targets, t_start=1.5)
            log = env.run_nominal(injector=inj, seed=s)
            rets.append(sum(r["reward"] for r in log))
            mm = {m: float(np.mean([r[m] for r in log])) for m in METRICS}
            b = env.reward_breakdown(mm)
            terms = b if terms is None else {k: terms[k] + b[k] for k in b}
        out[name] = {"return_mean": float(np.mean(rets)),
                     "return_std": float(np.std(rets, ddof=1) if len(rets) > 1 else 0.0),
                     "n_seeds": int(n_seeds),
                     "terms": {k: v / n_seeds for k, v in terms.items()}}
    return out


def report_separation(sep, label):
    """Print one separation table. Returns (ok, margins)."""
    base = sep["Normal"]["return_mean"]
    print(f"\n[{label}]")
    print(f"{'attack':<10}{'return':>12}{'std':>10}{'vs Normal':>12}"
          f"{'margin':>9}{'gap/SE':>8}{'max term':>10}  verdict")
    ok, margins, weak = True, {}, []
    n = sep["Normal"].get("n_seeds", 10)
    for name, r in sep.items():
        mx = max(r["terms"].values())
        gap = r["return_mean"] - base

        se = (r["return_std"] / max(n, 1) ** 0.5) or float("nan")
        t = abs(gap) / se if se and se == se and se > 0 else float("nan")
        if name == "Normal":
            margin, t = 0.0, float("nan")
            verdict = "baseline" if r["return_mean"] > 0 else "FAIL: healthy op is penalised"
        else:
            margin = abs(gap) / abs(base)
            if margin <= 0.25:
                verdict = "FAIL: no gradient on this class"
            elif not (t > 2.0):
                verdict = "ok BUT WITHIN NOISE"
                weak.append(name)
            else:
                verdict = "ok"
        margins[name] = margin
        ok &= not verdict.startswith("FAIL")
        print(f"{name:<10}{r['return_mean']:>12.0f}{r['return_std']:>10.0f}"
              f"{gap:>12.0f}{margin:>9.2%}{t:>8.1f}{mx:>10.2f}  {verdict}")
    if weak:
        print(f"  WITHIN NOISE: {', '.join(weak)} -- the margin clears 25% on the")
        print("  mean but the gap is under two standard errors, so it is not")
        print("  distinguishable from no separation at this seed count.")
    worst = max(max(r["terms"].values()) for r in sep.values())
    print(f"  largest term across all classes: {worst:.2f}"
          f"  {'ok' if worst < 40 else 'WARNING: consider raising a reference scale'}")
    return ok, margins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--percentile", type=float, default=90)
    ap.add_argument("--out", default="results")
    ap.add_argument("--sep-seeds", type=int, default=10,
                    help="seeds per class in the separation check. 5 does "
                         "not resolve DoS, whose per-seed std is larger "
                         "than its gap from Normal.")
    ap.add_argument("--no-compare", action="store_true",
                    help="skip the second separation check against the "
                         "measured references")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    env = ResilientMicrogridEnv(config=EnvConfig(seed=0))

    print("Calibrating reference scales on Normal episodes ...")
    refs = calibrate(env, a.episodes, a.percentile)
    print(f"\n{'metric':<24}{'p'+str(int(a.percentile)):>12}   -> RewardWeights field")
    for m, v in refs.items():
        print(f"{m:<24}{v:>12.3e}   {REF_ATTR[m]}")

    w = RewardWeights()
    print(f"\n{'metric':<24}{'measured':>12}{'in use':>12}{'ratio':>8}")
    stale = []
    for m, v in refs.items():
        cur = float(getattr(w, REF_ATTR[m]))
        ratio = v / cur if cur else float("inf")
        flag = "  <- STALE" if not (1 / 1.5 <= ratio <= 1.5) else ""
        if flag:
            stale.append((REF_ATTR[m], cur, v))
        print(f"{m:<24}{v:>12.3e}{cur:>12.3e}{ratio:>8.2f}{flag}")
    if stale:
        print("\n  The reward is running on reference scales that no longer match")
        print("  the plant. Each term is (metric/ref)^2 inside log1p, so a ref that")
        print("  is 2x too small makes that term read ~4x too large during HEALTHY")
        print("  operation and eats into the alive bonus. Update RewardWeights in")
        print("  src/mgxrl/envs/resilient_env.py, then RE-RUN this script: the")
        print("  separation margins below depend on these values, and raising a")
        print("  reference shrinks the margin on whichever class that term carries.")
        for attr, cur, new_v in stale:
            print(f"      {attr}: {cur:.3e} -> {new_v:.3e}")

    print("\nSeparation check (undefended baseline), references AS CONFIGURED ...")
    sep = separation(env, a.sep_seeds)
    ok, margins = report_separation(sep, "references in use")

    # Rerunning with the measured references answers the only question that
    # matters here -- does every class still have a learning signal once the
    # scales are correct -- without editing source and guessing. Raising a
    # reference reduces every term, but by ~2*ln(r_new/r_old) for a metric far
    # above it and by less for one sitting near it, so attacked classes gain
    # more than Normal does and EVERY margin shrinks. The class with the
    # smallest margin is the one that decides this.
    sep2, margins2, ok2 = None, None, None
    if stale and not a.no_compare:
        print("\nSeparation check with the MEASURED references ...")
        proposed = RewardWeights(**{REF_ATTR[m]: v for m, v in refs.items()})
        env2 = ResilientMicrogridEnv(config=EnvConfig(seed=0), weights=proposed)
        sep2 = separation(env2, a.sep_seeds)
        ok2, margins2 = report_separation(sep2, "measured references")

        print("\nMargin change per class:")
        print(f"{'attack':<10}{'in use':>10}{'measured':>10}{'change':>10}")
        for name in margins:
            if name == "Normal":
                continue
            d = margins2[name] - margins[name]
            print(f"{name:<10}{margins[name]:>10.2%}{margins2[name]:>10.2%}"
                  f"{d:>+10.2%}")

        if ok2:
            print("\n  Every class keeps its margin under the measured references.")
            print("  Update RewardWeights in src/mgxrl/envs/resilient_env.py:")
            for attr, cur, new_v in stale:
                print(f"      {attr}: {cur:.3e} -> {new_v:.3e}")
            print("  then re-run this script to confirm, and RETRAIN -- the reward")
            print("  has changed, so existing checkpoints are not comparable.")
        else:
            print("\n  A class loses its learning signal under the CORRECT scales.")
            print("  That is a result, not an obstacle: its apparent separation was")
            print("  resting on an under-scaled reference. Three options, in the")
            print("  order:")
            print("    1. Lengthen consensus_tau. It exists to separate SUSTAINED")
            print("       disagreement (DoS) from benign load-step transients, which")
            print("       is exactly the confound that inflated the p90 here.")
            print("    2. Raise the weight on the affected term. This is tuning to a")
            print("       threshold, and it amplifies the term during healthy")
            print("       operation too.")
            print("    3. Report the class as weakly separated. If the")
            print("       agent has never improved on it in evaluation, this is the")
            print("       explanation, and it is more useful than a passing check.")

    json.dump({"references_measured": refs,
               "references_in_use": {REF_ATTR[m]: float(getattr(w, REF_ATTR[m]))
                                     for m in METRICS},
               "separation_in_use": sep,
               "separation_measured_refs": sep2,
               "stale": [a_ for a_, _, _ in stale]},
              open(os.path.join(a.out, "reward_calibration.json"), "w"), indent=2)
    print(f"\nWrote {a.out}/reward_calibration.json")
    print("\nOVERALL (references in use):",
          "PASS -- safe to train" if ok else "FAIL -- fix before training")
    if ok2 is not None:
        print("OVERALL (measured references):",
              "PASS" if ok2 else "FAIL -- decide before adopting them")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
