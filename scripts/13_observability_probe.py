#!/usr/bin/env python3
"""
13_observability_probe.py -- can the attacked source be identified from the
observation the policy actually receives?

WHY THIS EXISTS
The action space assigns per-source trust: a[j] scales DG j's transmitted
consensus signal for every receiver. To use it, the policy must work out WHICH
source is lying. This probe asks whether that is possible at all, before any
compute is spent on training. It is the third of three pre-training tests:

    test_oracle_action_can_cancel_an_attack          -- expressiveness
    test_exploration_is_cheap_when_nothing_is_wrong  -- affordability
    this                                             -- observability

All three fail silently and look identical from the training curve.

WHAT IT MEASURES
Per attack class, roll out the UNDEFENDED controller (zero action) with the
target DG assigned round-robin, and fit a classifier from features -> target DG
index. Two feature sets:

    base       V, f, P, Q, xi, zeta -- what the observation held before the
               consensus innovation was added
    base+inn   the same, plus inn[j] = mean_i adj[i,j] * (x_i - x_rx[j])

The innovation is computable at each receiving DG from locally available data,
so it gives the policy nothing a deployed controller could not have. It is the
statistic W-MSR thresholds on.

CORRECTNESS NOTES -- read before trusting any number this prints
1. Episode-level splitting, never sample-level. Consecutive control steps are
   near-identical; a random split reports retrieval, not identification. Same
   leakage documented in FINDINGS.md for the Dataport release.
2. The split is STRATIFIED BY TARGET. An earlier version shuffled episodes
   without stratifying. With a fixed number of episodes per target that makes
   train-heavy classes test-light by construction, and the anticorrelation
   drove measured accuracy BELOW chance (Stealth 0.047, Sinusoid 0.010 against
   a chance level of 0.100) while making the trivial-predictor baseline exactly
   0.000, because the modal training class had no test episodes at all. Those
   numbers were artifacts of the split. Do not quote them.
3. The reference level is the PERMUTATION null, not 1/n. Targets are shuffled
   between episodes and the whole fit repeated; that null absorbs residual
   structure in the split, the class balance, and the classifier's own bias.
   A result counts only if it clears its own null by several standard
   deviations.
4. --stride decimates in time. Consecutive samples are near-duplicates, so this
   costs little information and a lot of runtime. Lower it to 1 to confirm a
   borderline result.

USAGE
  python scripts/13_observability_probe.py --episodes 60 --out results
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mgxrl.envs.resilient_env import ResilientMicrogridEnv, EnvConfig  # noqa: E402
from mgxrl.plant.attacks import ATTACK_REGISTRY                        # noqa: E402


# ---------------------------------------------------------------- features

def features(env, meas):
    """(base, innovation) built from the raw measurement, not from _observe.

    Keeping the probe independent of the observation builder means this
    comparison stays valid after the innovation is added to the observation --
    otherwise the two feature sets silently become the same thing.
    """
    pp = env.pp
    base = np.concatenate([
        meas["V"] - pp.v_nom,
        (meas["f"] - pp.f_nom) * 10.0,
        meas["P"] - pp.p_set,
        meas["Q"] - pp.q_set,
        meas["xi"] / pp.xi_limit,
        meas["zeta"] / pp.zeta_limit,
    ])
    adj = env.mg.adj
    deg = adj.sum(axis=0)
    deg_safe = np.where(deg > 0, deg, 1.0)
    inn_xi = (adj.T @ meas["xi"] - deg * meas["xi_rx"]) / deg_safe
    inn_zeta = (adj.T @ meas["zeta"] - deg * meas["zeta_rx"]) / deg_safe
    inn = np.concatenate([inn_xi / pp.xi_limit, inn_zeta / pp.zeta_limit])
    return base, inn


def episode_targets(env, attack_name, primary):
    """Targets for one episode, matching what sample_attack does in training.

    Coordinated Stealth is a MULTI-source attack: sample_attack gives it 2-4
    targets, and the class spreads a correlated signal across them with a
    per-target phase. Probing it with a single target -- which the first
    version did -- asks a question the attack is not defined for and makes the
    Stealth row unreadable in either direction. The primary target is still the
    training label; a prediction counts as correct if it names ANY compromised
    source, which is what the trust weights actually need.
    """
    if attack_name != "Stealth":
        return (primary,)
    n = env.n
    return tuple(sorted({primary, (primary + n // 3) % n, (primary + 2 * n // 3) % n}))


def collect(env, attack_name, n_episodes, stride, rng):
    """Undefended rollouts, primary targets assigned round-robin."""
    B, I, Y, E, S = [], [], [], [], []
    zero = np.zeros(env.n_actions, np.float32)
    for ep in range(n_episodes):
        target = ep % env.n
        tgts = episode_targets(env, attack_name, target)
        member = np.zeros(env.n, np.uint8)
        member[list(tgts)] = 1
        inj = ATTACK_REGISTRY[attack_name](
            targets=tgts, t_start=float(rng.uniform(0.5, 3.0))
        )
        env.reset(seed=int(rng.integers(0, 10_000)), options={"injector": inj})
        k = 0
        for _ in range(env.max_steps):
            _, _, term, trunc, info = env.step(zero)
            if info["y_bin"] == 1:
                if k % stride == 0:
                    b, i = features(env, info["raw_measurement"])
                    B.append(b)
                    I.append(i)
                    Y.append(target)
                    E.append(ep)
                    S.append(member)
                k += 1
            if term or trunc:
                break
    return (np.asarray(B, np.float32), np.asarray(I, np.float32),
            np.asarray(Y), np.asarray(E), np.asarray(S, np.uint8))


# ---------------------------------------------------------------- splitting

def stratified_episode_split(ep, y, n_classes, test_frac, seed):
    """Hold out the same fraction of episodes from EVERY target class.

    Stratifying is what stops a class being absent from one side of the split,
    which is what produced below-chance accuracies and a zero trivial baseline
    in the first version of this probe.
    """
    rng = np.random.default_rng(seed)
    ep_target = {int(e): int(y[ep == e][0]) for e in np.unique(ep)}
    test_eps = []
    for c in range(n_classes):
        eps_c = np.array([e for e, t in ep_target.items() if t == c])
        if len(eps_c) == 0:
            continue
        rng.shuffle(eps_c)
        k = max(1, int(round(test_frac * len(eps_c))))
        test_eps.extend(eps_c[:k].tolist())
    te = np.isin(ep, test_eps)
    return ~te, te


def fit_once(X, y, tr, te, n_classes, member=None):
    """Top-1 accuracy. `member` marks every compromised source per sample, so a
    multi-target class scores a hit for naming any of them."""
    import lightgbm as lgb

    if te.sum() == 0 or len(np.unique(y[tr])) < 2:
        return float("nan")
    clf = lgb.LGBMClassifier(
        n_estimators=200, num_leaves=31, learning_rate=0.1,
        objective="multiclass", num_class=n_classes, verbose=-1,
    )
    clf.fit(X[tr], y[tr])
    pred = clf.predict(X[te])
    if member is None:
        return float((pred == y[te]).mean())
    m = member[te]
    return float(m[np.arange(len(pred)), pred].mean())


def evaluate(X, y, ep, n_classes, repeats, test_frac, seed, permute_repeats,
             member=None):
    """Real accuracy and the permutation null, both over repeated splits.

    THE NULL IS SCORED UNDER THE SAME RULE AS THE REAL FIT. It previously was
    not: `member` was passed to the real fit and omitted from the null, so for
    a multi-target class the real accuracy counted naming ANY compromised
    source while the null counted exact match on one label. For every
    single-target class the two rules coincide and nothing changed, but for
    Coordinated Stealth -- three targets of ten -- a uniform guesser scores
    ~0.30 under membership and ~0.10 under exact match. That reported Stealth
    at 0.370 against a null of 0.067 and called it 6.4 sigma, when the honest
    comparison is 0.370 against ~0.30. The permutation now relabels whole
    episodes TOGETHER WITH their target sets, so the null inherits the same
    scoring geometry.
    """
    real, null, trivial = [], [], []
    for r in range(repeats):
        tr, te = stratified_episode_split(ep, y, n_classes, test_frac, seed + r)
        real.append(fit_once(X, y, tr, te, n_classes, member))
        trivial.append(_trivial_rate(y, te, n_classes, member))

    rng = np.random.default_rng(seed + 9999)
    eps = np.unique(ep)
    labels = np.array([int(y[ep == e][0]) for e in eps])
    ep_index = {int(e): i for i, e in enumerate(eps)}
    for r in range(permute_repeats):
        order = rng.permutation(len(eps))
        y_perm = np.zeros_like(y)
        member_perm = None if member is None else np.zeros_like(member)
        for i, e in enumerate(eps):
            src = eps[order[i]]
            m = ep == e
            y_perm[m] = labels[order[i]]
            if member is not None:
                # carry the SET, not just the label, or the null is scored
                # under a different rule from the measurement
                member_perm[m] = member[ep == src][0]
        tr, te = stratified_episode_split(ep, y_perm, n_classes, test_frac,
                                          seed + r)
        null.append(fit_once(X, y_perm, tr, te, n_classes, member_perm))

    return {
        "acc": float(np.nanmean(real)), "acc_std": float(np.nanstd(real)),
        "null": float(np.nanmean(null)), "null_std": float(np.nanstd(null)),
        "trivial": float(np.mean(trivial)),
    }


def _trivial_rate(y, te, n_classes, member=None):
    """Best a constant predictor could do on this test fold, under the same
    scoring rule as the measurement."""
    if te.sum() == 0:
        return float("nan")
    if member is None:
        counts = np.bincount(y[te], minlength=n_classes)
        return float(counts.max() / max(1, counts.sum()))
    m = member[te]
    return float(m.mean(axis=0).max())


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--permute-repeats", type=int, default=2)
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--attacks", nargs="*", default=None)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    env = ResilientMicrogridEnv(config=EnvConfig(seed=a.seed))
    names = a.attacks or [k for k in ATTACK_REGISTRY if k.lower() != "normal"]

    if a.episodes % env.n != 0:
        print(f"WARNING: --episodes {a.episodes} is not a multiple of n={env.n}; "
              f"targets will be unevenly represented.")
    if a.episodes < 5 * env.n:
        print(f"WARNING: fewer than {5 * env.n} episodes gives too few episodes "
              f"per target for a stable split. Results will be noisy.")

    rows = []
    for name in names:
        rng = np.random.default_rng(a.seed)
        B, I, Y, E, S = collect(env, name, a.episodes, a.stride, rng)
        if len(B) == 0:
            print(f"  {name}: no attacked samples, skipped")
            continue
        rb = evaluate(B, Y, E, env.n, a.repeats, a.test_frac, a.seed,
                      a.permute_repeats, member=S)
        ri = evaluate(np.hstack([B, I]), Y, E, env.n, a.repeats, a.test_frac,
                      a.seed, a.permute_repeats, member=S)
        # Reference is max(permutation null, trivial constant predictor). The
        # permutation null can fall BELOW what a constant predictor achieves --
        # measured on Stealth, null 0.234 against trivial 0.314 -- and scoring
        # against the null alone then flatters the result: 0.370 reads as 2.9
        # sigma over 0.234 but only 1.2 over 0.314. Always take the harder.
        ref = max(ri["null"], ri["trivial"])
        margin = (ri["acc"] - ref) / max(ri["acc_std"], 1e-6)
        # Normalised lift makes classes with different floors comparable. A
        # 3-of-10 membership task (Coordinated Stealth) has a floor near 0.30;
        # a single-target class has a floor near 0.10.
        headroom = (ri["acc"] - ref) / max(1.0 - ref, 1e-6)
        rows.append({
            "attack": name, "n_samples": len(B),
            "base": rb["acc"], "base_std": rb["acc_std"], "base_null": rb["null"],
            "base_inn": ri["acc"], "base_inn_std": ri["acc_std"],
            "base_inn_null": ri["null"], "trivial": ri["trivial"],
            "reference": ref,
            "lift_inn": ri["acc"] - rb["acc"],
            "lift_over_reference_norm": headroom,
            "sigma_over_reference": margin,
        })
        print(f"  {name:<10} base {rb['acc']:.3f}+/-{rb['acc_std']:.3f}   "
              f"base+inn {ri['acc']:.3f}+/-{ri['acc_std']:.3f}   "
              f"ref {ref:.3f} (null {ri['null']:.3f}, trivial {ri['trivial']:.3f})"
              f"   {margin:6.1f} sigma   headroom {headroom:+.2f}")

    df = pd.DataFrame(rows)
    os.makedirs(a.out, exist_ok=True)
    df.to_csv(os.path.join(a.out, "observability.csv"), index=False)

    print("\n" + "-" * 78)
    print(df.to_string(index=False))
    print("-" * 78)
    print(
        "\nReading it. Compare each accuracy with its REFERENCE column -- the\n"
        "harder of the permutation null and the best constant predictor -- not\n"
        "with 1/n. Treat anything under about 3 sigma as not measured.\n"
        "`headroom` is (acc - ref)/(1 - ref), which makes classes with\n"
        "different floors comparable: a multi-target class scored by set\n"
        "membership starts near 0.30, a single-target class near 0.10.\n"
        "  base+inn clears the null       -> the attacked source is identifiable\n"
        "                                    from locally available data. Per-source\n"
        "                                    trust is a usable action space for that\n"
        "                                    class, and the innovation belongs in the\n"
        "                                    observation.\n"
        "  base+inn sits at the null      -> not identifiable per-step. No reward or\n"
        "                                    exploration change fixes that; report the\n"
        "                                    class as out of scope rather than claim\n"
        "                                    coverage.\n"
        "  base already clears the null   -> observable without the innovation;\n"
        "                                    expect the agent to already handle it.\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
