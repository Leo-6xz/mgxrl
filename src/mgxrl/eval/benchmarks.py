"""Evaluation protocol: what actually goes in the results tables."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..plant.attacks import ATTACK_REGISTRY


def rollout(env, policy=None, injector=None, seed=0):
    """policy=None runs the nominal consensus controller (the primary baseline)."""
    opts = {"injector": injector} if injector is not None else {}
    obs, _ = env.reset(seed=seed, options=opts)
    rows, total = [], 0.0
    for _ in range(env.max_steps):
        a = (np.zeros(env.n_actions, np.float32) if policy is None
             else policy.predict(obs, deterministic=True)[0])
        obs, r, term, trunc, info = env.step(a)
        total += r
        rows.append({k: v for k, v in info.items() if k != "raw_measurement"})
        if term or trunc:
            break
    df = pd.DataFrame(rows)
    return df, {"return": total, "collapsed": bool(term), "steps": len(df)}


def episode_metrics(df: pd.DataFrame, settle_tol=0.02, settle_hold=25) -> dict:
    """Metrics that a power-systems expert will recognise."""
    attacked = df[df["y_bin"] == 1] if "y_bin" in df else df
    m = {
        "v_violation_rate": float(df["v_violation_any"].mean()),
        "f_rmse": float(np.sqrt((df["f_rmse"] ** 2).mean())),
        "v_rmse": float(np.sqrt((df["v_rmse"] ** 2).mean())),
        "p_sharing_error": float(df["p_sharing_error"].mean()),
        "q_sharing_error": float(df["q_sharing_error"].mean()),
        "worst_f_rmse": float(df["f_rmse"].max()),
    }
    if len(attacked):
        m["f_rmse_under_attack"] = float(np.sqrt((attacked["f_rmse"] ** 2).mean()))
        m["p_sharing_under_attack"] = float(attacked["p_sharing_error"].mean())
        m["recovery_steps"] = _recovery_steps(
            (attacked["f_rmse"] < settle_tol).to_numpy(), settle_hold
        )
    return m


def _recovery_steps(ok: np.ndarray, hold: int) -> int:
    """First index after which the signal stays inside tolerance for `hold` steps.

    The previous version returned the first index where `ok` was True, which is
    a single sample and fires on any momentary dip through the tolerance band
    during a still-diverging transient. `hold` steps at 10 ms is 250 ms of
    sustained regulation.
    """
    if not ok.any():
        return -1
    # run-length of consecutive True ending at each index
    run = 0
    for i, good in enumerate(ok):
        run = run + 1 if good else 0
        if run >= hold:
            return int(i - hold + 1)
    return -1


PROTOCOL = "sweep-v2: training attack distribution, paired seeds"


def make_injector(env, name, rng):
    """Build an attack the SAME WAY TRAINING DOES.

    This is the fix for a train/evaluate mismatch that invalidated the previous
    headline table. The old version constructed every class by hand as

        ATTACK_REGISTRY[name](targets=(one_dg,), t_start=U(0.5, 3.0))

    while the environment trains on `sample_attack`, which randomises severity
    and can select MULTIPLE targets, over an onset window of
    (0.5, 0.6 * episode_seconds) = (0.5, 7.2) s.

    Two consequences, both visible in the numbers:

      * Coordinated Stealth and DoS were evaluated single-target -- a strictly
        weaker attack than the one trained against. Against 00_calibrate_reward,
        which uses the training sampler, the undefended baselines disagreed by
        92% on Stealth (-11,477 vs -21,991) and 132% on DoS (+1,137 vs +2,642),
        while the four genuinely single-target classes agreed within 3%. Those
        two classes are exactly where the agent showed no benefit.
      * Onset was confined to the first 3 s of a 12 s episode, so late-onset
        attacks were never evaluated at all.

    Any table produced before this change measures a different threat model
    from the one the agent was trained on and must be regenerated.
    """
    from ..plant.attacks import NoAttack, sample_attack

    if name.lower() == "normal":
        return NoAttack()
    return sample_attack(
        rng,
        n_dg=env.n,
        t_window=(0.5, env.cfg.episode_seconds * 0.6),
        p_normal=0.0,
        subset=(name,),
    )


def sweep(env, policy=None, n_seeds=20, attacks=None, verbose=False):
    """Per-attack-class results with seed variance.

    We report mean +/- std over seeds, not a single run. A single-seed RL result is
    not evidence, and this is the first thing an ML-literate reviewer checks.

    The injector RNG is keyed on the seed alone, so the nominal and agent arms
    see identical attacks on identical seeds. That pairing is what makes
    `paired_delta` valid -- do not change the keying without updating it.
    """
    attacks = attacks or list(ATTACK_REGISTRY)
    out = []
    for name in attacks:
        for s in range(n_seeds):
            rng = np.random.default_rng(1000 + s)
            inj = make_injector(env, name, rng)
            df, info = rollout(env, policy, injector=inj, seed=s)
            out.append({"attack": name, "seed": s, "protocol": PROTOCOL,
                        **episode_metrics(df), **info})
        if verbose:
            print(f"  {name} done")
    return pd.DataFrame(out)


KEYS = ["v_violation_rate", "f_rmse", "p_sharing_error", "return"]


def compare(nominal_df, agent_df, pct_min_snr: float = 1.0) -> pd.DataFrame:
    """Nominal consensus controller vs. the trained agent, per attack class.

    Reports the ABSOLUTE difference alongside the percentage, and suppresses the
    percentage when the baseline is small relative to its own spread across
    seeds. A percentage on a near-zero noisy baseline is not a result: DoS
    nominal return is +1137 +/- 3444, so "-71%" is -811 absolute, well inside
    one standard deviation, and it will be read as a large regression.
    """
    a = nominal_df.groupby("attack")[KEYS].agg(["mean", "std"])
    b = agent_df.groupby("attack")[KEYS].agg(["mean", "std"])
    out = a.join(b, lsuffix="_nominal", rsuffix="_agent")
    for k in KEYS:
        mu_n, sd_n, mu_a = a[(k, "mean")], a[(k, "std")], b[(k, "mean")]
        delta = mu_a - mu_n                      # agent minus nominal, absolute
        out[(f"{k}_delta", "")] = delta
        pct = (delta / mu_n.abs().replace(0, np.nan) * 100 if k == "return"
               else -delta / mu_n.replace(0, np.nan) * 100)
        snr = mu_n.abs() / sd_n.replace(0, np.nan)
        out[(f"{k}_improvement_pct", "")] = pct.where(snr >= pct_min_snr)
    return out


def paired_delta(nominal_df, agent_df, keys=None) -> pd.DataFrame:
    """Per-seed paired differences.

    `sweep` runs both arms on the same seeds with the same injectors, so the
    runs pair up and the difference has far less variance than either arm on
    its own. Unpaired means with per-arm stds understate the evidence on the
    classes where the agent genuinely helps and overstate it where it does not.

    Columns: mean difference (agent - nominal), its std, the standard error
    ratio (mean/SE, a paired t statistic), and how many of the n seeds improved.
    """
    keys = keys or KEYS
    m = nominal_df.merge(agent_df, on=["attack", "seed"], suffixes=("_nom", "_agt"))
    frames = []
    for k in keys:
        d = m[f"{k}_agt"] - m[f"{k}_nom"]
        better = d > 0 if k == "return" else d < 0      # lower is better except return
        # Ties counted separately. Without this, a metric that is identically
        # zero in BOTH arms -- v_violation_rate on every class that never
        # breaches the envelope -- reports n_better = 0 of n, which reads as
        # the agent losing every seed when in fact nothing happened either way.
        tied = d == 0
        g = d.groupby(m["attack"])
        n = g.count()
        mean, std = g.mean(), g.std()
        se = std / np.sqrt(n.clip(lower=1))
        frames.append(pd.DataFrame({
            (k, "delta_mean"): mean,
            (k, "delta_std"): std,
            (k, "t"): mean / se.replace(0, np.nan),
            (k, "n_better"): better.groupby(m["attack"]).sum().astype(int),
            (k, "n_tied"): tied.groupby(m["attack"]).sum().astype(int),
            (k, "n"): n.astype(int),
        }))
    return pd.concat(frames, axis=1)


def _split_channels(a: np.ndarray, n: int):
    """(T, 2n) action array -> (xi block, zeta block)."""
    return a[:, :n], a[:, n:]


def false_trigger_rate(env, policy, n_seeds=50, trust_floor=0.5) -> dict:
    """How much coordination does the agent give up with NO attack present?

    WHAT THE PREVIOUS VERSION MEASURED, AND WHY IT WAS WRONG
    It reported max|a| over the action vector. Two problems, both fatal:

    1. Sign. The plant maps a -> trust = clip(1 + a, 0, 1), so every a >= 0 is
       full trust and behaviourally identical to the nominal controller. The
       reward charges mean(1 - trust) precisely because |a| taxes a harmless
       half of the action space. Reporting |a| reintroduces the error the
       reward was written to avoid: a policy sitting at a = +0.87 on every
       channel IS the published controller and scores 0.87 here.

    2. Dimension. max over 2n = 20 components is an extreme-value statistic.
       One channel at 0.21 sets frac_steps_above_0.2 to 1.0 for that step, so
       the threshold tests are close to vacuous at this width.

    What matters for deployability is connectivity loss, mean(1 - trust) -- the
    same quantity the reward charges -- plus the worst single channel, since one
    fully isolated DG is a different failure from twenty slightly discounted
    ones. Both are reported, split by channel, because the frequency and voltage
    consensus paths fail differently.
    """
    from ..plant.attacks import NoAttack

    n = env.n
    rows = []
    for s in range(n_seeds):
        df, _ = rollout(env, policy, injector=NoAttack(), seed=s)
        a = np.stack(df["action_norm"].to_numpy())          # (T, 2n)
        loss = np.clip(-a, 0.0, 1.0)                        # per-channel 1 - trust
        xi_loss, zeta_loss = _split_channels(loss, n)
        rows.append({
            "loss_mean": loss.mean(axis=1),                 # per step
            "loss_max": loss.max(axis=1),
            "n_isolated": (loss > (1.0 - trust_floor)).sum(axis=1),
            "xi_mean": xi_loss.mean(axis=1),
            "zeta_mean": zeta_loss.mean(axis=1),
            "abs_max": np.abs(a).max(axis=1),
            "frac_nonneg": (a >= 0.0).mean(axis=1),
        })
    cat = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}

    return {
        # --- the deployability numbers ---
        "mean_connectivity_loss": float(cat["loss_mean"].mean()),
        "p99_connectivity_loss": float(np.percentile(cat["loss_mean"], 99)),
        "frac_steps_loss_above_0.05": float((cat["loss_mean"] > 0.05).mean()),
        "frac_steps_loss_above_0.20": float((cat["loss_mean"] > 0.20).mean()),
        "worst_channel_loss_p99": float(np.percentile(cat["loss_max"], 99)),
        "mean_channels_isolated": float(cat["n_isolated"].mean()),
        "frac_steps_any_isolated": float((cat["n_isolated"] > 0).mean()),
        "mean_loss_xi": float(cat["xi_mean"].mean()),
        "mean_loss_zeta": float(cat["zeta_mean"].mean()),
        # --- degeneracy diagnostic, not a trigger metric ---
        # Fraction of action components parked in the reward-flat region a >= 0.
        # If this is near 1 while mean_connectivity_loss is near 0, the policy
        # has drifted into the half of the action space where every point is
        # behaviourally identical, and the |a| statistics below mean nothing.
        "frac_components_in_flat_region": float(cat["frac_nonneg"].mean()),
        "mean_abs_action_NOT_A_TRIGGER_METRIC": float(cat["abs_max"].mean()),
    }
