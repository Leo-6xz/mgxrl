"""Non-learned comparison controllers for the trust-weighted agent.

Three references, all evaluated through the same `sweep` protocol, seeds and
attacks as the agent:

* W-MSR (receiver-side trimming, F = 1): the standard resilient-consensus rule.
  It is not an action of the agent's action space; it is switched on inside the
  plant with `use_wmsr(env)` and the episode is run with policy=None.
* Innovation threshold: the same source-level trust action as the agent, set by
  a fixed rule. Source j is distrusted on a channel while the magnitude of its
  consensus innovation exceeds a threshold calibrated on attack-free episodes
  with seeds disjoint from evaluation. It passes through the same safety
  projection (rate limit, connectivity floor) as the agent.
* Oracle isolation: distrusts exactly the attacked sources from the true onset,
  using the attack's ground truth. Not deployable; an upper reference for what
  source-level trust can achieve.
"""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np

from ..plant.attacks import NoAttack


@contextmanager
def use_wmsr(env, F: int = 1):
    """Run the plant with receiver-side W-MSR filtering, restoring it after."""
    old = getattr(env.mg, "consensus_filter", "none"), getattr(env.mg, "wmsr_F", 1)
    env.mg.consensus_filter, env.mg.wmsr_F = "wmsr", F
    try:
        yield env
    finally:
        env.mg.consensus_filter, env.mg.wmsr_F = old


def _innovation_slices(env):
    n, det = env.n, env._det_dim
    end = env.obs_dim - det
    frame0 = end - 8 * n
    return slice(frame0 + 6 * n, frame0 + 7 * n), slice(frame0 + 7 * n, end)


def calibrate_threshold(env, seeds, quantile=1.0):
    """Largest (or quantile) |innovation| per channel over attack-free episodes
    under the nominal controller, including the benign load steps."""
    s_xi, s_zeta = _innovation_slices(env)
    vals_xi, vals_zeta = [], []
    for s in seeds:
        obs, _ = env.reset(seed=int(s), options={"injector": NoAttack()})
        for _ in range(env.max_steps):
            vals_xi.append(np.abs(obs[s_xi]))
            vals_zeta.append(np.abs(obs[s_zeta]))
            obs, _, term, trunc, _ = env.step(np.zeros(env.n_actions, np.float32))
            if term or trunc:
                break
    q = lambda v: float(np.quantile(np.concatenate(v), quantile))
    return q(vals_xi), q(vals_zeta)


class ThresholdPolicy:
    """Distrust source j on a channel while |innovation_j| > threshold."""

    def __init__(self, env, tau_xi: float, tau_zeta: float):
        self.n = env.n
        self.s_xi, self.s_zeta = _innovation_slices(env)
        self.tau_xi, self.tau_zeta = tau_xi, tau_zeta

    def predict(self, obs, deterministic=True):
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        a = np.zeros(2 * self.n, np.float32)
        a[: self.n][np.abs(obs[self.s_xi]) > self.tau_xi] = -1.0
        a[self.n :][np.abs(obs[self.s_zeta]) > self.tau_zeta] = -1.0
        return a, None


class OraclePolicy:
    """Distrust exactly the attacked sources from the true onset (ground truth)."""

    def __init__(self, env):
        self.env = env

    def predict(self, obs, deterministic=True):
        env = self.env
        a = np.zeros(env.n_actions, np.float32)
        inj = env.injector
        if inj.class_index != 0 and inj.active(env.mg.state.t):
            for j in inj.targets:
                a[j] = -1.0
                a[env.n + j] = -1.0
        return a, None


def wmsr_edges_kept(env, n_seeds=20, F=1, seed0=0):
    """Fraction of consensus edges W-MSR keeps with NO attack present."""
    kept_xi, kept_zeta = [], []
    with use_wmsr(env, F):
        for s in range(seed0, seed0 + n_seeds):
            env.reset(seed=s, options={"injector": NoAttack()})
            for _ in range(env.max_steps):
                _, _, term, trunc, _ = env.step(np.zeros(env.n_actions, np.float32))
                kept_xi.append(env.mg.wmsr_kept_xi)
                kept_zeta.append(env.mg.wmsr_kept_zeta)
                if term or trunc:
                    break
    return {"wmsr_edges_kept_xi": float(np.mean(kept_xi)),
            "wmsr_edges_kept_zeta": float(np.mean(kept_zeta)),
            "wmsr_connectivity_loss": float(1 - np.mean(kept_xi + kept_zeta))}


# --------------------------------------------------------------- topology

def circulant_adjacency(n: int, k: int) -> np.ndarray:
    """Each node linked to its k nearest neighbours on either side (k=1: ring)."""
    A = np.zeros((n, n))
    for i in range(n):
        for d in range(1, k + 1):
            A[i, (i + d) % n] = A[i, (i - d) % n] = 1.0
    return A


def set_comm_graph(env, adj):
    """Replace the CONSENSUS graph (the electrical ring is unchanged)."""
    env.mg.adj = np.asarray(adj, float)
    env._deg = env.mg.adj.sum(axis=0)
    env._deg_safe = np.where(env._deg > 0, env._deg, 1.0)


def r_robustness(adj) -> int:
    """Largest r such that the undirected graph is r-robust (brute force, n <= 14).

    r-robust: for every pair of disjoint nonempty node sets S1, S2, some node of
    S1 or of S2 has at least r neighbours outside its own set.
    """
    import itertools
    A = np.asarray(adj) > 0
    n = A.shape[0]

    def reach(S, r):
        out = ~S
        return bool((A[S][:, out].sum(axis=1) >= r).any())

    best = 0
    for r in range(1, n):
        ok = True
        for lab in itertools.product((0, 1, 2), repeat=n):
            lab = np.array(lab)
            S1, S2 = lab == 1, lab == 2
            if not S1.any() or not S2.any():
                continue
            if not (reach(S1, r) or reach(S2, r)):
                ok = False
                break
        if not ok:
            break
        best = r
    return best


# ------------------------------------------------------------ latency

def isolation_latency(env, policy, attacks, n_seeds=20, trust_floor=0.5):
    """Time from onset until every attacked source is distrusted on the xi
    channel (trust < trust_floor), plus false isolations of untouched sources."""
    import pandas as pd
    from .benchmarks import make_injector, rollout
    rows = []
    for name in attacks:
        for s in range(n_seeds):
            inj = make_injector(env, name, np.random.default_rng(1000 + s))
            df, _ = rollout(env, policy, injector=inj, seed=s)
            a = np.stack(df["action_norm"].to_numpy())[:, : env.n]
            trust = np.clip(1.0 + a, 0.0, 1.0)
            t = df["t"].to_numpy()
            tg = list(inj.targets)
            iso = (trust[:, tg] < trust_floor).all(axis=1)
            after = t >= inj.t_start
            hit = np.flatnonzero(iso & after)
            others = [j for j in range(env.n) if j not in tg]
            rows.append({
                "attack": name, "seed": s, "onset": inj.t_start, "targets": len(tg),
                "latency_s": float(t[hit[0]] - inj.t_start) if hit.size else np.nan,
                "isolated": bool(hit.size),
                "false_isolation_steps": int((trust[:, others] < trust_floor).any(axis=1).sum()),
            })
    return pd.DataFrame(rows)
