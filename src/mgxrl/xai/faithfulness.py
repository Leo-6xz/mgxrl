"""Faithfulness testing for the explanation layer.

This is what turns the XAI component from decoration into a contribution. An
LLM-generated sentence is worthless as evidence unless you can show it tracks
the policy's actual causal structure. G-Eval alone does not show that: it
measures whether an LLM finds the text plausible, which correlates with fluency
rather than truth. We report these metrics *alongside* G-Eval, and lead with them.

TWO CORRECTIONS OVER THE FIRST VERSION, both of which manufactured a perfect
score rather than measuring one:

1. The counterfactual probe perturbed each feature by `delta * (|x_i| + 1e-3)`.
   With baseline = 0 the attribution is x_i * grad_i, so the top-ranked feature
   is systematically one with a large |x_i| -- and therefore received the
   largest absolute nudge, while a random comparison feature usually received a
   smaller one. The probe was partly measuring perturbation size. It returned a
   win rate of exactly 1.0, which it would also have returned for attributions
   carrying no information. The perturbation is now a fixed absolute step; the
   observations are VecNormalize'd, so features are already comparable and the
   scaling bought nothing.

2. The interval came from bootstrapping the win vector. Resampling an all-ones
   array yields all ones, so a perfect score reported ci95 = [1.0, 1.0]. That
   is arithmetic, not evidence. A Wilson score interval is well defined at the
   boundary: 20/20 gives [0.84, 1.00].

Two controls are added, because a faithfulness number without a null is not
interpretable:

  null_win_rate       the same procedure with a RANDOM feature standing in for
                      the top-ranked one. Should sit near 0.5. If it does not,
                      the probe is measuring something other than the ranking.
  deletion_area_vs_magnitude
                      ranked ablation compared against ordering features by
                      |x| instead of by attribution. If attribution does not
                      beat plain magnitude, Integrated Gradients is adding
                      nothing over "this input is large". We report this even when it is
                      unflattering.
"""

from __future__ import annotations

import numpy as np


def _scalar(attributor, x):
    return float(attributor._scalarize(np.asarray(x, np.float32)[None])
                 .detach().cpu().numpy()[0])


def wilson_interval(k: int, n: int, z: float = 1.96):
    """Score interval for a binomial proportion. Unlike a bootstrap of the win
    vector it remains informative at k == 0 and k == n."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def deletion_curve(attributor, obs, model, k_max=None, baseline=None):
    """Ablate features in attribution order and compare against two controls.

    A faithful ranking degrades the explained quantity faster than a random
    ordering (the usual control) AND faster than ordering by raw |x| (the
    control that matters here, since attribution is proportional to x under a
    zero baseline).
    """
    x = np.asarray(obs, np.float32)
    b = np.zeros_like(x) if baseline is None else np.asarray(baseline, np.float32)
    attr = attributor.integrated_gradients(x)
    k_max = k_max or min(20, len(x))

    def curve(order):
        xs, vals = x.copy(), []
        for i in order[:k_max]:
            xs[i] = b[i]
            vals.append(_scalar(attributor, xs))
        return np.asarray(vals)

    ranked = curve(np.argsort(-np.abs(attr)))
    by_magnitude = curve(np.argsort(-np.abs(x - b)))
    rng = np.random.default_rng(0)
    random = np.mean([curve(rng.permutation(len(x))) for _ in range(5)], axis=0)
    return {
        "ranked": ranked, "random": random, "by_magnitude": by_magnitude,
        "area_between": float(np.mean(random - ranked)),
        "area_vs_magnitude": float(np.mean(by_magnitude - ranked)),
    }


def counterfactual_consistency(attributor, model, obs, n_probe=10, delta=0.10,
                               seed=0):
    """Perturb the top-ranked feature vs. random ones and compare the induced
    change in action.

    The step is a FIXED absolute delta, identical for every feature. Scaling it
    by |x_i| made the comparison unfair in the direction of the hypothesis --
    see the module docstring.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(obs, np.float32)
    attr = attributor.integrated_gradients(x)
    top = int(np.argmax(np.abs(attr)))

    def action(v):
        a, _ = model.predict(np.asarray(v, np.float32), deterministic=True)
        return np.asarray(a, np.float64).reshape(-1)

    a0 = action(x)

    def induced(j):
        xj = x.copy()
        xj[j] += delta
        return float(np.linalg.norm(action(xj) - a0))

    # Without replacement, and the null drawn from OUTSIDE the comparison set.
    # Sampling with rng.integers allowed duplicates, which biases the mean the
    # top feature is scored against.
    pool = np.array([j for j in range(len(x)) if j != top])
    picks = rng.choice(pool, size=min(n_probe + 1, len(pool)), replace=False)
    others, null_idx = picks[:-1].tolist(), int(picks[-1])

    d_top = induced(top)
    d_rand = [induced(j) for j in others]
    mean_rand = float(np.mean(d_rand)) if d_rand else 0.0
    d_null = induced(null_idx)

    return {
        "delta_top": d_top,
        "delta_random_mean": mean_rand,
        "ratio": float(d_top / mean_rand) if mean_rand > 0 else float("nan"),
        "win": bool(d_top > mean_rand),
        "null_win": bool(d_null > mean_rand),
    }


def aggregate(attributor, model, states, **kw):
    """Run the battery over many states.

    Read the win rate against `null_win_rate`, not against 0.5 in the abstract.
    A win rate of 0.95 with a null of 0.90 is a much weaker result than 0.80
    with a null of 0.50, and only the second says the ranking is doing work.
    """
    # A DISTINCT seed per state. counterfactual_consistency defaults to seed=0
    # and this loop used to accept that default for every state, so all of them
    # drew the same comparison indices and the same null index: one comparison
    # replicated N times, not N independent trials. Both the win rate and the
    # null rate then collapse to exactly 0.0 or 1.0, and any interval computed
    # over them -- bootstrap or Wilson -- is meaningless because the trials are
    # perfectly correlated. Observed directly: win 1.0 and null 1.0 on the same
    # 20 states.
    base_seed = int(kw.pop("seed", 0))
    wins, nulls, ratios, areas, mags = [], [], [], [], []
    for i, s in enumerate(states):
        r = counterfactual_consistency(attributor, model, s,
                                       seed=base_seed + i, **kw)
        wins.append(r["win"])
        nulls.append(r["null_win"])
        if r["ratio"] == r["ratio"]:
            ratios.append(r["ratio"])
        d = deletion_curve(attributor, s, model)
        areas.append(d["area_between"])
        mags.append(d["area_vs_magnitude"])

    n = len(wins)
    k = int(np.sum(wins))
    lo, hi = wilson_interval(k, n)
    return {
        "counterfactual_win_rate": float(k / n) if n else float("nan"),
        "ci95_wilson": [lo, hi],
        "null_win_rate": float(np.mean(nulls)) if n else float("nan"),
        "effect_ratio_median": float(np.median(ratios)) if ratios else float("nan"),
        "deletion_area_vs_random": float(np.mean(areas)),
        "deletion_area_vs_magnitude": float(np.mean(mags)),
        "n_states": int(n),
    }
