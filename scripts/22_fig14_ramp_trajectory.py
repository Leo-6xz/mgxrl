#!/usr/bin/env python3
"""
22_fig14_ramp_trajectory.py

Fig. 14: closed-loop response to a Ramp attack, undefended consensus controller
(a = 0) against the trained trust-weighted policy, on the SAME episode.

WHY THIS SCRIPT EXISTS
03_evaluate.py keeps only per-episode summaries (benchmarks.rollout drops
info["raw_measurement"]), so no time series was ever written". This script reruns one
episode per arm and keeps the measurements.

PAIRING
The episode is built exactly as benchmarks.sweep builds it: env.reset(seed=s)
and an injector from make_injector(env, "Ramp", default_rng(1000 + s)). Both
arms therefore see the same load draw, the same benign load steps and the same
attack (targets, onset, slope), so the difference between the curves is the
policy and nothing else. A fresh injector is built for each arm so no
injector state carries over from the first rollout into the second.

WHICH EPISODE
Showing a hand-picked seed is cherry-picking. By default the seed is the one
whose paired Ramp return difference (agent - nominal) is the MEDIAN of the
sweep, read from sweep_nominal.csv / sweep_agent.csv written by 03_evaluate.py
for the same agent.

OUTPUT (in --out)
  fig14_ramp_trajectory.csv   long format, one row per arm per control step,
                              with per-DG f and V columns
  Fig14_ramp_response.svg/.png  6.3 in wide, 600 dpi, same style as the other figures.

Usage (PowerShell, one line):
  python scripts\\22_fig14_ramp_trajectory.py --agent runs\\ppo2_s0 --sweep-dir results
"""
from __future__ import annotations

import argparse, glob, importlib.util, os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import numpy as np
import pandas as pd

from mgxrl.envs.resilient_env import ResilientMicrogridEnv
from mgxrl.eval.benchmarks import make_injector


def _evaluate_module():
    """Reuse NormalizedPolicy and load_vecnorm from 03_evaluate.py rather than
    copying them: a second copy of the normalisation path is how the raw-obs
    bug happened the first time."""
    path = os.path.join(HERE, "03_evaluate.py")
    spec = importlib.util.spec_from_file_location("evaluate03", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def resolve_agent(p: str) -> str:
    if os.path.isfile(p):
        return p
    zips = sorted(glob.glob(os.path.join(p, "*.zip")))
    if len(zips) == 1:
        return zips[0]
    raise SystemExit(f"--agent {p}: expected a .zip or a folder holding exactly one; "
                     f"found {len(zips)}: {zips}. Pass the .zip explicitly.")


def median_seed(sweep_dir: str, attack: str = "Ramp"):
    fn, fa = (os.path.join(sweep_dir, f) for f in ("sweep_nominal.csv", "sweep_agent.csv"))
    if not (os.path.exists(fn) and os.path.exists(fa)):
        return None, None
    n, a = pd.read_csv(fn), pd.read_csv(fa)
    m = n[n.attack == attack].merge(a[a.attack == attack], on="seed", suffixes=("_nom", "_agt"))
    if m.empty:
        return None, None
    m["d"] = m["return_agt"] - m["return_nom"]
    med = m["d"].median()
    row = m.iloc[(m["d"] - med).abs().argsort().iloc[0]]
    return int(row["seed"]), float(row["d"])


def run_arm(env, policy, seed: int, arm: str) -> tuple[pd.DataFrame, object]:
    inj = make_injector(env, "Ramp", np.random.default_rng(1000 + seed))
    obs, _ = env.reset(seed=seed, options={"injector": inj})
    n, rows, total = env.n, [], 0.0
    for _ in range(env.max_steps):
        a = (np.zeros(env.n_actions, np.float32) if policy is None
             else policy.predict(obs, deterministic=True)[0])
        obs, r, term, trunc, info = env.step(a)
        total += r
        m = info["raw_measurement"]
        f = np.asarray(m["f"], float).reshape(-1)[:n]
        V = np.asarray(m["V"], float).reshape(-1)[:n]
        w = np.clip(1.0 + np.asarray(info["action_norm"], float), 0.0, 1.0)
        row = {"arm": arm, "t": float(info["t"]), "y_bin": int(info["y_bin"]), "reward": float(r),
               "f_mean": f.mean(), "f_min": f.min(), "f_max": f.max(),
               "V_mean": V.mean(), "V_min": V.min(), "V_max": V.max(),
               "conn_loss": float(np.mean(1.0 - w)),
               "conn_loss_xi": float(np.mean(1.0 - w[:n])), "conn_loss_zeta": float(np.mean(1.0 - w[n:])),
               "collapsed": bool(term)}
        row.update({f"f_dg{i + 1}": f[i] for i in range(n)})
        row.update({f"V_dg{i + 1}": V[i] for i in range(n)})
        rows.append(row)
        if term or trunc:
            break
    df = pd.DataFrame(rows)
    df.attrs["return"] = total
    return df, inj


def plot(df: pd.DataFrame, pp, onset: float, out: str, with_trust: bool) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, GREY, RED = "#1a1a1a", "#5f5f5f", "#8c1515"
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
        "font.size": 9, "axes.labelsize": 9.5, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 8.5, "axes.linewidth": 0.8, "axes.spines.top": False,
        "axes.spines.right": False, "legend.frameon": False, "svg.fonttype": "none",
        "savefig.dpi": 600})
    style = {"nominal": dict(color=RED, lw=1.2, label="undefended consensus controller"),
             "agent": dict(color=INK, lw=1.2, label="trust-weighted policy")}
    nrow = 3 if with_trust else 2
    fig, axs = plt.subplots(nrow, 1, figsize=(6.3, 2.0 * nrow + 0.4), sharex=True, layout="constrained")
    fig.get_layout_engine().set(w_pad=6 / 72, h_pad=4 / 72)
    a1, a2 = axs[0], axs[1]

    for arm, g in df.groupby("arm", sort=False):
        s = style[arm]
        a1.plot(g.t, (g.f_mean - pp.f_nom) * 1e3, **s)
        a1.fill_between(g.t, (g.f_min - pp.f_nom) * 1e3, (g.f_max - pp.f_nom) * 1e3,
                        color=s["color"], alpha=0.12, lw=0)
        a2.plot(g.t, g.V_min, color=s["color"], lw=1.1, ls="-")
        a2.plot(g.t, g.V_max, color=s["color"], lw=1.1, ls="--")
        if g.collapsed.any():
            tc = g.t.iloc[-1]
            for ax in (a1, a2):
                ax.axvline(tc, color=s["color"], lw=0.8, ls=(0, (1, 1.5)))
            a1.annotate("collapse", xy=(tc, 1.0), xycoords=("data", "axes fraction"), xytext=(-3, -3),
                        textcoords="offset points", fontsize=8.5, color=s["color"], ha="right", va="top")
    a1.axhline(0.0, color=GREY, lw=0.6, ls=":")
    a1.set_ylabel(f"f − {pp.f_nom:g} Hz (mHz)")
    a1.legend(loc="best", handlelength=1.8)
    a1.text(0.0, 1.02, "(a) mean DG frequency; shading spans the ten DGs", transform=a1.transAxes,
            fontsize=8.5, color=GREY, va="bottom")

    # Voltage envelope: shade it, but never let it flatten the data. If the
    # trajectories stay well inside 0.95-1.05 pu the band would fill the axis
    # and show nothing, so the axis follows the data and the margin is stated.
    lo, hi = df[["V_min"]].min().item(), df[["V_max"]].max().item()
    pad = max(0.15 * (hi - lo), 1e-4)
    ylo, yhi = lo - pad, hi + pad
    a2.axhspan(pp.v_min, pp.v_max, color="#6e6e6e", alpha=0.12, lw=0)
    for lim in (pp.v_min, pp.v_max):
        if ylo <= lim <= yhi:
            a2.axhline(lim, color=GREY, lw=0.8, ls=(0, (4, 2)))
    a2.set_ylim(ylo, yhi)
    inside = (lo >= pp.v_min) and (hi <= pp.v_max)
    note = (f"(b) min (solid) and max (dashed) DG voltage; shaded = {pp.v_min:g}–{pp.v_max:g} pu envelope"
            + ("" if not inside else f" (never left: {lo:.4f}–{hi:.4f} pu)"))
    a2.text(0.0, 1.02, note, transform=a2.transAxes, fontsize=8.5, color=GREY, va="bottom")
    a2.set_ylabel("voltage (pu)")

    if with_trust:
        a3 = axs[2]
        g = df[df.arm == "agent"]
        a3.plot(g.t, g.conn_loss_xi, color=INK, lw=1.1, label="frequency channel (ξ)")
        a3.plot(g.t, g.conn_loss_zeta, color=GREY, lw=1.1, ls="--", label="voltage channel (ζ)")
        a3.set_ylabel("connectivity loss")
        a3.set_ylim(0, 1)
        a3.legend(loc="best")
        a3.text(0.0, 1.02, "(c) policy connectivity loss, mean(1 − w)", transform=a3.transAxes,
                fontsize=8.5, color=GREY, va="bottom")

    for ax in axs:
        ax.axvline(onset, color=INK, lw=0.8, ls=(0, (2, 2)))
    a1.annotate("attack onset", xy=(onset, a1.get_ylim()[1]), xytext=(4, -10),
                textcoords="offset points", fontsize=8.5, va="top")
    axs[-1].set_xlabel("time (s)")

    base = os.path.join(out, "Fig14_ramp_response")
    fig.savefig(base + ".png")
    fig.savefig(base + ".svg")
    plt.close(fig)
    s = open(base + ".svg", encoding="utf-8").read()
    s = re.sub(r"<metadata>.*?</metadata>", "", s, flags=re.S)
    open(base + ".svg", "w", encoding="utf-8").write(s)
    print(f"Wrote {base}.svg and .png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True, help="PPO .zip, or the run folder holding it")
    ap.add_argument("--vecnorm", default=None, help="defaults to vecnormalize.pkl beside the agent")
    ap.add_argument("--sweep-dir", default="results",
                    help="folder with sweep_nominal.csv / sweep_agent.csv from 03_evaluate.py for THIS agent")
    ap.add_argument("--seed", type=int, default=None, help="override the median-seed choice")
    ap.add_argument("--with-trust", action="store_true", help="add panel (c): the policy's connectivity loss")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    ev = _evaluate_module()
    from stable_baselines3 import PPO

    env = ResilientMicrogridEnv()
    agent = resolve_agent(a.agent)
    vn_path = a.vecnorm or os.path.join(os.path.dirname(agent), "vecnormalize.pkl")
    if not os.path.exists(vn_path):
        raise SystemExit(f"No VecNormalize statistics at {vn_path}. Pass --vecnorm.")
    policy = ev.NormalizedPolicy(PPO.load(agent), ev.load_vecnorm(vn_path, env.obs_dim))

    if a.seed is not None:
        seed, how = a.seed, "set by --seed"
    else:
        seed, d = median_seed(a.sweep_dir)
        if seed is None:
            raise SystemExit(f"No Ramp rows in {a.sweep_dir}/sweep_*.csv. Run 03_evaluate.py for this "
                             f"agent first, point --sweep-dir at its output, or pass --seed.")
        how = f"median paired Ramp return difference ({d:+,.0f}) in {a.sweep_dir}"
    print(f"agent: {agent}\nvecnorm: {vn_path}\nseed: {seed} ({how})")

    nom, inj = run_arm(env, None, seed, "nominal")
    agt, _ = run_arm(env, policy, seed, "agent")
    # An arm may END early: env.step terminates on meas["collapsed"]. That is a
    # result, not a pairing failure, so compare the shared prefix and require
    # any length difference to be explained by a collapse.
    k = min(len(nom), len(agt))
    print(f"steps: undefended {len(nom)} (collapsed={bool(nom.collapsed.any())}), "
          f"trust-weighted {len(agt)} (collapsed={bool(agt.collapsed.any())}), max {env.max_steps}")
    tn, ta = nom.t.to_numpy()[:k], agt.t.to_numpy()[:k]
    if not np.allclose(tn, ta):
        j = int(np.flatnonzero(~np.isclose(tn, ta))[0])
        raise SystemExit(f"Time bases diverge at step {j}: t = {tn[j]} vs {ta[j]}. The pairing is broken.")
    if len(nom) != len(agt):
        short = nom if len(nom) < len(agt) else agt
        if not short.collapsed.any():
            raise SystemExit("Arms differ in length without a collapse; the pairing is broken.")
        print(f"NOTE: the {'undefended' if short is nom else 'trust-weighted'} arm COLLAPSED at "
              f"t = {short.t.iloc[-1]:.2f} s. Its curve ends there; say so in the caption.")
    if not (nom.y_bin.to_numpy()[:k] == agt.y_bin.to_numpy()[:k]).all():
        raise SystemExit("The two arms disagree on when the attack is active; the pairing is broken.")

    print(f"injector: {inj!r}")
    print("  CHECK the caption against this line: which channel (xi = frequency consensus, "
          "zeta = voltage consensus) and which DG(s) the Ramp corrupts.")
    hit = np.flatnonzero(nom.y_bin.to_numpy() == 1)
    if hit.size == 0:
        raise SystemExit("y_bin is never 1 in this episode: the attack never became active.")
    onset = float(nom.t.iloc[hit[0]])

    df = pd.concat([nom, agt], ignore_index=True)
    df.to_csv(os.path.join(a.out, "fig14_ramp_trajectory.csv"), index=False)

    pp = env.pp
    print("\nNumbers for the caption / text:")
    print(f"  attack onset t = {onset:.2f} s; episode ends t = {df.t.max():.2f} s")
    for name, g in (("undefended", nom), ("trust-weighted", agt)):
        post = g[g.t >= onset]
        print(f"  {name:15s} return {g.attrs['return']:+12,.0f} | peak |f - f_nom| after onset "
              f"{(post.f_mean - pp.f_nom).abs().max():.5f} Hz (worst DG "
              f"{np.maximum((post.f_max - pp.f_nom).abs(), (post.f_min - pp.f_nom).abs()).max():.5f} Hz)"
              f" | V range {g.V_min.min():.4f}-{g.V_max.max():.4f} pu"
              f" | steps outside {pp.v_min:g}-{pp.v_max:g} pu: "
              f"{int(((g.V_min < pp.v_min) | (g.V_max > pp.v_max)).sum())}"
              f"{' | COLLAPSED' if g.collapsed.any() else ''}")
    pre = agt[agt.t < onset].conn_loss.mean()
    post = agt[agt.t >= onset].conn_loss.mean()
    print(f"  policy connectivity loss: {pre:.3f} before onset, {post:.3f} after")

    plot(df, pp, onset, a.out, a.with_trust)


if __name__ == "__main__":
    main()
