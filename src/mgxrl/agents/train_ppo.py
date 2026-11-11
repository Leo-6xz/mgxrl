"""PPO training against the residual environment.

Curriculum matters here. Training directly on the full attack mix with random
onsets gives a very sparse learning signal, because for most of each episode
nothing is wrong. The staged schedule below is what actually converges.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, asdict, replace

import numpy as np

try:
    from stable_baselines3.common.callbacks import BaseCallback
except ImportError:      # keep the module importable without the RL stack
    BaseCallback = object

from ..envs.resilient_env import ResilientMicrogridEnv, EnvConfig, RewardWeights
from ..plant.microgrid import MicrogridParams


@dataclass
class TrainConfig:
    total_timesteps: int = 2_000_000
    n_envs: int = 8
    algo: str = "ppo"                 # "ppo" or "sac"
    learning_rate: float = 3e-4
    n_steps: int = 512
    batch_size: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    # ent_coef was 0.005. With a 20-dimensional action space the Gaussian
    # entropy is ~29, so that term paid ~0.147 reward per step purely for
    # keeping the distribution wide -- directly opposing the shrinkage the
    # policy needs. Combined with log_std_init=0 it kept std at 1.04 after
    # 100k steps, which puts 34% of every sampled dimension outside [-1,1].
    ent_coef: float = 0.001
    # PPO's clip_range bounds each epoch's move, not the cumulative move across
    # epochs. With the default n_epochs=10 the observed approx_kl reached 4.2
    # (healthy is ~0.01-0.03) and explained_variance flipped between 0.86 and
    # -4.56 as the critic lost track of a policy being thrown across the space.
    # target_kl aborts the update early once the distribution has moved far
    # enough; n_epochs=5 reduces how far it can get before that triggers.
    target_kl: float = 0.03
    n_epochs: int = 5
    # std = exp(-1.0) = 0.37, so only ~0.7% of samples clip at initialisation
    # instead of 34%. Exploration still happens; it just does not begin from a
    # distribution that is mostly actuator saturation.
    log_std_init: float = -1.0
    net_arch: tuple = (256, 256)
    seed: int = 0
    outdir: str = "runs/ppo"

    # --- reward overrides -------------------------------------------------
    w_action: float | None = None
    """Weight on CONNECTIVITY LOSS, mean(1 - trust). None keeps RewardWeights'
    default of 1.0.

    This is the knob for the nominal-cost / attack-protection frontier. Trained
    to convergence at w_action = 1.0, the policy holds 2.7-4.7 sources below
    full trust during HEALTHY operation, costing ~1,800-5,400 reward on Normal
    while buying 14,000-47,000 on Additive and Ramp. That is rational under
    this action space: a trust weight acts on the signal BEFORE it enters the
    consensus update, so a source trusted at attack onset has already injected
    a corrupted value, and standing distrust is the only way to be protected at
    onset.

    Raising p_normal does NOT change this. Measured connectivity loss 0.159 at
    p_normal = 1/7 against 0.196 at p_normal = 0.5 with matched 5M budgets,
    both inside the 0.159-0.258 spread across three seeds at the default. The
    training distribution is not the cause; the pre-consensus position of the
    weight is.
    """


CURRICULUM = [
    # (fraction of budget, description, EnvConfig field overrides)
    (0.15, "nominal + benign load steps only",
     dict(p_normal=1.0, attack_subset=("Normal",))),
    (0.35, "overt attacks (Additive, Ramp, Sinusoid, DoS)",
     dict(p_normal=1 / 5,
          attack_subset=("Normal", "Additive", "Ramp", "Sinusoid", "DoS"))),
    (0.50, "full mix incl. SlowRamp and Coordinated Stealth", dict()),
]


def build_weights(cfg: TrainConfig) -> RewardWeights:
    """RewardWeights with any TrainConfig override applied.

    Built once and threaded through every curriculum stage AND through
    deterministic_return, so the reward the policy is judged on is the reward it
    was trained on. Constructing RewardWeights() separately in the evaluation
    path would silently score a w_action-swept agent under the default weight,
    which would make the frontier meaningless.
    """
    over = {}
    if cfg.w_action is not None:
        over["w_action"] = float(cfg.w_action)
    return RewardWeights(**over)


def deterministic_return(model, venv, env_cfg, weights=None, n_episodes=3,
                         seed=10_000):
    """Mean return of the GREEDY policy on nominal episodes.

    `ep_rew_mean` measures the exploring policy, and at std = 0.368 the noise
    alone costs ~18,000 reward per episode -- a policy whose mean action is
    exactly zero scores -13,108 where doing nothing scores +5,170. Judging the
    agent by ep_rew_mean therefore says almost nothing about the controller
    that gets deployed, which is the deterministic one.

    Observations are normalised with the running VecNormalize statistics, since
    that is what the policy was trained on.
    """
    from dataclasses import replace as _replace
    from ..plant.attacks import NoAttack

    env = ResilientMicrogridEnv(
        plant_params=MicrogridParams(seed=seed),
        config=_replace(env_cfg, seed=seed),
        weights=weights or RewardWeights(),
    )
    totals = []
    for k in range(n_episodes):
        obs, _ = env.reset(seed=seed + k, options={"injector": NoAttack()})
        total = 0.0
        for _ in range(env.max_steps):
            o = venv.normalize_obs(obs.reshape(1, -1))
            a, _ = model.predict(o, deterministic=True)
            obs, r, term, trunc, _ = env.step(np.asarray(a).reshape(-1))
            total += r
            if term or trunc:
                break
        totals.append(total)
    return float(np.mean(totals))


class SaturationMonitor(BaseCallback):
    """Log how often the actuator is pinned, and say so loudly.

    The collapsed policy was only discovered after training, from the
    evaluation. Actuator saturation is the single clearest early symptom, so it
    belongs in the training log: `train/action_saturation` should fall, not sit
    near 1.0.
    """

    def __init__(self, warn_above: float = 0.5, verbose: int = 0):
        super().__init__(verbose)
        self.warn_above = warn_above
        self._warned = False

    def _on_step(self) -> bool:
        acts = self.locals.get("actions")
        if acts is None:
            return True
        a = np.asarray(acts, dtype=float)
        frac = float((np.abs(a) > 0.9).mean())
        self.logger.record("train/action_saturation", frac)
        self.logger.record("train/action_magnitude", float(np.abs(a).mean()))
        # The quantity the reward actually charges. Unlike |a| this is
        # sign-resolved, so it distinguishes a policy parked in the harmless
        # positive orthant (every a >= 0 is full trust) from one dismantling
        # the consensus layer. Watch it rise during training instead of only
        # discovering the hedging at evaluation.
        self.logger.record("train/connectivity_loss",
                           float(np.clip(-a, 0.0, 1.0).mean()))
        if frac > self.warn_above and self.num_timesteps > 50_000 and not self._warned:
            print(f"\n  WARNING: {frac:.0%} of action dimensions are saturated at "
                  f"{self.num_timesteps:,} steps.")
            print("  Check train/connectivity_loss too: if that is also high the")
            print("  policy is isolating sources permanently, not merely pinning")
            print("  |a|, and w_action is the knob for it.\n")
            self._warned = True
        return True


def make_env(seed=0, detector=None, plant=None, env_cfg=None, weights=None,
             monitor_dir=None):
    def _init():
        from stable_baselines3.common.monitor import Monitor

        cfg = replace(env_cfg, seed=seed) if env_cfg is not None else EnvConfig(seed=seed)
        env = ResilientMicrogridEnv(
            plant_params=plant or MicrogridParams(seed=seed),
            config=cfg,
            weights=weights or RewardWeights(),
            detector=detector,
        )
        # Monitor is REQUIRED, not optional. Without it SB3's ep_info_buffer
        # stays empty and `rollout/ep_rew_mean` is never logged -- the entire
        # first training run produced no episode-return metric at all, so there
        # was no way to see whether learning was happening. Everything else in
        # the log describes the optimiser, not the task.
        return Monitor(env, filename=monitor_dir)
    return _init


def _stage_env(cfg, overrides, detector, base_env_cfg, weights=None):
    """Build a vectorised env for one curriculum stage.

    The overrides used to be unpacked from CURRICULUM and then discarded -- the
    loop bound them to `_over` and never passed them anywhere, so all three
    "stages" trained on the identical full attack mix. The staged schedule was
    documentation, not behaviour. Rebuilding the vec env per stage is the
    simplest way to make it real, and costs a few seconds per transition.

    `weights` is threaded through for the same reason: a stage rebuilt without
    it would silently revert to the default reward part-way through a run.
    """
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

    stage_cfg = replace(base_env_cfg, **overrides) if overrides else base_env_cfg
    venv = SubprocVecEnv([make_env(cfg.seed + i, detector, env_cfg=stage_cfg,
                                   weights=weights)
                          for i in range(cfg.n_envs)])
    return VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)


def train(cfg: TrainConfig | None = None, detector=None):
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
    from stable_baselines3.common.callbacks import CheckpointCallback

    cfg = cfg or TrainConfig()
    os.makedirs(cfg.outdir, exist_ok=True)

    # include_detector defaults False in EnvConfig, and the env raises if it is
    # True with no detector. Set it from what was actually passed so the flag
    # cannot be silently wrong from EITHER side: previously
    # `_det_dim = 7 if (include_detector and detector is not None) else 0`
    # dropped the feature when the detector was missing; leaving the flag off
    # here would drop it when the detector was PRESENT. Both directions are now
    # closed.
    base_env_cfg = EnvConfig(seed=cfg.seed, include_detector=detector is not None)
    if detector is not None:
        print(f"[obs] detector wired: obs_dim {4 * 8 * 10} -> {4 * 8 * 10 + 7} "
              f"(7 class posteriors appended). Checkpoints trained without it "
              f"will NOT load against this observation space.")
    weights = build_weights(cfg)
    if cfg.w_action is not None:
        print(f"[reward] w_action = {weights.w_action}  "
              f"(connectivity loss, mean(1 - trust))")

    # Observation normalisation is not optional: V is O(1) while frequency
    # deviation is O(1e-4). Without it the critic ignores frequency entirely.
    venv = _stage_env(cfg, CURRICULUM[0][2], detector, base_env_cfg, weights)

    policy_kwargs = dict(net_arch=dict(pi=list(cfg.net_arch), vf=list(cfg.net_arch)),
                         log_std_init=cfg.log_std_init)
    Algo = {"ppo": PPO, "sac": SAC}[cfg.algo]
    kw = dict(policy="MlpPolicy", env=venv, learning_rate=cfg.learning_rate,
              gamma=cfg.gamma, seed=cfg.seed, verbose=1,
              policy_kwargs=policy_kwargs, tensorboard_log=cfg.outdir)
    if cfg.algo == "ppo":
        kw.update(n_steps=cfg.n_steps, batch_size=cfg.batch_size,
                  gae_lambda=cfg.gae_lambda, clip_range=cfg.clip_range,
                  ent_coef=cfg.ent_coef, target_kl=cfg.target_kl,
                  n_epochs=cfg.n_epochs)
    model = Algo(**kw)

    ckpt = CheckpointCallback(save_freq=50_000, save_path=cfg.outdir, name_prefix="agent")
    sat = SaturationMonitor()
    for i, (frac, desc, over) in enumerate(CURRICULUM):
        steps = int(cfg.total_timesteps * frac)
        print(f"[curriculum {i + 1}/{len(CURRICULUM)}] {desc}: {steps:,} steps")
        if i > 0:
            # Carry the running observation/reward statistics across stages so
            # the value function is not reset by the change of distribution.
            obs_rms, ret_rms = venv.obs_rms, venv.ret_rms
            venv.close()
            venv = _stage_env(cfg, over, detector, base_env_cfg, weights)
            venv.obs_rms, venv.ret_rms = obs_rms, ret_rms
            model.set_env(venv)
        model.learn(total_timesteps=steps, reset_num_timesteps=False,
                    callback=[ckpt, sat])
        buf = getattr(model, "ep_info_buffer", None)
        stoch = float(np.mean([e["r"] for e in buf])) if buf else float("nan")
        det = deterministic_return(model, venv, base_env_cfg, weights=weights,
                                   n_episodes=3)
        print(f"[curriculum {i + 1}/{len(CURRICULUM)}] "
              f"stochastic return {stoch:,.0f} | deterministic {det:,.0f}")
        print("   (stochastic includes exploration noise and will be far worse; "
              "judge the policy on the deterministic figure)")
        if i == 0 and det < 0:
            print("  WARNING: stage 1 is nominal-only. Acting sparingly scores about")
            print("  +5,000 there, so a NEGATIVE deterministic return means the policy")
            print("  is actively making things worse. Stop before spending the budget.")

    model.save(os.path.join(cfg.outdir, "agent_final"))
    venv.save(os.path.join(cfg.outdir, "vecnormalize.pkl"))
    # Record the reward the agent was trained under, so a sweep directory is
    # self-describing and a checkpoint cannot be compared against one trained
    # under a different weight by accident.
    import json
    json.dump({"w_action": weights.w_action, "seed": cfg.seed,
               "total_timesteps": cfg.total_timesteps,
               # Observation layout. A run directory that does not record this
               # cannot be told apart from one trained at a different width
               # except by loading it, which is how three 5M runs went by at
               # obs_dim 320 while the config read include_detector = True.
               "include_detector": base_env_cfg.include_detector,
               "obs_dim": int(venv.observation_space.shape[0]),
               "history_len": base_env_cfg.history_len},
              open(os.path.join(cfg.outdir, "train_config.json"), "w"), indent=2)
    return model, venv
