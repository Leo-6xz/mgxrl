"""Regression tests. Run these before every training job -- a silently broken
plant produces a beautifully converged agent that has learned nothing."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from mgxrl.plant.microgrid import Microgrid, MicrogridParams
from mgxrl.plant.attacks import ATTACK_REGISTRY, NoAttack, sample_attack


def settled(seed=0, **kw):
    mg = Microgrid(MicrogridParams(seed=seed, **kw))
    mg.attach_attack(NoAttack())
    return mg, mg.reset(settle_time=3.0)


def test_secondary_restores_nominal():
    _, m = settled()
    assert abs(m["f"].mean() - 50.0) < 0.005
    assert abs(m["V"].mean() - 1.0) < 0.002


def test_droop_only_leaves_steady_state_error():
    """Sanity check that the secondary controller is doing real work."""
    _, m = settled(k_p=0, c_f=0, k_q=0, c_v=0)
    assert abs(m["f"].mean() - 50.0) > 0.01


def test_consensus_reaches_agreement():
    mg, _ = settled()
    assert mg.state.xi.std() < 1e-3


@pytest.mark.parametrize("name", ["Additive", "Ramp", "SlowRamp", "Sinusoid", "Stealth"])
def test_attacks_perturb_the_grid(name):
    inj = ATTACK_REGISTRY[name](targets=(0,), t_start=0.5)
    mg = Microgrid(MicrogridParams(seed=1)); mg.attach_attack(inj)
    mg.reset(settle_time=3.0)
    dev = 0.0
    for _ in range(400):
        m = mg.step()
        dev = max(dev, abs(m["f"].mean() - 50) + abs(m["V"].mean() - 1.0) * 10)
    assert dev > 1e-3, f"{name} produced no measurable effect"


def test_dos_degrades_sharing_after_a_disturbance():
    """DoS is invisible at steady state; it only bites when the grid must
    respond. This test encodes that -- see README 'benign disturbances'."""
    out = {}
    for name in ("Normal", "DoS"):
        inj = ATTACK_REGISTRY[name](targets=(0,), t_start=1.0)
        mg = Microgrid(MicrogridParams(seed=2)); mg.attach_attack(inj)
        mg.reset(settle_time=3.0)
        for _ in range(600):
            if abs(mg.state.t - 2.0) < 5e-3:
                mg.apply_load_step(np.where(np.arange(10) == 3, 1.35, 1.0))
            m = mg.step()
        out[name] = mg.metrics(m)["p_sharing_error"]
    assert out["DoS"] > 3 * out["Normal"]


def test_stealth_stays_inside_the_normal_envelope():
    """If Stealth violates limits it is not stealthy and the class is trivial."""
    inj = ATTACK_REGISTRY["Stealth"](targets=(0, 1, 2), t_start=0.5)
    mg = Microgrid(MicrogridParams(seed=3)); mg.attach_attack(inj)
    mg.reset(settle_time=3.0)
    for _ in range(500):
        m = mg.step()
        assert m["V"].min() > 0.95 and m["V"].max() < 1.05


def test_no_divergence_over_long_horizon():
    mg, _ = settled()
    for _ in range(3000):
        m = mg.step()
    assert not m["collapsed"] and np.all(np.isfinite(m["V"]))


def test_dataset_row_schema():
    mg, m = settled()
    row = mg.to_dataset_row(m)
    assert row.shape == (37,)   # time + 3V + 3I + 10*(P,Q,f); label added separately


def test_attack_sampling_randomises_onset():
    rng = np.random.default_rng(0)
    starts = {sample_attack(rng).t_start for _ in range(30)}
    assert len(starts) > 5, "fixed onset lets the agent learn the clock"


# --------------------------------------------------------------- reward tests

def test_every_attack_class_has_reward_gradient():
    """Regression guard for the failure that motivated the reward rewrite.

    The original linear reward scored SlowRamp, Stealth and DoS within 1% of
    Normal, so an agent trained on it would have ignored them while the reward
    curve looked healthy. This test fails loudly if that recurs.
    """
    from mgxrl.envs.resilient_env import ResilientMicrogridEnv, EnvConfig

    env = ResilientMicrogridEnv(config=EnvConfig(seed=0))
    seeds = (0, 1, 2)
    base = float(np.mean([
        sum(r["reward"] for r in env.run_nominal(injector=NoAttack(), seed=s))
        for s in seeds]))
    assert base > 0, "healthy operation must not be net-penalised"

    # Averaged over seeds, not measured on one. DoS sits at ~28% margin with a
    # per-seed spread larger than its gap from Normal, so a single-seed
    # threshold test on it is a coin flip and would go red for reasons that
    # have nothing to do with a regression. See scripts/00_calibrate_reward.py,
    # which reports the gap in standard errors.
    for name in ("Additive", "Ramp", "SlowRamp", "Sinusoid", "Stealth", "DoS"):
        targets = (0,) if name != "Stealth" else (0, 1, 2)
        rets = [sum(x["reward"] for x in env.run_nominal(
                    injector=ATTACK_REGISTRY[name](targets=targets, t_start=1.5),
                    seed=s))
                for s in seeds]
        margin = abs(float(np.mean(rets)) - base) / abs(base)
        assert margin > 0.25, (
            f"{name}: only {margin:.1%} margin over {len(seeds)} seeds "
            f"-- no learning signal")


def test_reward_terms_are_not_saturated():
    """Log compression must keep gradients alive even at extreme severity."""
    from mgxrl.envs.resilient_env import ResilientMicrogridEnv, EnvConfig

    env = ResilientMicrogridEnv(config=EnvConfig(seed=0))
    inj = ATTACK_REGISTRY["Ramp"](targets=(0,), t_start=1.5)   # most severe class
    log = env.run_nominal(injector=inj, seed=0)
    keys = ("f_rmse", "v_rmse", "p_sharing_error", "q_sharing_error",
            "xi_spread_sustained", "zeta_spread_sustained")
    mm = {k: float(np.mean([r[k] for r in log])) for k in keys}
    assert max(env.reward_breakdown(mm).values()) < 40.0


def test_dos_severity_scales_with_frozen_links():
    """The consensus term must rank severity, not merely detect presence."""
    from mgxrl.envs.resilient_env import ResilientMicrogridEnv, EnvConfig

    env = ResilientMicrogridEnv(config=EnvConfig(seed=0))
    rets = [
        np.mean([sum(x["reward"] for x in env.run_nominal(
            injector=ATTACK_REGISTRY["DoS"](targets=tuple(range(n)), t_start=1.5), seed=s))
            for s in range(2)])
        for n in (1, 2, 4)
    ]
    assert rets[0] > rets[1] > rets[2], f"DoS severity not monotone: {rets}"


def test_benign_load_step_does_not_trigger_sustained_term():
    """A load step spikes instantaneous disagreement but must not sustain it --
    this is what separates benign disturbances from DoS."""
    from mgxrl.envs.resilient_env import ResilientMicrogridEnv, EnvConfig

    env = ResilientMicrogridEnv(config=EnvConfig(seed=0, n_load_steps=3))
    log = env.run_nominal(injector=NoAttack(), seed=0)
    sustained = np.array([r["xi_spread_sustained"] for r in log])
    instant = np.array([r["xi_spread"] for r in log])
    assert sustained.max() < instant.max(), "EMA filter is not attenuating transients"


# ------------------------------------------------- training-failure guards

def _constant_action_return(env, value, seed=0):
    from mgxrl.plant.attacks import NoAttack

    env.reset(seed=seed, options={"injector": NoAttack()})
    a = np.full(env.n_actions, value, dtype=np.float32)
    total = 0.0
    for _ in range(env.max_steps):
        _, r, term, trunc, _ = env.step(a)
        total += r
        if term or trunc:
            break
    return total


def test_isolation_is_expensive_and_the_flat_half_is_nearly_free():
    """Regression guard for the collapsed policy -- against the REAL reward.

    The previous version of this test reimplemented the cost as
    w_action*mean(a**2) + w_saturation*mean(|a|>0.9) and asserted that
    a = +1 was expensive. The reward has not charged that since the action
    space became trust weights: it charges mean(1 - trust) with
    trust = clip(1 + a, 0, 1), so a = +1 IS the nominal controller and costs
    almost nothing. The test passed only because it scored a formula the env
    no longer uses -- false assurance about exactly the degeneracy that let a
    trained policy drift to mean |a| = 0.87 with 40-55% of components parked
    in the reward-flat half.

    What is actually true, and what this now asserts:
      * a = -1 (isolating everything the safety floor permits) is expensive
      * a = +1 is behaviourally identical to a = 0 and costs only the small
        w_flat tie-break, which exists to break the degeneracy without
        reshaping behaviour
    """
    from mgxrl.envs.resilient_env import ResilientMicrogridEnv, EnvConfig, RewardWeights

    env = ResilientMicrogridEnv(config=EnvConfig(seed=0))
    quiet = _constant_action_return(env, 0.0)
    isolate = _constant_action_return(env, -1.0)
    full_trust = _constant_action_return(env, 1.0)

    assert quiet > 0, "healthy operation must not be net-penalised"
    assert isolate < 0.5 * quiet, (
        f"isolating every link scored {isolate:,.0f} against {quiet:,.0f} for "
        f"doing nothing -- dismantling the consensus layer must be a real cost"
    )

    w = RewardWeights()
    expected_tie_break = w.w_flat * env.max_steps
    gap = quiet - full_trust
    assert 0 <= gap < 3 * expected_tie_break, (
        f"a = +1 differs from a = 0 by {gap:,.1f}; the positive half of the "
        f"action space should cost only the w_flat tie-break "
        f"(~{expected_tie_break:,.1f})"
    )


def test_connectivity_floor_is_enforced():
    """The safety projection must keep the coordination layer alive.

    EnvConfig.min_trusted_links carried the comment 'see _project' while
    _project implemented only a rate limit and a directional constraint. The
    floor was documented, not enforced.
    """
    from mgxrl.envs.resilient_env import ResilientMicrogridEnv, EnvConfig
    from mgxrl.plant.attacks import NoAttack

    env = ResilientMicrogridEnv(config=EnvConfig(seed=0))
    env.reset(seed=0, options={"injector": NoAttack()})
    a = np.full(env.n_actions, -1.0, dtype=np.float32)
    for _ in range(50):
        _, _, term, trunc, info = env.step(a)
        applied = np.asarray(info["action_norm"], dtype=float)
        trust = np.clip(1.0 + applied, 0.0, 1.0)
        for blk in (trust[: env.n], trust[env.n :]):
            n_ok = int((blk >= env.cfg.trust_ok).sum())
            assert n_ok >= env.cfg.min_trusted_links, (
                f"only {n_ok} trusted channel(s); floor is "
                f"{env.cfg.min_trusted_links}"
            )
        if term or trunc:
            break


def test_curriculum_stages_change_the_distribution():
    """The overrides were unpacked and discarded, so all three stages trained
    on the identical full mix. The staged schedule was documentation only."""
    from dataclasses import replace
    from mgxrl.envs.resilient_env import EnvConfig, ResilientMicrogridEnv
    from mgxrl.agents.train_ppo import CURRICULUM

    base = EnvConfig(seed=0)
    seen = []
    for _, _, over in CURRICULUM:
        env = ResilientMicrogridEnv(config=replace(base, **over) if over else base)
        classes = {env.reset(seed=k)[1]["attack"] for k in range(40)}
        seen.append(classes)
    assert seen[0] == {"Normal"}, f"stage 1 must be nominal-only, got {seen[0]}"
    assert "Stealth" not in seen[1], "stage 2 must exclude the subtle classes"
    assert len(seen[2]) > len(seen[1]), "stage 3 must be the widest"


def test_oracle_action_can_cancel_an_attack():
    """The action space must be able to SOLVE the task. Run this before training.

    Two earlier action spaces failed this. A residual on dxi/dt could not
    restore nominal at all -- it only traded frequency against power sharing.
    An additive offset on the received signal could, but the useful precision
    was ~1/300 of the action range, so no policy could find it by sampling.
    Trust weights isolate the compromised link and recover almost everything.
    """
    from mgxrl.envs.resilient_env import ResilientMicrogridEnv, EnvConfig
    from mgxrl.plant.attacks import AdditiveBias, NoAttack

    env = ResilientMicrogridEnv(config=EnvConfig(seed=0))

    def run(inj, act):
        env.reset(seed=0, options={"injector": inj})
        total = 0.0
        for _ in range(env.max_steps):
            _, r, term, trunc, _ = env.step(act)
            total += r
            if term or trunc:
                break
        return total

    zero = np.zeros(env.n_actions, dtype=np.float32)
    isolate = np.zeros(env.n_actions, dtype=np.float32)
    isolate[0] = -1.0            # distrust DG1's frequency link
    isolate[env.n] = -1.0        # and its voltage link

    nominal = run(NoAttack(), zero)
    undefended = run(AdditiveBias(targets=(0,), t_start=0.0,
                                  bias_xi=0.3, bias_zeta=0.01), zero)
    defended = run(AdditiveBias(targets=(0,), t_start=0.0,
                                bias_xi=0.3, bias_zeta=0.01), isolate)

    assert nominal > 0 and undefended < 0
    recovered = (defended - undefended) / (nominal - undefended)
    assert recovered > 0.9, f"isolation recovered only {recovered:.1%} of the damage"


def test_zero_action_is_exactly_nominal():
    """a = 0 maps to trust = 1 on every link, which IS the published
    consensus controller."""
    from mgxrl.envs.resilient_env import ResilientMicrogridEnv, EnvConfig
    from mgxrl.plant.attacks import NoAttack

    env = ResilientMicrogridEnv(config=EnvConfig(seed=0))
    log = env.run_nominal(injector=NoAttack(), seed=0)
    with_zero = sum(x["reward"] for x in log)

    env2 = ResilientMicrogridEnv(config=EnvConfig(seed=0))
    env2.reset(seed=0, options={"injector": NoAttack()})
    total = 0.0
    for _ in range(env2.max_steps):
        _, r, te, tr, _ = env2.step(np.zeros(env2.n_actions, dtype=np.float32))
        total += r
        if te or tr:
            break
    assert abs(total - with_zero) < 1e-6


def test_exploration_is_cheap_when_nothing_is_wrong():
    """The reason the previous action space was untrainable.

    With an additive compensator, Gaussian exploration at std 0.368 cost
    ~46,000 reward per nominal episode -- PPO could not explore without
    destroying the grid. Under trust weights the neighbours already agree at
    consensus, so down-weighting a link costs almost nothing when there is no
    attack. Exploration has to be affordable or the policy cannot be found.
    """
    from mgxrl.envs.resilient_env import ResilientMicrogridEnv, EnvConfig
    from mgxrl.plant.attacks import NoAttack

    env = ResilientMicrogridEnv(config=EnvConfig(seed=0))
    rng = np.random.default_rng(0)

    def run(policy):
        env.reset(seed=0, options={"injector": NoAttack()})
        total = 0.0
        for _ in range(env.max_steps):
            _, r, te, tr, _ = env.step(policy())
            total += r
            if te or tr:
                break
        return total

    quiet = run(lambda: np.zeros(env.n_actions, dtype=np.float32))
    noisy = run(lambda: rng.normal(0, 0.368, env.n_actions).astype(np.float32))
    assert quiet > 0
    assert noisy > 0.5 * quiet, (
        f"exploration costs {quiet - noisy:.0f} of {quiet:.0f}; PPO cannot "
        f"explore this action space"
    )
