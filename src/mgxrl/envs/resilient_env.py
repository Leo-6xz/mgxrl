"""
Gymnasium environment for attack-resilient secondary control.

Two design decisions here carry our paper, so they are worth stating plainly:

1. RESIDUAL POLICY. The nominal consensus controller (eqs. 3-6) stays in the
   loop at all times. The agent outputs a bounded *correction* on top of it.
   This means (a) the untrained policy is already a working controller, so
   early episodes don't collapse the grid, (b) sample efficiency improves by
   roughly an order of magnitude, and (c) states a nominal-case
   stability argument, because with the residual saturated to zero the closed
   loop reduces exactly to the published consensus controller.

2. SAFETY PROJECTION. Actions are projected onto a feasible set before they
   reach the plant: magnitude bound, rate bound, and a directional constraint
   that forbids corrections which push a variable further outside its envelope.
   The projection is model-based -- that, plus the physics terms in the reward,
   is what makes the "physics-informed" claim defensible rather than decorative.

Deliberately NOT included: an interpretability penalty in the reward. Shaping
the control objective with an unvalidated proxy for explainability corrupts the
objective and invites the question "how did you validate that proxy?" We keep
the policy clean and validate the explanation layer post-hoc instead
(mgxrl/xai/faithfulness.py).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from ..plant.microgrid import Microgrid, MicrogridParams, TWO_PI
from ..plant.attacks import AttackInjector, NoAttack, sample_attack

try:  # keep the plant importable without a full RL stack installed
    import gymnasium as gym
    from gymnasium import spaces
    _BASE = gym.Env
except ImportError:  # pragma: no cover
    gym = None
    spaces = None
    _BASE = object


@dataclass
class RewardWeights:
    """Scale-normalised reward.

    WHY THE FIRST VERSION FAILED
    Fixed linear weights on raw metrics cannot work here, because the metrics
    live on incompatible scales: f_rmse spans 3e-4 to 5e-2, v_rmse 1e-4 to
    1e-2, p_sharing 3e-3 to 3e-1. Any weight large enough to make Coordinated
    Stealth register makes Ramp saturate the return, and any weight small
    enough to keep Ramp finite makes Stealth invisible. Measured on the
    undefended baseline, SlowRamp, Stealth and DoS all scored within 1% of
    Normal -- the agent would have learned to ignore exactly the three attacks
    our paper claims to handle.

    THE FIX, three parts:

    1. NORMALISE each metric by a reference value representing good control,
       so every term is expressed in units of "times worse than well-regulated
       nominal operation" and all terms are commensurable.

    2. SOFT-SATURATE with cap*e/(cap+e) rather than clipping. A hard clip has
       zero gradient in the saturated region, so the agent gets no signal to
       improve from catastrophic to merely bad. The rational form is smooth and
       strictly monotone, so the gradient survives everywhere while a single
       Ramp episode still cannot dominate the whole return.

    3. PENALISE CONSENSUS DISAGREEMENT (xi_spread / zeta_spread). This is the
       only term that responds to DoS, which perturbs neither voltage nor
       frequency appreciably but drives correction-state disagreement up ~60x.

    The reward uses NO attack label. That is deliberate: the same policy must
    restore performance whether the cause is an attack or an ordinary load
    step, and a label-dependent reward would not transfer to deployment.
    """

    # --- reference scales ---
    # These are NOT guesses. They are the 90th percentile of each metric
    # measured over 12 undefended Normal episodes with benign load steps
    # (reproduce with scripts/00_calibrate_reward.py). Setting them this way
    # makes every term ~1.0 during healthy operation, so the terms are
    # commensurable and `alive_bonus` has a defensible value rather than a
    # tuned one.
    #
    # Note ref_q_share is ~30x larger than ref_p_share. That is correct and
    # worth a sentence: reactive-power sharing is inherently poor
    # under Q-V droop because voltage is a local quantity, not a global one.
    # A reward that treats P and Q dispersion symmetrically would spend its
    # entire budget fighting normal droop behaviour.
    ref_f_rmse: float = 5.0e-4         # Hz
    ref_v_rmse: float = 1.8e-3         # pu
    ref_p_share: float = 6.0e-3
    ref_q_share: float = 1.8e-1
    # Measured by scripts/00_calibrate_reward.py as the p90 over 12 undefended
    # Normal episodes. The previous values (1.2e-3 / 4.0e-4) predated benign
    # load steps being mandatory, and were 1.8x and 2.6x too small. Because
    # each term is (metric/ref)^2 inside log1p, that charged roughly 1,640
    # reward per episode against HEALTHY operation -- about 15% of the alive
    # bonus -- for consensus disagreement that is ordinary load-step response.
    # Adopting these raised the undefended Normal return from 4,691 to 6,330.
    # Re-measure after any change to the plant, the load model, or n_load_steps.
    ref_xi_spread: float = 2.166e-3    # rad/s, on the SUSTAINED (EMA) value
    ref_zeta_spread: float = 1.057e-3  # pu, sustained

    # Time constant for the sustained-disagreement filter. Benign load steps
    # produce a brief spike in consensus disagreement; DoS produces a permanent
    # offset. Without this filter the two are indistinguishable by magnitude,
    # and the DoS term fires on ordinary switching events instead.
    consensus_tau: float = 2.0         # s

    # --- weights on the normalised, saturated terms ---
    w_f: float = 3.0
    w_v: float = 2.0
    w_p_share: float = 1.5
    w_q_share: float = 1.0
    w_xi_spread: float = 2.5           # the DoS / stealth channel
    w_zeta_spread: float = 1.5
    w_v_violation: float = 50.0        # hard envelope breach, unnormalised

    # --- control cost ---
    # w_action was 0.10, which made a permanently saturated actuator cost 1.1%
    # of the alive bonus and ~0.3% of the penalty it was avoiding. Saturation
    # was effectively free, and PPO duly collapsed to a constant maximal
    # action: |a| = 0.999 on 99.8% of steps INCLUDING healthy operation,
    # leaving the agent worse than the undefended controller on five of seven
    # attack classes. Control effort has to be a real cost or there is no
    # pressure to act only when acting helps.
    w_action: float = 1.0
    w_smooth: float = 0.40
    w_saturation: float = 6.0           # explicit penalty for pinning the actuator
    saturation_threshold: float = 0.9

    # Tie-break across the reward-flat half of the action space. trust =
    # clip(1 + a, 0, 1), so every a >= 0 is full trust and behaviourally
    # identical: the whole positive orthant has zero reward gradient and the
    # only term touching it was w_smooth, which penalises MOVEMENT and so holds
    # the policy wherever it drifts. Measured: 50-63% of action components sat
    # in that region (eval3_s0..2: 0.500, 0.574, 0.630) with mean |a| up to 0.87 after 5M steps, while
    # action_magnitude rose 0.25 -> 0.67 and std rose 0.368 -> 0.389.
    # At a = +0.87 this costs 0.038/step, ~46 per 1200-step episode, against
    # returns in the thousands -- it breaks the tie without reshaping
    # behaviour. Its derivative at a = 0 is zero, so unlike a penalty on |a|
    # it does not push the policy off the nominal optimum.
    w_flat: float = 0.05

    sat_cap: float = 1.0               # scale on the log-compressed terms
    collapse_penalty: float = 200.0
    alive_bonus: float = 9.0           # ~= total penalty of healthy operation


def _compress(e2: float, scale: float) -> float:
    """Logarithmic compression of a normalised squared error.

    Rational saturation (cap*x/(cap+x)) was the obvious choice and it is wrong.
    At the severity a Ramp attack reaches -- normalised error ~100, so e^2 ~1e4
    against a cap of 50 -- its derivative is ~2.6e-5. The term is numerically
    pinned, so the agent receives no gradient on the most damaging attacks until
    it has already fixed them, which it has no incentive to start doing. The
    reward would look well-shaped and teach nothing.

    log1p is unbounded, strictly monotone, and compresses decades smoothly: at
    e=100 its derivative is still ~0.02, roughly a thousand times larger. The
    dynamic range across attack classes stays ~13x instead of ~50x, which is
    ample for ranking while keeping every class learnable.
    """
    return scale * float(np.log1p(max(e2, 0.0)))


@dataclass
class EnvConfig:
    # 12 s, not 6 s. SlowRamp at 0.004 pu/s needs ~10 s post-onset to reach a
    # deviation that matters; on a 6 s episode the attack is still ramping when
    # the episode ends and the agent never sees the consequence it is meant to
    # prevent. Slow attacks require a horizon proportional to their timescale.
    episode_seconds: float = 12.0
    settle_seconds: float = 1.5
    # Retained for the reference-trajectory exporter; the trust-weight action
    # space does not use them.
    residual_xi_max: float = 1.5
    residual_zeta_max: float = 0.04
    # Connectivity floor, enforced per channel in _project. The agent may not
    # dismantle the coordination layer it is defending: at least this many of
    # the n channels must keep trust >= trust_ok, in each of the xi and zeta
    # blocks independently. This was declared here but never implemented; the
    # projection did a rate limit and a directional constraint and nothing else.
    min_trusted_links: int = 3
    trust_ok: float = 0.5               # trust at or above this counts as trusted
    residual_rate_max: float = 0.25     # fraction of bound per control step
    history_len: int = 4                # observation stacking (temporal context)
    randomize_load: bool = True
    # Append the detector's 7-class posterior to the observation.
    #
    # DEFAULT IS FALSE, AND THAT IS A DELIBERATE CHOICE, NOT AN OVERSIGHT.
    # This flag read True from the day it was written while `detector` defaulted
    # to None everywhere above it, so `_det_dim` was 0 and the posterior was
    # never appended. All three 5M-step runs trained at obs_dim = 320 with the
    # flag nominally on. Leaving the default True would keep claiming a
    # capability the trained agents do not have.
    #
    # Enabling it changes obs_dim from 320 to 327. That invalidates every
    # existing checkpoint: SB3 will refuse to load them against the new space,
    # which is the good failure. Set it True only together with a real
    # detector, and expect to retrain.
    include_detector: bool = False

    # Window length for the in-the-loop detector, in CONTROL STEPS, matching
    # `window` in features.window_features. The benchmark uses 20.
    #
    # TIMESCALE WARNING, and it is not fixable in code. The corpus from
    # scripts/09 is written at --log-dt 0.001 and benchmarked with
    # --decimate 1, so a 20-sample window there spans 20 ms. The env steps at
    # control_dt = 0.010, so the same 20 samples span 200 ms. Every slope, std
    # and meanabsdiff feature would then mean something ten times slower than
    # what the booster was fit on. Generate a second corpus at
    # --log-dt 0.01 for any detector used in the loop.
    detector_window: int = 20
    detector_windowed: bool = True      # False = pointwise features, no buffer
    safety_projection: bool = True
    p_normal: float = 1 / 7             # probability an episode has no attack
    attack_subset: tuple = ()           # restrict classes; () = all
    n_load_steps: int = 3               # benign disturbances per episode
    load_step_mag: float = 0.15         # +/- fraction of nominal load
    seed: int | None = None


class ResilientMicrogridEnv(_BASE):
    """Observation: history_len stacked frames of EIGHT quantities per DG --
    V, f, P, Q, xi, zeta, and the per-source consensus innovation on the xi and
    zeta channels -- optionally followed by the detector's 7-class posterior.

    The innovation is what makes the per-source trust action learnable at all;
    without it the policy is asked to attribute corruption with no per-source
    evidence. See _innovation and scripts/13_observability_probe.py.

    The detector posterior is OFF by default and was inert for every run to
    date, so the trained policies act on physical measurements and the
    consensus innovation alone. That is worth stating rather than hiding: the
    detector and the controller are independent, so a detector false positive
    cannot propagate into a control action.

    Action: per-source trust offsets [a_xi_1..a_xi_10, a_zeta_1..a_zeta_10] in
    [-1, 1], applied as trust_j = clip(1 + a_j, 0, 1) on the signal
    TRANSMITTED by DG j. NOT a residual on dxi/dzeta -- that was the first
    action space and it was abandoned; see mg_dynamics.m for why deploying
    through that path fails silently.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        plant_params: MicrogridParams | None = None,
        config: EnvConfig | None = None,
        weights: RewardWeights | None = None,
        detector=None,
    ) -> None:
        self.cfg = config or EnvConfig()
        self.w = weights or RewardWeights()
        self.pp = plant_params or MicrogridParams(seed=self.cfg.seed)
        self.mg = Microgrid(self.pp)
        self.n = self.mg.n
        self.rng = np.random.default_rng(self.cfg.seed)
        self.detector = detector

        self.n_actions = 2 * self.n
        # V,f,P,Q,xi,zeta + the per-source consensus innovation on both channels
        self._per_step_feats = 8 * self.n
        self._deg = self.mg.adj.sum(axis=0)
        self._deg_safe = np.where(self._deg > 0, self._deg, 1.0)
        # The old form was
        #     7 if (include_detector and detector is not None) else 0
        # which silently produced a 320-wide observation whenever the caller
        # forgot the detector, with the flag still reading True. Nothing
        # errored, nothing warned, and three 5M-step runs trained without the
        # feature the config said they had. Ask for it and it is missing ->
        # stop.
        if self.cfg.include_detector and detector is None:
            raise ValueError(
                "include_detector=True but no detector was passed. Load one "
                "and pass detector=..., or set include_detector=False. "
                "Silently dropping it changes obs_dim from "
                f"{self.cfg.history_len * self._per_step_feats + 7} to "
                f"{self.cfg.history_len * self._per_step_feats} and trains a "
                "different agent than the config describes."
            )
        self._det_dim = 7 if self.cfg.include_detector else 0

        # Raw-row buffer for the windowed detector features, and the column
        # names that to_dataset_row produces AFTER dataio._canonical_columns
        # normalisation. The env must feed the booster the same names in the
        # same order the benchmark did, or the feature vector is the right
        # width and the wrong content -- which no dimension check would catch.
        self._det_buf: list[np.ndarray] = []
        self._det_cols = (
            ["v1", "v2", "v3", "i1", "i2", "i3"]
            + [f"{q}_dg{i}" for i in range(1, self.n + 1) for q in ("p", "q", "f")]
        )
        if self._det_dim:
            self._check_detector_width(detector)
        self.obs_dim = self.cfg.history_len * self._per_step_feats + self._det_dim

        if spaces is not None:
            self.action_space = spaces.Box(-1.0, 1.0, (self.n_actions,), np.float32)
            self.observation_space = spaces.Box(
                -np.inf, np.inf, (self.obs_dim,), np.float32
            )

        self.max_steps = int(round(self.cfg.episode_seconds / self.pp.control_dt))
        self._reset_internals()

    # ------------------------------------------------------------------ utils

    def _reset_internals(self) -> None:
        self._prev_action = np.zeros(self.n_actions)
        self._history: list[np.ndarray] = []
        # Cleared per episode: a window carried across a reset would mix the
        # tail of one trajectory with the head of the next, and the detector
        # would see a step change that never happened.
        self._det_buf = []
        self._step_count = 0
        self._ema = {"xi_spread": 0.0, "zeta_spread": 0.0}
        self.injector: AttackInjector = NoAttack()
        self.episode_log: list[dict] = []
        self._load_steps: list = []

    def _scale_action(self, a: np.ndarray) -> np.ndarray:
        """Actions are per-link trust offsets and need no unit scaling.

        The plant maps them as w = clip(1 + a, 0, 1): a = 0 is full trust and
        exactly the nominal controller, a = -1 isolates the link. Positive
        values saturate at full trust and are harmless, which is what makes
        exploration cheap.
        """
        return np.clip(np.asarray(a, dtype=float), -1.0, 1.0)

    def _project(self, a_raw: np.ndarray, meas: dict) -> np.ndarray:
        """Model-based safety projection. Returns a *normalised* action."""
        a = np.clip(np.asarray(a_raw, dtype=float), -1.0, 1.0)
        if not self.cfg.safety_projection:
            return a

        # 1) Rate limit -- prevents chattering that the plant cannot follow.
        lo = self._prev_action - self.cfg.residual_rate_max
        hi = self._prev_action + self.cfg.residual_rate_max
        a = np.clip(a, lo, hi)

        # 2) Connectivity floor -- the model-based constraint for this action
        #    space. The old directional constraint was written for the additive
        #    offset action space, where a[:n] really was a residual on xi and
        #    its sign meant "raises / lowers frequency". Under trust weights the
        #    sign means "trusts / distrusts a neighbour", so zeroing a positive
        #    component was a no-op (positive already clips to full trust) and
        #    zeroing a negative one forced the agent to trust a link at exactly
        #    the moment the variable was outside its envelope. It is removed
        #    rather than repaired: there is no admissible-direction test on a
        #    trust weight without knowing which neighbour is anomalous, and that
        #    is the detector's job, not the projection's.
        d_xi = self._enforce_connectivity(a[: self.n])
        d_zeta = self._enforce_connectivity(a[self.n :])
        return np.concatenate([d_xi, d_zeta])

    def _enforce_connectivity(self, block: np.ndarray) -> np.ndarray:
        """Restore trust on the least-distrusted channels until the floor holds.

        Minimal intervention: the channels promoted are the ones already closest
        to trusted, and they are raised only to trust_ok, not to full trust. The
        agent keeps whatever isolation it can afford under the constraint.
        """
        k = int(self.cfg.min_trusted_links)
        if k <= 0:
            return block
        trust = np.clip(1.0 + block, 0.0, 1.0)
        ok = trust >= self.cfg.trust_ok
        deficit = k - int(ok.sum())
        if deficit <= 0:
            return block
        order = [i for i in np.argsort(-trust) if not ok[i]][:deficit]
        out = np.array(block, dtype=float, copy=True)
        out[order] = np.maximum(out[order], self.cfg.trust_ok - 1.0)
        return out

    def _check_detector_width(self, detector) -> None:
        """Fail at construction, not 30 seconds into a SubprocVecEnv worker."""
        import pandas as pd
        from ..detection.features import physics_features

        probe = pd.DataFrame(np.zeros((2, len(self._det_cols))),
                             columns=self._det_cols)
        n_feat = len(self._det_cols) + physics_features(probe).shape[1]
        if self.cfg.detector_windowed:
            n_feat *= 5                      # mean, std, slope, range, meanabsdiff
        got = getattr(detector, "n_features_in_", None)
        if got is not None and int(got) != n_feat:
            raise ValueError(
                f"detector expects {got} features; this env produces {n_feat} "
                f"({'windowed' if self.cfg.detector_windowed else 'pointwise'}"
                f", {len(self._det_cols)} raw + physics). Train the detector "
                f"with the same --windowed setting, on a corpus generated at "
                f"the control rate (scripts/09 --log-dt "
                f"{self.pp.control_dt:g})."
            )

    def _detector_features(self, meas: dict) -> np.ndarray:
        if self._det_dim == 0:
            return np.empty(0)

        # Reusing the benchmark's own transform rather than reimplementing it.
        # A parallel implementation drifts the moment either side changes, and
        # the drift is invisible: the vector keeps its width and quietly stops
        # meaning the same thing.
        import pandas as pd
        from ..detection.features import physics_features, _slope

        row = self.mg.to_dataset_row(meas)[1:]          # drop the time column
        self._det_buf.append(np.asarray(row, dtype=np.float32))
        w = self.cfg.detector_window if self.cfg.detector_windowed else 1
        # Warm-up: repeat the first frame until the buffer fills, exactly as
        # _observe does for history_len. Zero-padding would teach the policy
        # that an all-zero posterior means "episode just started" -- a value
        # the detector never actually emits.
        while len(self._det_buf) < w:
            self._det_buf.insert(0, self._det_buf[0])
        self._det_buf = self._det_buf[-w:]

        frame = pd.DataFrame(np.vstack(self._det_buf), columns=self._det_cols)
        phys = physics_features(frame)
        arr = np.concatenate([frame.to_numpy(np.float32),
                              phys.to_numpy(np.float32)], axis=1)

        if self.cfg.detector_windowed:
            x = np.concatenate([arr.mean(0), arr.std(0), _slope(arr),
                                arr.max(0) - arr.min(0),
                                np.abs(np.diff(arr, axis=0)).mean(0)])
        else:
            x = arr[-1]

        # No blanket except here. A detector that raises on every call would
        # otherwise return zeros forever, and the agent would train on seven
        # dead inputs while the config claims a detection-informed policy.
        try:
            proba = np.asarray(
                self.detector.predict_proba(x.reshape(1, -1)))[0]
        except Exception as exc:                        # noqa: BLE001
            raise RuntimeError(
                f"detector.predict_proba failed on a {x.size}-feature vector: "
                f"{exc}"
            ) from exc
        if proba.size != 7:
            raise RuntimeError(
                f"detector returned {proba.size} class probabilities, expected "
                "7 (Normal + 6 attack classes). obs_dim assumes 7."
            )
        return proba.astype(float)

    def _innovation(self, meas: dict):
        """Per-source consensus innovation: inn[j] = mean_i adj[i,j]*(x_i - x_rx[j]).

        How far source j's TRANSMITTED value sits from what its receivers hold.
        Each receiver can compute its own term from locally available data, so
        this adds no information a deployed controller could not have -- it is
        the statistic W-MSR thresholds on, and it is the only per-source
        evidence in the observation. Without it the policy must infer which
        transmitter is lying from aggregate physical dynamics alone.
        """
        adj = self.mg.adj
        inn_xi = (adj.T @ meas["xi"] - self._deg * meas["xi_rx"]) / self._deg_safe
        inn_zeta = (adj.T @ meas["zeta"] - self._deg * meas["zeta_rx"]) / self._deg_safe
        return inn_xi, inn_zeta

    def _observe(self, meas: dict) -> np.ndarray:
        inn_xi, inn_zeta = self._innovation(meas)
        frame = np.concatenate(
            [
                meas["V"] - self.pp.v_nom,
                (meas["f"] - self.pp.f_nom) * 10.0,          # scale to O(1)
                meas["P"] - self.pp.p_set,
                meas["Q"] - self.pp.q_set,
                meas["xi"] / self.pp.xi_limit,
                meas["zeta"] / self.pp.zeta_limit,
                inn_xi / self.pp.xi_limit,
                inn_zeta / self.pp.zeta_limit,
            ]
        )
        self._history.append(frame)
        while len(self._history) < self.cfg.history_len:
            self._history.insert(0, frame)
        self._history = self._history[-self.cfg.history_len :]
        obs = np.concatenate(self._history + [self._detector_features(meas)])
        return obs.astype(np.float32)

    # ------------------------------------------------------------------- API

    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.mg.rng = np.random.default_rng(seed)
        self._reset_internals()

        options = options or {}
        self.injector = options.get(
            "injector",
            sample_attack(
                self.rng,
                n_dg=self.n,
                t_window=(0.5, self.cfg.episode_seconds * 0.6),
                p_normal=self.cfg.p_normal,
                subset=tuple(self.cfg.attack_subset) or None,
            ),
        )
        self.mg.attach_attack(self.injector)
        meas = self.mg.reset(
            randomize=self.cfg.randomize_load and options.get("randomize", True),
            settle_time=self.cfg.settle_seconds,
        )
        self._schedule_load_steps()
        return self._observe(meas), {"attack": self.injector.name}

    def _schedule_load_steps(self) -> None:
        """Draw benign load-step times. These are the disturbances the agent
        must NOT mistake for an attack; the false-trigger rate on them is a
        headline metric, not an afterthought."""
        k = self.cfg.n_load_steps
        self._load_steps = []
        if k <= 0:
            return
        times = self.rng.uniform(0.2, self.cfg.episode_seconds * 0.9, size=k)
        for t in np.sort(times):
            delta = self.rng.uniform(-self.cfg.load_step_mag, self.cfg.load_step_mag)
            bus = int(self.rng.integers(0, self.n))
            self._load_steps.append((float(t), bus, float(delta)))

    def _service_load_steps(self, t: float) -> bool:
        fired = False
        while self._load_steps and t >= self._load_steps[0][0]:
            _, bus, delta = self._load_steps.pop(0)
            scale = self.mg.load_scale.copy()
            scale[bus] *= 1.0 + delta
            self.mg.apply_load_step(scale)
            fired = True
        return fired

    def step(self, action):
        self._service_load_steps(self.mg.state.t)
        a_norm = self._project(action, self.mg.measure(noisy=False))
        residual = self._scale_action(a_norm)
        meas = self.mg.step(residual=residual, attack_active=True)

        m = self.mg.metrics(meas)
        m.update(self._sustained(m))
        reward = self._reward(m, a_norm, meas)

        self._prev_action = a_norm
        self._step_count += 1

        y_bin, y_multi = self.injector.label(meas["t"])
        terminated = bool(meas["collapsed"])
        truncated = self._step_count >= self.max_steps

        info = {
            **m,
            "t": meas["t"],
            "attack": self.injector.name,
            "y_bin": y_bin,
            "y_multi": y_multi,
            "action_norm": a_norm,
            "raw_measurement": meas,
        }
        self.episode_log.append(info)
        return self._observe(meas), float(reward), terminated, truncated, info

    def _sustained(self, m: dict) -> dict:
        """Exponentially-weighted consensus disagreement.

        This is what separates a DoS attack from an ordinary load step. Both
        spike the instantaneous disagreement; only DoS holds it there, because
        the frozen link permanently prevents the correction states from
        re-agreeing. Filtering with tau=2 s keeps the sustained component and
        discards the transient, so the term fires on the attack and not on
        normal switching.
        """
        a = self.pp.control_dt / max(self.w.consensus_tau, self.pp.control_dt)
        for k in ("xi_spread", "zeta_spread"):
            self._ema[k] = (1 - a) * self._ema[k] + a * m[k]
        return {"xi_spread_sustained": self._ema["xi_spread"],
                "zeta_spread_sustained": self._ema["zeta_spread"]}

    def _reward(self, m: dict, a_norm: np.ndarray, meas: dict) -> float:
        w = self.w
        cap = w.sat_cap

        def term(value: float, ref: float) -> float:
            """Normalised squared error, log-compressed."""
            return _compress((value / ref) ** 2, cap)

        penalty = (
            w.w_f * term(m["f_rmse"], w.ref_f_rmse)
            + w.w_v * term(m["v_rmse"], w.ref_v_rmse)
            + w.w_p_share * term(m["p_sharing_error"], w.ref_p_share)
            + w.w_q_share * term(m["q_sharing_error"], w.ref_q_share)
            + w.w_xi_spread * term(m["xi_spread_sustained"], w.ref_xi_spread)
            + w.w_zeta_spread * term(m["zeta_spread_sustained"], w.ref_zeta_spread)
            + w.w_v_violation * m["v_violation"]
        )
        # Saturation is penalised separately from magnitude. A quadratic cost
        # alone barely distinguishes |a| = 0.9 from |a| = 1.0, yet the second
        # means the actuator is pinned and the policy has stopped modulating.
        # Cost is CONNECTIVITY LOSS, not action magnitude. Positive actions
        # saturate at full trust and do nothing, so penalising |a| would tax a
        # harmless half of the action space and push the policy off the
        # nominal optimum at a = 0.
        trust = np.clip(1.0 + a_norm, 0.0, 1.0)
        isolation = float(np.mean(1.0 - trust))
        control_cost = (
            w.w_action * isolation
            + w.w_smooth * float(np.mean((a_norm - self._prev_action) ** 2))
            + w.w_saturation * float(isolation > 0.5)
            + w.w_flat * float(np.mean(np.maximum(a_norm, 0.0) ** 2))
        )
        r = w.alive_bonus - penalty - control_cost
        if meas["collapsed"]:
            r -= w.collapse_penalty
        return r

    def reward_breakdown(self, m: dict) -> dict:
        """Per-term contributions, for tuning.

        Call this on the undefended baseline for each attack class before
        training. If a class shows no term above ~1.0, the agent has no gradient
        on it and training will silently ignore that attack.
        """
        w = self.w
        pairs = [
            ("f_rmse", w.ref_f_rmse, w.w_f),
            ("v_rmse", w.ref_v_rmse, w.w_v),
            ("p_sharing_error", w.ref_p_share, w.w_p_share),
            ("q_sharing_error", w.ref_q_share, w.w_q_share),
            ("xi_spread_sustained", w.ref_xi_spread, w.w_xi_spread),
            ("zeta_spread_sustained", w.ref_zeta_spread, w.w_zeta_spread),
        ]
        return {k: wt * _compress((m[k] / ref) ** 2, w.sat_cap) for k, ref, wt in pairs}

    # --------------------------------------------------------------- baselines

    def run_nominal(self, injector: AttackInjector | None = None, seed: int | None = None):
        """Roll out the *unmodified* consensus controller under the same attack.

        This is our primary baseline: our own controller, undefended.
        Every resilience number we reported was against this, not against
        an uncontrolled grid.
        """
        opts = {"injector": injector} if injector is not None else {}
        self.reset(seed=seed, options=opts)
        zero = np.zeros(self.n_actions, dtype=np.float32)
        log = []
        for _ in range(self.max_steps):
            _, r, term, trunc, info = self.step(zero)
            log.append({**{k: v for k, v in info.items() if k != "raw_measurement"},
                        "reward": r})
            if term or trunc:
                break
        return log

    def config_dict(self) -> dict:
        return {
            "plant": asdict(self.pp),
            "env": asdict(self.cfg),
            "reward": asdict(self.w),
        }
