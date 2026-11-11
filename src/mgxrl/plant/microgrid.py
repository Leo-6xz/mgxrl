"""
Reduced-order averaged model of a 10-DG islanded inverter-based AC microgrid.

This is the RL *training* plant. It deliberately omits PWM switching and the
inner voltage/current loops, because those dynamics are 3-4 orders of magnitude
faster than the secondary-control timescale the agent acts on (100-200 ms).
Retaining them would make RL training computationally infeasible while adding
nothing the agent can act upon.

Model structure (following Pogaku et al. 2007 and the droop/consensus equations
of Ogiesoba-Eguakun & Rath):

    Power filtering      tau_f * dP_i/dt = P_meas_i - P_i           (and Q)
    Primary droop        w_i = w* - m_i (P_i - P*_i) + xi_i         (eq. 1)
                         V_i = V* - n_i (Q_i - Q*_i) + zeta_i       (eq. 2)
    Angle dynamics       ddelta_i/dt = w_i - w_ref
    Secondary consensus  dxi_i/dt   = -k_p (w_i - w*) - c_f * sum_j a_ij (xi_i - xi_j~)    (eq. 3)
                         dzeta_i/dt = -k_q (V_i - V*) - c_v * sum_j a_ij (zeta_i - zeta_j~) (eq. 5)

where xi_j~ / zeta_j~ are the values *as received over the communication link*,
i.e. the attack surface. Corrupting those (rather than the local states) is what
makes the injected attack propagate through the consensus graph exactly as
described in Section III-C of the source paper.

Network solution is algebraic: constant-impedance loads are folded into the bus
admittance matrix and injections are computed from the standard power-flow
equations. All quantities are per-unit unless noted.

Fidelity caveat: this model reproduces secondary-control
*dynamics* and attack propagation, not switching ripple or harmonics. The
Simulink model remains the validation target (see matlab/validate_fidelity.m).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

TWO_PI = 2.0 * np.pi


@dataclass
class MicrogridParams:
    """Physical and control parameters. Defaults give a stable 10-DG ring."""

    n_dg: int = 10

    # --- Base / nominal ---
    f_nom: float = 50.0                 # Hz
    v_nom: float = 1.0                  # pu
    s_base: float = 10e3                # VA per DG (for reporting only)
    v_base: float = 380.0               # V, line-to-line (for reporting only)

    # --- Primary droop ---
    # m: rad/s per pu of P. 0.5% frequency droop over 1.0 pu power swing.
    m_droop: float = 0.005 * TWO_PI * 50.0
    # n: pu volts per pu of Q. 5% voltage droop over 1.0 pu reactive swing.
    n_droop: float = 0.05
    tau_f: float = 0.016                # power measurement filter, s (~10 Hz)

    # --- Distributed secondary consensus (eqs. 3-6) ---
    k_p: float = 5.0                    # frequency restoration gain
    c_f: float = 10.0                   # frequency consensus coupling
    k_q: float = 5.0                    # voltage restoration gain
    c_v: float = 10.0                   # voltage consensus coupling
    xi_limit: float = 0.10 * TWO_PI * 50.0   # saturation on xi, rad/s
    zeta_limit: float = 0.10                 # saturation on zeta, pu

    # --- Network (ring of n_dg buses) ---
    r_line: float = 0.05                # pu, per segment
    x_line: float = 0.15                # pu, per segment
    load_p: float = 0.65                # pu per bus, nominal
    load_q: float = 0.28                # pu per bus, nominal

    # --- Setpoints ---
    p_set: float = 0.50                 # pu
    q_set: float = 0.20                 # pu

    # --- Communication graph (ring; matches Fig. 3 of the source paper) ---
    comm_topology: str = "ring"

    # --- Measurement realism (statistics fitted from the Dataport CSVs;
    #      see scripts/01_fit_noise.py) ---
    noise_v: float = 2.0e-4             # pu, std of voltage measurement noise
    noise_p: float = 1.0e-3             # pu
    noise_q: float = 1.0e-3             # pu
    noise_f: float = 2.0e-4             # Hz
    quantization_v: float = 0.0         # pu LSB; 0 disables

    # --- Integration ---
    dt: float = 1.0e-3                  # s, plant integration step
    control_dt: float = 0.010           # s, secondary-control / RL step

    # --- Safety envelope ---
    v_min: float = 0.95
    v_max: float = 1.05
    f_tol: float = 0.5                  # Hz; beyond this, declare collapse

    seed: int | None = None

    def __post_init__(self) -> None:
        if self.control_dt < self.dt:
            raise ValueError("control_dt must be >= dt")
        self.substeps = int(round(self.control_dt / self.dt))


@dataclass
class MicrogridState:
    """Continuous state vector of the plant."""

    delta: np.ndarray       # (n,) bus angles, rad
    p_filt: np.ndarray      # (n,) filtered active power, pu
    q_filt: np.ndarray      # (n,) filtered reactive power, pu
    xi: np.ndarray          # (n,) secondary frequency correction, rad/s
    zeta: np.ndarray        # (n,) secondary voltage correction, pu
    t: float = 0.0

    def copy(self) -> "MicrogridState":
        return MicrogridState(
            self.delta.copy(), self.p_filt.copy(), self.q_filt.copy(),
            self.xi.copy(), self.zeta.copy(), self.t,
        )


def ring_adjacency(n: int) -> np.ndarray:
    """Undirected ring communication graph, as in Fig. 3 of the source paper."""
    a = np.zeros((n, n))
    for i in range(n):
        a[i, (i + 1) % n] = 1.0
        a[i, (i - 1) % n] = 1.0
    return a


def build_ybus(params: MicrogridParams, load_scale: np.ndarray | None = None) -> np.ndarray:
    """Bus admittance matrix for a ring network with constant-impedance loads."""
    n = params.n_dg
    z = complex(params.r_line, params.x_line)
    y_line = 1.0 / z
    y = np.zeros((n, n), dtype=complex)

    for i in range(n):
        j = (i + 1) % n
        y[i, i] += y_line
        y[j, j] += y_line
        y[i, j] -= y_line
        y[j, i] -= y_line

    scale = np.ones(n) if load_scale is None else np.asarray(load_scale, dtype=float)
    # Constant-impedance load at V = 1.0 pu: Y_load = (P - jQ) / |V|^2
    y_load = (params.load_p * scale) - 1j * (params.load_q * scale)
    y[np.diag_indices(n)] += y_load
    return y


class Microgrid:
    """Averaged 10-DG islanded microgrid with droop primary + consensus secondary.

    Usage
    -----
    >>> mg = Microgrid(MicrogridParams(seed=0))
    >>> mg.reset()
    >>> meas = mg.step(control_dt=0.01)          # nominal consensus control
    >>> meas = mg.step(residual=np.zeros(20))    # with an RL residual action
    """

    def __init__(self, params: MicrogridParams | None = None) -> None:
        self.p = params or MicrogridParams()
        self.n = self.p.n_dg
        self.rng = np.random.default_rng(self.p.seed)
        self.adj = ring_adjacency(self.n)
        self.w_nom = TWO_PI * self.p.f_nom
        self.attack = None          # set via attach_attack()
        self.load_scale = np.ones(self.n)
        self.ybus = build_ybus(self.p, self.load_scale)
        self.state = self._initial_state()
        self.collapsed = False

    # ------------------------------------------------------------------ setup

    def _initial_state(self) -> MicrogridState:
        n = self.n
        return MicrogridState(
            delta=np.zeros(n),
            p_filt=np.full(n, self.p.load_p),
            q_filt=np.full(n, self.p.load_q),
            xi=np.zeros(n),
            zeta=np.zeros(n),
            t=0.0,
        )

    def reset(
        self,
        load_scale: np.ndarray | float | None = None,
        settle_time: float = 1.5,
        randomize: bool = False,
    ) -> dict:
        """Reset to a settled steady state.

        Parameters
        ----------
        load_scale : per-bus load multiplier (reset randomizes 0.8-1.2).
        settle_time : seconds of nominal-controller operation before handing
            control to the agent. Starting an episode mid-transient makes the
            reward signal unattributable to the policy, so don't skip this.
        randomize : draw load_scale uniformly in [0.8, 1.2] per bus.
        """
        if randomize:
            self.load_scale = self.rng.uniform(0.8, 1.2, size=self.n)
        elif load_scale is None:
            self.load_scale = np.ones(self.n)
        else:
            self.load_scale = np.broadcast_to(
                np.asarray(load_scale, dtype=float), (self.n,)
            ).copy()

        self.ybus = build_ybus(self.p, self.load_scale)
        self.state = self._initial_state()
        self.collapsed = False
        if self.attack is not None:
            self.attack.reset()

        n_settle = int(round(settle_time / self.p.dt))
        for _ in range(n_settle):
            self._integrate_once(residual=None, attack_active=False)
        self.state.t = 0.0
        return self.measure()

    def attach_attack(self, attack) -> None:
        """Attach an AttackInjector (see mgxrl.plant.attacks)."""
        self.attack = attack

    def apply_load_step(self, scale: np.ndarray | float) -> None:
        """Apply a *benign* load change by rebuilding the admittance matrix.

        Benign disturbances are essential, not decorative. Without them:
          - DoS is invisible (freezing a constant signal changes nothing);
          - the detector learns "any transient == attack" and its false-positive
            rate on real switching events is unmeasured;
          - the agent never has to distinguish an attack from a load step, which
            is the operationally hard case.
        Every resilience claim should be paired with a false-trigger rate
        measured on benign-disturbance episodes.
        """
        self.load_scale = np.broadcast_to(
            np.asarray(scale, dtype=float), (self.n,)
        ).copy()
        self.ybus = build_ybus(self.p, self.load_scale)

    # -------------------------------------------------------------- dynamics

    def _network_injection(self, v: np.ndarray, delta: np.ndarray):
        """Algebraic power flow. Returns (P_inj, Q_inj, I_mag) in pu."""
        vc = v * np.exp(1j * delta)
        i = self.ybus @ vc
        s = vc * np.conj(i)
        return s.real, s.imag, np.abs(i)

    def _terminal_voltage(self, state: MicrogridState) -> np.ndarray:
        v = (
            self.p.v_nom
            - self.p.n_droop * (state.q_filt - self.p.q_set)
            + state.zeta
        )
        return np.clip(v, 0.4, 1.6)

    def _omega(self, state: MicrogridState) -> np.ndarray:
        return (
            self.w_nom
            - self.p.m_droop * (state.p_filt - self.p.p_set)
            + state.xi
        )

    def _derivatives(self, state: MicrogridState, residual, attack_active: bool):
        p, n = self.p, self.n
        v = self._terminal_voltage(state)
        w = self._omega(state)
        p_inj, q_inj, _ = self._network_injection(v, state.delta)

        # Angle dynamics in a frame rotating with DG1
        w_ref = w[0]
        d_delta = w - w_ref

        # Power measurement filters
        d_p = (p_inj - state.p_filt) / p.tau_f
        d_q = (q_inj - state.q_filt) / p.tau_f

        # --- Consensus, over the *communicated* (possibly corrupted) values ---
        xi_rx = state.xi.copy()
        zeta_rx = state.zeta.copy()
        if attack_active and self.attack is not None:
            xi_rx, zeta_rx = self.attack.corrupt(state.t, xi_rx, zeta_rx)

        # --- Agent action: per-link TRUST WEIGHTS ----------------------------
        # Two earlier formulations failed, and the reason is worth recording.
        #
        #   residual on dxi/dt   -- the loop settles at k_p*(w-w_nom) =
        #       -c_f*cons_xi, so a persistent residual holds cons_xi away from
        #       zero and can only trade frequency against power sharing. No
        #       constant action restores nominal.
        #
        #   additive offset on the received signal -- expressive enough (an
        #       oracle cancels the attack exactly) but hypersensitive: c_f
        #       amplifies the action by 10x, so a normalised action of 0.005
        #       already cost ~7,500 reward per episode. The useful precision
        #       was ~1/300 of the action range against an exploration std of
        #       0.368. No amount of rescaling fixed that, and temporally
        #       correlated exploration made it worse, not better.
        #
        # Trust weights invert the problem. w_ij in [0,1] scales each incoming
        # consensus link; w = 1 everywhere is exactly the nominal controller.
        # At consensus the neighbours already agree, so down-weighting a link
        # costs NOTHING when nothing is wrong -- exploration is free where it
        # used to be catastrophic. Under attack the response is smooth and
        # monotone in w, and isolating the compromised link (w = 0) restores
        # nominal exactly.
        #
        # This is also the standard resilient-consensus formulation, and it is
        # directly explainable: "the agent reduced trust in the link from DG1"
        # is a sentence an operator can act on, which the additive form never
        # produced.
        # The received values are the ONLY place the attack is directly
        # visible. They used to be consumed here and discarded, which left the
        # policy assigning per-source trust with no per-source evidence: an
        # observability probe put source identification on Additive at 0.287
        # from the observation alone and 0.996 once the consensus innovation
        # was available. Exposed via measure() so both the observation and the
        # probe read the same values instead of re-invoking the injector.
        self.last_xi_rx, self.last_zeta_rx = xi_rx.copy(), zeta_rx.copy()

        A = self.adj
        if residual is not None:
            r = np.asarray(residual, dtype=float).reshape(2, n)
            trust_xi = np.clip(1.0 + r[0], 0.0, 1.0)
            trust_zeta = np.clip(1.0 + r[1], 0.0, 1.0)
        else:
            trust_xi = trust_zeta = np.ones(n)
        A_xi = self.adj * trust_xi[None, :]
        A_zeta = self.adj * trust_zeta[None, :]

        # --- (retired) additive compensator ---------------------------------
        # The residual used to be added to dxi/dt. That was the wrong place.
        # At equilibrium the loop satisfies k_p*(w - w_nom) = -c_f*cons_xi, so
        # the integrator only drives frequency to nominal when cons_xi is zero.
        # A residual injected into the derivative holds cons_xi away from zero
        # permanently: a steady normalised action of just 0.06 produced a
        # 0.0029 Hz offset costing ~10,600 reward per episode, which is exactly
        # the degradation the trained agent showed on nominal episodes.
        #
        # Applying it to the RECEIVED signals instead makes the agent a
        # compensator on the attack surface itself. An additive bias b on the
        # link is cancelled exactly by a residual of -b, restoring cons_xi to
        # zero and with it BOTH frequency regulation and power sharing. The
        # derivative form could only trade one against the other.


        # sum_j w_ij a_ij (xi_i - xi_j_received)
        cons_xi = A_xi.sum(axis=1) * state.xi - A_xi @ xi_rx
        cons_zeta = A_zeta.sum(axis=1) * state.zeta - A_zeta @ zeta_rx

        d_xi = -p.k_p * (w - self.w_nom) - p.c_f * cons_xi
        d_zeta = -p.k_q * (v - p.v_nom) - p.c_v * cons_zeta

        return d_delta, d_p, d_q, d_xi, d_zeta

    def _integrate_once(self, residual, attack_active: bool) -> None:
        """One forward-Euler step of size dt. Euler is adequate at 1 ms given
        tau_f = 16 ms; switch to RK4 here if you raise dt."""
        s, p = self.state, self.p
        d_delta, d_p, d_q, d_xi, d_zeta = self._derivatives(s, residual, attack_active)

        s.delta = s.delta + p.dt * d_delta
        s.p_filt = s.p_filt + p.dt * d_p
        s.q_filt = s.q_filt + p.dt * d_q
        s.xi = np.clip(s.xi + p.dt * d_xi, -p.xi_limit, p.xi_limit)
        s.zeta = np.clip(s.zeta + p.dt * d_zeta, -p.zeta_limit, p.zeta_limit)
        s.t += p.dt

        # Keep angles bounded
        s.delta = np.mod(s.delta + np.pi, TWO_PI) - np.pi

    def step(self, residual=None, attack_active: bool = True) -> dict:
        """Advance one control interval (control_dt). Returns a measurement dict."""
        for _ in range(self.p.substeps):
            self._integrate_once(residual, attack_active)
        self._check_collapse()
        return self.measure()

    def _check_collapse(self) -> None:
        v = self._terminal_voltage(self.state)
        f = self._omega(self.state) / TWO_PI
        bad = (
            not np.all(np.isfinite(v))
            or not np.all(np.isfinite(f))
            or np.any(v < 0.5)
            or np.any(v > 1.5)
            or np.any(np.abs(f - self.p.f_nom) > self.p.f_tol)
        )
        self.collapsed = bool(bad)

    # ---------------------------------------------------------- measurement

    def measure(self, noisy: bool = True) -> dict:
        """Sensor view of the plant, matching the Dataport signal set."""
        s, p = self.state, self.p
        v = self._terminal_voltage(s)
        w = self._omega(s)
        f = w / TWO_PI
        p_inj, q_inj, i_mag = self._network_injection(v, s.delta)

        if noisy:
            v = v + self.rng.normal(0, p.noise_v, self.n)
            f = f + self.rng.normal(0, p.noise_f, self.n)
            p_inj = p_inj + self.rng.normal(0, p.noise_p, self.n)
            q_inj = q_inj + self.rng.normal(0, p.noise_q, self.n)
            if p.quantization_v > 0:
                v = np.round(v / p.quantization_v) * p.quantization_v

        return {
            "t": s.t,
            "V": v, "f": f, "P": p_inj, "Q": q_inj, "I": i_mag,
            "xi": s.xi.copy(), "zeta": s.zeta.copy(),
            "xi_rx": np.asarray(getattr(self, "last_xi_rx", s.xi)).copy(),
            "zeta_rx": np.asarray(getattr(self, "last_zeta_rx", s.zeta)).copy(),
            "delta": s.delta.copy(),
            "collapsed": self.collapsed,
        }

    def to_dataset_row(self, meas: dict) -> np.ndarray:
        """Project a measurement onto the 38-column Dataport schema:

            time, V1, V2, V3, I1, I2, I3,
            P_DG1, Q_DG1, f_DG1, ..., P_DG10, Q_DG10, f_DG10, label

        (label excluded here.) Use this to feed the dataset-trained detector
        with simulator output -- it is the interface that lets the detector
        transfer from the Dataport CSVs into the RL loop.
        """
        row = [meas["t"], *meas["V"][:3], *meas["I"][:3]]
        for i in range(self.n):
            row += [meas["P"][i], meas["Q"][i], meas["f"][i]]
        return np.asarray(row, dtype=np.float64)

    # ------------------------------------------------------------- diagnostics

    def metrics(self, meas: dict) -> dict:
        """Control-performance metrics used by the reward and by evaluation.

        `xi_spread` / `zeta_spread` are the consensus-disagreement metrics. They
        matter more than they look: a DoS attack barely moves voltage or
        frequency, but it drives xi_spread up by ~60x because the frozen link
        stops the correction states from agreeing. Proportional power sharing is
        *enforced* by that agreement, so disagreement is the physically correct
        thing to penalise -- and it is the only term that gives the agent a
        usable gradient on DoS and Coordinated Stealth.
        """
        p = self.p
        v, f = meas["V"], meas["f"]
        v_viol = np.sum(
            np.maximum(0.0, p.v_min - v) ** 2 + np.maximum(0.0, v - p.v_max) ** 2
        )
        f_dev = float(np.sqrt(np.mean((f - p.f_nom) ** 2)))
        v_dev = float(np.sqrt(np.mean((v - p.v_nom) ** 2)))
        # Power-sharing error: dispersion of normalised injections
        p_share = float(np.std(meas["P"]) / (np.mean(np.abs(meas["P"])) + 1e-9))
        q_share = float(np.std(meas["Q"]) / (np.mean(np.abs(meas["Q"])) + 1e-9))
        return {
            "v_violation": float(v_viol),
            "v_violation_any": bool(np.any((v < p.v_min) | (v > p.v_max))),
            "f_rmse": f_dev,
            "v_rmse": v_dev,
            "p_sharing_error": p_share,
            "q_sharing_error": q_share,
            "xi_spread": float(np.std(meas["xi"])),
            "zeta_spread": float(np.std(meas["zeta"])),
        }
