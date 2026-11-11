"""
The six secondary-control cyberattacks, implemented as corruptions of the
*communicated* consensus signals (eqs. 9-17 of Ogiesoba-Eguakun & Rath).

Design note: the attack corrupts what neighbours *receive*, not the attacked
DG's local state. This matters. Corrupting local state is a sensor attack;
corrupting the link is a communication attack, and only the latter reproduces
the propagation-through-the-consensus-graph behaviour that Section III-C
describes and that Fig. 7 shows.

All injectors expose the same interface so the environment can sample one
uniformly at reset:

    inj.reset()
    xi_rx, zeta_rx = inj.corrupt(t, xi, zeta)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

# Class indices match the Dataport multiclass labelling.
ATTACK_CLASSES = {
    0: "Normal",
    1: "Additive",
    2: "Ramp",
    3: "SlowRamp",
    4: "Sinusoid",
    5: "Stealth",
    6: "DoS",
}
CLASS_TO_INDEX = {v: k for k, v in ATTACK_CLASSES.items()}


@dataclass
class AttackInjector:
    """Base class. `targets` are the DG indices whose outgoing links are spoofed."""

    targets: Sequence[int] = (0,)          # DG1, per Fig. 3
    t_start: float = 0.0                   # attack onset, s (relative to episode)
    class_index: int = 0
    name: str = "Normal"
    rng: np.random.Generator = field(default_factory=np.random.default_rng)

    def reset(self) -> None:
        self._t0 = None

    def active(self, t: float) -> bool:
        return t >= self.t_start

    def elapsed(self, t: float) -> float:
        return max(0.0, t - self.t_start)

    def corrupt(self, t, xi, zeta):
        return xi, zeta

    def label(self, t: float) -> tuple[int, int]:
        """(binary, multiclass) ground-truth labels at time t.

        Note the multiclass label is 0 (Normal) *before* onset, mirroring the
        Dataport convention where attack folders contain both 0s and 1s.
        """
        if self.class_index == 0 or not self.active(t):
            return 0, 0
        return 1, self.class_index


@dataclass
class NoAttack(AttackInjector):
    class_index: int = 0
    name: str = "Normal"


@dataclass
class AdditiveBias(AttackInjector):
    """eq. (9)-(10): constant offset on both consensus signals."""

    bias_xi: float = 0.3        # rad/s
    bias_zeta: float = 0.01     # pu
    class_index: int = 1
    name: str = "Additive"

    def corrupt(self, t, xi, zeta):
        if not self.active(t):
            return xi, zeta
        xi, zeta = xi.copy(), zeta.copy()
        for i in self.targets:
            xi[i] += self.bias_xi
            zeta[i] += self.bias_zeta
        return xi, zeta


@dataclass
class Ramp(AttackInjector):
    """eq. (11): growing bias on the frequency-correction signal."""

    slope: float = 0.5          # rad/s per s
    class_index: int = 2
    name: str = "Ramp"

    def corrupt(self, t, xi, zeta):
        if not self.active(t):
            return xi, zeta
        xi = xi.copy()
        drift = self.slope * self.elapsed(t)
        for i in self.targets:
            xi[i] += drift
        return xi, zeta


@dataclass
class SlowRamp(AttackInjector):
    """eq. (12): very small slope on the voltage-restoration signal, r_s << r."""

    slope: float = 0.004        # pu per s
    class_index: int = 3
    name: str = "SlowRamp"

    def corrupt(self, t, xi, zeta):
        if not self.active(t):
            return xi, zeta
        zeta = zeta.copy()
        drift = self.slope * self.elapsed(t)
        for i in self.targets:
            zeta[i] += drift
        return xi, zeta


@dataclass
class Sinusoid(AttackInjector):
    """eq. (13)-(14): coherent oscillation injected into both signals."""

    amp_xi: float = 0.4         # rad/s
    amp_zeta: float = 0.012     # pu
    freq: float = 2.0           # Hz
    class_index: int = 4
    name: str = "Sinusoid"

    def corrupt(self, t, xi, zeta):
        if not self.active(t):
            return xi, zeta
        xi, zeta = xi.copy(), zeta.copy()
        s = np.sin(2 * np.pi * self.freq * self.elapsed(t))
        for i in self.targets:
            xi[i] += self.amp_xi * s
            zeta[i] += self.amp_zeta * s
        return xi, zeta


@dataclass
class CoordinatedStealth(AttackInjector):
    """eq. (15)-(16): low-amplitude correlated deviations across several DGs.

    This is the class your controller should be judged on. It is engineered to
    stay inside normal operating envelopes, so a detector that only sees
    threshold violations will miss it, and a controller that only reacts to
    violations will never act.
    """

    amp: float = 0.06           # rad/s -- deliberately inside the noise envelope
    alpha: float = 0.02         # zeta/xi correlation coefficient
    freq: float = 0.7           # Hz
    class_index: int = 5
    name: str = "Stealth"

    def corrupt(self, t, xi, zeta):
        if not self.active(t):
            return xi, zeta
        xi, zeta = xi.copy(), zeta.copy()
        te = self.elapsed(t)
        # Low-amplitude, smoothly varying, correlated across targets
        base = self.amp * (
            0.6 * np.sin(2 * np.pi * self.freq * te)
            + 0.4 * np.sin(2 * np.pi * 0.31 * self.freq * te + 1.1)
        )
        for k, i in enumerate(self.targets):
            phase = 1.0 - 0.1 * k       # slight decorrelation across DGs
            xi[i] += base * phase
            zeta[i] += self.alpha * base * phase
        return xi, zeta


@dataclass
class DenialOfService(AttackInjector):
    """eq. (17): freeze the frequency-correction signal at its last value."""

    class_index: int = 6
    name: str = "DoS"
    _held: np.ndarray | None = None

    def reset(self) -> None:
        super().reset()
        self._held = None

    def corrupt(self, t, xi, zeta):
        if not self.active(t):
            self._held = None
            return xi, zeta
        if self._held is None:
            self._held = xi.copy()
        xi = xi.copy()
        for i in self.targets:
            xi[i] = self._held[i]
        return xi, zeta


ATTACK_REGISTRY = {
    "Normal": NoAttack,
    "Additive": AdditiveBias,
    "Ramp": Ramp,
    "SlowRamp": SlowRamp,
    "Sinusoid": Sinusoid,
    "Stealth": CoordinatedStealth,
    "DoS": DenialOfService,
}


def sample_attack(
    rng: np.random.Generator,
    n_dg: int = 10,
    t_window: tuple[float, float] = (0.5, 2.0),
    p_normal: float = 1 / 7,
    randomize_severity: bool = True,
    subset: tuple[str, ...] | None = None,
) -> AttackInjector:
    """Draw a random attack for episode reset.

    Randomising onset time and severity is not optional. Train on a fixed onset
    and the agent learns the clock instead of the signal -- the same failure
    mode that inflates the published detector's accuracy.
    """
    names = [n for n in ATTACK_REGISTRY if subset is None or n in subset]
    if "Normal" not in names:
        names = ["Normal"] + names
    if len(names) == 1:                      # Normal-only curriculum stage
        weights = np.array([1.0])
    else:
        weights = np.array(
            [p_normal] + [(1 - p_normal) / (len(names) - 1)] * (len(names) - 1)
        )
    name = rng.choice(names, p=weights / weights.sum())
    cls = ATTACK_REGISTRY[name]

    n_targets = 1 if name != "Stealth" else int(rng.integers(2, 5))
    targets = tuple(rng.choice(n_dg, size=n_targets, replace=False).tolist())
    t_start = float(rng.uniform(*t_window)) if name != "Normal" else 0.0

    inj = cls(targets=targets, t_start=t_start, rng=rng)

    if randomize_severity and name != "Normal":
        scale = float(rng.uniform(0.4, 1.8))
        for attr in ("bias_xi", "bias_zeta", "slope", "amp", "amp_xi", "amp_zeta"):
            if hasattr(inj, attr):
                setattr(inj, attr, getattr(inj, attr) * scale)
        if hasattr(inj, "freq"):
            inj.freq = float(inj.freq * rng.uniform(0.5, 2.0))

    inj.reset()
    return inj
