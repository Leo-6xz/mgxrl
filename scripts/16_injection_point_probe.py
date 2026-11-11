"""
16_injection_point_probe.py

Usage (PowerShell, one line):

  python scripts\\16_injection_point_probe.py --root C:\\research\\mgxrl\\Dataset
"""

from __future__ import annotations

import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
from mgxrl.detection.dataio import LoadConfig, load_scenario
from mgxrl.plant.microgrid import Microgrid, MicrogridParams
from mgxrl.plant.attacks import ATTACK_REGISTRY

N_DG = 10
GATED_OUT = ("DoS",)          # fails the shared-startup gate; see script 14
CLASSES = ("Additive", "Ramp", "SlowRamp", "Sinusoid", "Stealth")


# --------------------------------------------------------------------------
# node-local variant, confined to this file
# --------------------------------------------------------------------------

class NodeLocalMicrogrid(Microgrid):
    """Same plant, attack applied to the DG's own secondary state.

    The corrupted value then feeds three places instead of one: the attacked
    DG's own omega through _omega, its own consensus row, and -- because
    xi_rx is a copy of state.xi -- what its neighbours receive. That last part
    matters: the compromised node still broadcasts honestly, it is simply
    honest about a wrong number. attack_active is then passed as False to the
    parent so the injector is not applied a second time on the wire.
    """

    def _derivatives(self, state, residual, attack_active: bool):
        if attack_active and self.attack is not None:
            s = state.copy()
            s.xi, s.zeta = self.attack.corrupt(state.t, s.xi, s.zeta)
            return super()._derivatives(s, residual, False)
        return super()._derivatives(state, residual, attack_active)


def plant_run(cls, scenario, t_start, t_end, attacked, dt, seed=0):
    mg = cls(MicrogridParams(seed=seed, control_dt=dt))
    if attacked:
        targets = (0,) if scenario != "Stealth" else (0, 1, 2)
        mg.attach_attack(ATTACK_REGISTRY[scenario](targets=targets,
                                                   t_start=t_start))
    mg.reset(settle_time=3.0)
    n = int(round(t_end / dt)) + 1
    P = np.empty((n, N_DG)); F = np.empty((n, N_DG)); T = np.empty(n)
    for k in range(n):
        m = mg.measure(noisy=False)
        T[k] = m["t"]; P[k] = m["P"][:N_DG]; F[k] = m["f"][:N_DG]
        mg.step()
    return T, P, F


def plant_signature(cls, scenario, onset, window, dt, long_s):
    ta, Pa, Fa = plant_run(cls, scenario, onset, onset + long_s, True, dt)
    tb, Pb, Fb = plant_run(cls, scenario, onset, onset + long_s, False, dt)
    dP, dF = Pa - Pb, Fa - Fb
    win = (ta >= onset) & (ta <= onset + window)
    return dP[win], dF[win], dF[-max(3, int(0.02 / dt)):]


# --------------------------------------------------------------------------
# release side
# --------------------------------------------------------------------------

def release_differential(root, scenario, decimate, onset, window, dt):
    def grab(scen):
        cfg = LoadConfig(root=root, decimate=decimate, max_files_per_scenario=1)
        df = load_scenario(root, scen, cfg)
        cmap = {c.strip().lower(): c for c in df.columns}
        t = pd.to_numeric(df[cmap.get("time", df.columns[0])],
                          errors="coerce").to_numpy(dtype=float)
        P = np.column_stack([pd.to_numeric(df[cmap[f"p_dg{i}"]], errors="coerce")
                             for i in range(1, N_DG + 1)]).astype(float)
        F = np.column_stack([pd.to_numeric(df[cmap[f"f_dg{i}"]], errors="coerce")
                             for i in range(1, N_DG + 1)]).astype(float)
        return t, P, F

    ta, Pa, Fa = grab(scenario)
    tn, Pn, Fn = grab("Normal")
    grid = np.arange(onset, onset + window, dt)

    def rg(t, M):
        return np.column_stack([np.interp(grid, t, M[:, j])
                                for j in range(M.shape[1])])
    return rg(ta, Pa) - rg(tn, Pn), rg(ta, Fa) - rg(tn, Fn)


# --------------------------------------------------------------------------
# signature scoring
# --------------------------------------------------------------------------

def end_vec(M, frac=0.2):
    k = max(3, int(M.shape[0] * frac))
    return np.nanmean(M[-k:], axis=0)


def score(dP, dF, target=0):
    """Unit-free signature of a per-DG response vector."""
    p = end_vec(dP)
    mag = np.abs(p)
    tot = float(np.nansum(mag))
    others = np.delete(p, target)
    mu = float(np.nanmean(np.abs(others)))
    f = end_vec(dF)
    return {
        "dP1_share": float(mag[target] / tot) if tot > 0 else np.nan,
        "other9_cv": float(np.nanstd(np.abs(others)) / mu) if mu > 0 else np.nan,
        "sign_dP1": int(np.sign(p[target])),
        "others_same_sign": bool(np.all(np.sign(others) == np.sign(others[0]))),
        "f_spread_Hz": float(np.nanmax(f) - np.nanmin(f)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--decimate", type=int, default=500)
    ap.add_argument("--dt", type=float, default=1e-3)
    ap.add_argument("--onset", type=float, default=0.5005)
    ap.add_argument("--window", type=float, default=0.3)
    ap.add_argument("--long", type=float, default=3.0,
                    help="seconds after onset for the synchronism check")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    rows = []
    for scen in CLASSES:
        if scen in GATED_OUT:
            continue
        rP, rF = release_differential(a.root, scen, a.decimate,
                                      a.onset, a.window, a.dt)
        rows.append(dict(attack=scen, source="release", **score(rP, rF),
                         f_spread_long_Hz=np.nan))

        for label, cls in (("plant-link", Microgrid),
                           ("plant-node", NodeLocalMicrogrid)):
            dP, dF, dF_end = plant_signature(cls, scen, a.onset, a.window,
                                             a.dt, a.long)
            s = score(dP, dF)
            fe = np.nanmean(dF_end, axis=0)
            s["f_spread_long_Hz"] = float(np.nanmax(fe) - np.nanmin(fe))
            rows.append(dict(attack=scen, source=label, **s))

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", lambda v: f"{v:.5g}")
    print(df.to_string(index=False))
    df.to_csv(os.path.join(a.out, "injection_point_probe.csv"), index=False)

    print("\nQ1 -- read dP1_share and other9_cv together.")
    print("  The release's Additive signature is a large dP1_share with a")
    print("  small other9_cv: one node sheds, the rest absorb it evenly. If")
    print("  plant-node reproduces that and plant-link does not, the release")
    print("  implements a node-local attack and attacks.py models a different")
    print("  threat. Neither is wrong on its own terms; they are not the same")
    print("  experiment has to say which one it claims.")
    print("  A caveat that applies to BOTH plant rows: this probe changes only")
    print("  the injection point. It does not tune amplitudes, and the")
    print("  release's injection magnitudes were never published, so a match")
    print("  on shape here is not a match on severity.")
    print("\nQ2 -- f_spread_long_Hz is the synchronism check.")
    print(f"  Ten interconnected buses cannot hold different frequencies in")
    print(f"  steady state. If this has not collapsed toward zero by")
    print(f"  {a.long:.1f} s after onset while f_spread_Hz at 0.3 s is ~0.02,")
    print("  the plant is not reaching synchronism and that is a defect Tier A")
    print("  cannot detect, since MATLAB integrates the same equations.")
    print(f"\nWrote {a.out}/injection_point_probe.csv")


if __name__ == "__main__":
    main()
