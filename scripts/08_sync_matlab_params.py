#!/usr/bin/env python3
"""Regenerate matlab/mg_params.m from configs/microgrid.yaml.

Prevents the two parameter sets from drifting. If they drift, the Tier-A
fidelity check measures the parameter mismatch rather than the modeling
assumption, and you will chase a discrepancy that is not there.

Run this after every edit to configs/microgrid.yaml.
"""
import argparse, os, sys

import yaml

TEMPLATE = """function p = mg_params()
%MG_PARAMS  AUTO-GENERATED from configs/microgrid.yaml by
%           scripts/08_sync_matlab_params.py -- DO NOT EDIT BY HAND.
%           Re-run that script after changing the YAML.

p.n        = {n_dg};
p.f_nom    = {f_nom};
p.w_nom    = 2*pi*p.f_nom;
p.v_nom    = {v_nom};

p.m_droop  = {m_droop};
p.n_droop  = {n_droop};
p.tau_f    = {tau_f};

p.k_p      = {k_p};   p.c_f = {c_f};
p.k_q      = {k_q};   p.c_v = {c_v};
p.xi_lim   = 0.10 * p.w_nom;
p.zeta_lim = 0.10;

p.r_line   = {r_line};  p.x_line = {x_line};
p.load_p   = {load_p};  p.load_q = {load_q};
p.p_set    = {p_set};   p.q_set  = {q_set};

% Inner loops: the fidelity increment over the Python training plant, which
% treats terminal voltage as algebraic. Set false to isolate discretisation
% error from the inner-loop assumption when diagnosing a Tier-A failure.
p.use_inner_loops = true;
p.tau_v    = 1/(2*pi*100);
p.tau_i    = 1/(2*pi*1000);

p.adj = zeros(p.n);
for i = 1:p.n
    j = mod(i, p.n) + 1;
    p.adj(i,j) = 1;  p.adj(j,i) = 1;
end
p.deg = sum(p.adj, 2);

p.load_scale = ones(p.n,1);
p.ybus = mg_ybus(p);
end
"""


def main():
    root = os.path.join(os.path.dirname(__file__), "..")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(root, "configs/microgrid.yaml"))
    ap.add_argument("--out", default=os.path.join(root, "matlab/mg_params.m"))
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))["plant"]
    open(a.out, "w").write(TEMPLATE.format(**cfg))
    print(f"Wrote {a.out} from {a.config}")
    print("Reward reference scales are NOT synced -- they are calibrated, not "
          "configured. Re-run scripts/00_calibrate_reward.py after plant changes.")


if __name__ == "__main__":
    main()
