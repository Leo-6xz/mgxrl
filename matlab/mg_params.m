function p = mg_params()
%MG_PARAMS  AUTO-GENERATED from configs/microgrid.yaml by
%           scripts/08_sync_matlab_params.py

p.n        = 10;
p.f_nom    = 50.0;
p.w_nom    = 2*pi*p.f_nom;
p.v_nom    = 1.0;

p.m_droop  = 1.5708;
p.n_droop  = 0.05;
p.tau_f    = 0.016;

p.k_p      = 5.0;   p.c_f = 10.0;
p.k_q      = 5.0;   p.c_v = 10.0;
p.xi_lim   = 0.10 * p.w_nom;
p.zeta_lim = 0.10;

p.r_line   = 0.05;  p.x_line = 0.15;
p.load_p   = 0.65;  p.load_q = 0.28;
p.p_set    = 0.5;   p.q_set  = 0.2;

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
