function [dx, y] = mg_dynamics(t, x, u, p, atk)
%MG_DYNAMICS  Reference microgrid model with inner voltage/current loops.
%
% State layout (n = 10):
%   x( 1:10) delta   bus angles, rad
%   x(11:20) Pf      filtered active power, pu
%   x(21:30) Qf      filtered reactive power, pu
%   x(31:40) xi      secondary frequency correction, rad/s
%   x(41:50) zeta    secondary voltage correction, pu
%   x(51:60) Vt      terminal voltage state (inner loop), pu   [if enabled]
%
% u : 20x1 TRUST OFFSETS from the RL policy, [a_xi(1:10); a_zeta(1:10)].
%
%     trust_j = min(max(1 + a_j, 0), 1), applied as a weight on the signal
%     TRANSMITTED BY DG j, so column j of the adjacency matrix is scaled. a = 0
%     is full trust and reduces exactly to the published consensus controller;
%     a = -1 isolates that source.
%
%     This file previously summed u into dxi/dzeta. That is the FIRST action
%     space, which was abandoned: at equilibrium the loop satisfies
%     k_p*(w - w_nom) = -c_f*cons_xi, so a residual in the derivative holds
%     frequency away from nominal permanently and no constant action can
%     restore it. Deploying the exported policy through that path applies a
%     trust vector as a derivative offset -- the numbers arrive, nothing
%     errors, and the controller is unrelated to the one that was trained.
%
% Codegen-safe: fixed sizes, no dynamic allocation, so this file can be pasted
% straight into a MATLAB Function block (see build_reference_model.m) as well as
% called from ode23tb.

n = p.n;
delta = x(1:n);  Pf = x(n+1:2*n);  Qf = x(2*n+1:3*n);
xi    = x(3*n+1:4*n);  zeta = x(4*n+1:5*n);

% --- Droop set-points (eqs. 1-2) ---
w      = p.w_nom  - p.m_droop*(Pf - p.p_set) + xi;
V_cmd  = p.v_nom  - p.n_droop*(Qf - p.q_set) + zeta;
V_cmd  = min(max(V_cmd, 0.4), 1.6);

% --- Inner loops: terminal voltage lags its command ---
if p.use_inner_loops
    Vt  = x(5*n+1:6*n);
    % Cascaded voltage + current loop, collapsed to a single dominant lag.
    % tau_i << tau_v, so the current loop contributes negligible phase at the
    % secondary timescale; retaining tau_v is what actually tests the Python
    % model's algebraic-voltage assumption.
    dVt = (V_cmd - Vt) / p.tau_v;
    V   = Vt;
else
    dVt = zeros(n,1);
    V   = V_cmd;
end

% --- Algebraic network solution ---
Vc  = V .* exp(1j*delta);
I   = p.ybus * Vc;
S   = Vc .* conj(I);
Pinj = real(S);  Qinj = imag(S);

% --- Angle dynamics in the frame of DG1 ---
ddelta = w - w(1);

% --- Power measurement filters ---
dPf = (Pinj - Pf) / p.tau_f;
dQf = (Qinj - Qf) / p.tau_f;

% --- Consensus over the COMMUNICATED (possibly corrupted) values ---
[xi_rx, zeta_rx] = mg_attack(t, xi, zeta, atk);

% Per-source trust weights. With u = 0 both weight vectors are all ones,
% sum(A,2) == p.deg, and this reduces exactly to the published controller
%     cons = p.deg .* x - p.adj * x_rx
% which is what the nominal-case stability argument rests on.
tr_xi   = min(max(1 + u(1:n),     0), 1);
tr_zeta = min(max(1 + u(n+1:2*n), 0), 1);
A_xi    = p.adj .* repmat(tr_xi.',   n, 1);
A_zeta  = p.adj .* repmat(tr_zeta.', n, 1);

cons_xi   = sum(A_xi,   2) .* xi   - A_xi   * xi_rx;
cons_zeta = sum(A_zeta, 2) .* zeta - A_zeta * zeta_rx;

dxi   = -p.k_p*(w - p.w_nom) - p.c_f*cons_xi;
dzeta = -p.k_q*(V - p.v_nom) - p.c_v*cons_zeta;

% Anti-windup on the saturation limits (the Python model clips after the Euler
% step; here the derivative is zeroed at the bound, which is the correct
% continuous-time equivalent and avoids a spurious discrepancy at saturation).
dxi(xi   >=  p.xi_lim   & dxi   > 0) = 0;
dxi(xi   <= -p.xi_lim   & dxi   < 0) = 0;
dzeta(zeta >=  p.zeta_lim & dzeta > 0) = 0;
dzeta(zeta <= -p.zeta_lim & dzeta < 0) = 0;

if p.use_inner_loops
    dx = [ddelta; dPf; dQf; dxi; dzeta; dVt];
else
    dx = [ddelta; dPf; dQf; dxi; dzeta];
end

% --- Measurement vector, matching the Python plant's ordering ---
% The RECEIVED consensus values are appended. They are the only place the
% attack is directly visible, and the observation's per-source innovation
% needs them; Microgrid.measure() in Python exposes them for the same reason.
% Anything indexing y beyond 7n must therefore expect 9n, not 7n.
y = [V; w/(2*pi); Pinj; Qinj; abs(I); xi; zeta; xi_rx; zeta_rx];
end
