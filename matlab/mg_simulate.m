function out = mg_simulate(atk, p, T, dt_log, u_fcn)
%MG_SIMULATE  Run the reference model with a stiff variable-step solver.
%
% Using ode23tb rather than a fixed-step Euler integrator is deliberate. The
% Python training plant uses forward Euler at 1 ms

if nargin < 3 || isempty(T),      T = 12.0;   end
if nargin < 4 || isempty(dt_log), dt_log = 0.010; end
if nargin < 5, u_fcn = @(t,x) zeros(2*p.n,1); end

n  = p.n;
nx = 5*n + p.use_inner_loops*n;

% Settle under the nominal controller before the attack window opens. Starting
% mid-transient makes every downstream comparison meaningless.
x0 = zeros(nx,1);
x0(n+1:2*n)   = p.load_p;
x0(2*n+1:3*n) = p.load_q;
if p.use_inner_loops, x0(5*n+1:6*n) = p.v_nom; end

quiet = atk;  quiet.type = 'Normal';
opts  = odeset('RelTol',1e-8,'AbsTol',1e-10,'MaxStep',0.01);
[~, Xs] = ode23tb(@(t,x) mg_dynamics(t,x,u_fcn(t,x),p,quiet), [0 3.0], x0, opts);
x0 = Xs(end,:).';

% DoS freezes the signal at the value it held AT ONSET. Capturing it from the
% end of the settling run instead (which is what this did) freezes the
% pre-attack value, and the two differ by whatever the grid did between
% settling and t_start -- benign load steps, in particular. The Python
% injector captures at the first active step, so the previous version made
% Tier A report a DoS discrepancy that was a protocol difference, not a
% modeling one.
if strcmp(atk.type,'DoS')
    if atk.t_start > 0
        [~, Xa] = ode23tb(@(t,x) mg_dynamics(t,x,u_fcn(t,x),p,quiet), ...
                          [0 atk.t_start], x0, opts);
        atk.held = Xa(end, 3*n+(1:n)).';
    else
        atk.held = x0(3*n+1:4*n);
    end
end

tspan = 0:dt_log:T;
[tt, XX] = ode23tb(@(t,x) mg_dynamics(t,x,u_fcn(t,x),p,atk), tspan, x0, opts);

ny = 9*n;   % mg_dynamics now also returns xi_rx / zeta_rx
Y  = zeros(numel(tt), ny);
for k = 1:numel(tt)
    [~, yk] = mg_dynamics(tt(k), XX(k,:).', u_fcn(tt(k),XX(k,:).'), p, atk);
    Y(k,:)  = yk.';
end

out.t     = tt;
out.V     = Y(:,        1:n);
out.f     = Y(:,   n + (1:n));
out.P     = Y(:, 2*n + (1:n));
out.Q     = Y(:, 3*n + (1:n));
out.I     = Y(:, 4*n + (1:n));
out.xi    = Y(:, 5*n + (1:n));
out.zeta  = Y(:, 6*n + (1:n));
out.xi_rx   = Y(:, 7*n + (1:n));
out.zeta_rx = Y(:, 8*n + (1:n));
out.X     = XX;
end
