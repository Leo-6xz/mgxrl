function R = validate_threshold_rule(varargin)
%VALIDATE_THRESHOLD_RULE  Closed-loop check of the innovation-threshold rule
%on the MATLAB reference model (stiff variable-step solver, inner voltage
%loop), against the same scenarios as the Python plant.
%
% Run from the matlab folder, after scripts/25_matlab_validation_scenarios.py:
%
%   R = validate_threshold_rule();                    % all scenarios
%   R = validate_threshold_rule('MaxRuns', 4);        % quick smoke test
%   R = validate_threshold_rule('InnerLoops', false); % isolate the solver
%
% The rule is evaluated every control period (10 ms) on the measurement at the
% end of the period and held over the next one, exactly as in Python:
%   source j is distrusted on a channel while |innovation_j| > threshold,
%   then the same rate limit (0.25 per step) and connectivity floor
%   (at least 3 sources with trust >= 0.5 per channel) are applied.
% Thresholds come from thresholds.json, calibrated on the PYTHON plant and not
% re-tuned here, so this also tests whether the rule transfers.
%
% Writes results/matlab_validation/matlab_metrics.csv, with the same metric
% definitions as the Python script. Compare with
%   python scripts\26_compare_matlab_validation.py

ip = inputParser;
ip.addParameter('Dir', fullfile('..', 'results', 'matlab_validation'));
ip.addParameter('InnerLoops', true);
ip.addParameter('MaxRuns', inf);
ip.addParameter('RelTol', 1e-8);
ip.addParameter('AbsTol', 1e-10);
ip.parse(varargin{:});
o = ip.Results;

p = mg_params();
p.use_inner_loops = logical(o.InnerLoops);
th = jsondecode(fileread(fullfile(o.Dir, 'thresholds.json')));

scenFile = fullfile(o.Dir, 'scenarios.csv');
io = detectImportOptions(scenFile);
io = setvartype(io, 'targets', 'string');
io = setvartype(io, 'attack', 'string');
S = readtable(scenFile, io);

nRun = min(height(S), o.MaxRuns);
rows = cell(2*nRun, 9);
r = 0;
tic;
for k = 1:nRun
    atk = scenario_attack(S(k, :));
    arms = {'nominal', 'threshold'};
    for a = 1:2
        res = closed_loop(p, atk, th, a == 2, o);
        r = r + 1;
        rows(r, :) = {char(S.attack(k)), S.seed(k), arms{a}, res.collapsed, ...
                      res.f_rms_mHz, res.f_max_mHz, res.p_share, res.latency_ms, ...
                      logical(o.InnerLoops)};
    end
    fprintf('%3d/%d  %-9s seed %2d   nominal %8.3f mHz   threshold %8.3f mHz   (%.0f s)\n', ...
            k, nRun, char(S.attack(k)), S.seed(k), rows{r-1, 5}, rows{r, 5}, toc);
end

R = cell2table(rows(1:r, :), 'VariableNames', ...
    {'attack', 'seed', 'arm', 'collapsed', 'f_rms_mHz', 'f_max_mHz', ...
     'p_share', 'latency_ms', 'inner_loops'});
if o.InnerLoops
    outFile = fullfile(o.Dir, 'matlab_metrics.csv');
else
    outFile = fullfile(o.Dir, 'matlab_metrics_no_inner_loops.csv');
end
writetable(R, outFile);
fprintf('written: %s\n', outFile);
end

% -------------------------------------------------------------------------
function atk = scenario_attack(row)
% Build the mg_attack struct from one scenario row (Python's sampled values).
atk.type    = char(row.attack);
atk.t_start = row.t_start;
atk.targets = str2double(split(row.targets, ';')).';
f = @(v) fillnan(v);
atk.bias_xi   = f(row.bias_xi);   atk.bias_zeta = f(row.bias_zeta);
atk.slope     = f(row.slope);
atk.amp_xi    = f(row.amp_xi);    atk.amp_zeta  = f(row.amp_zeta);
atk.freq      = f(row.freq);
atk.amp       = f(row.amp);       atk.alpha     = f(row.alpha);
atk.held      = zeros(10, 1);
atk.held_is_placeholder = true;
end

function v = fillnan(v)
if isempty(v) || isnan(v), v = 0; end
end

% -------------------------------------------------------------------------
function res = closed_loop(p, atk, th, useRule, o)
n  = p.n;
nx = 5*n + p.use_inner_loops*n;
dt = 0.01;
N  = 1200;                         % 12 s, as in the Python episodes
opts = odeset('RelTol', o.RelTol, 'AbsTol', o.AbsTol, 'MaxStep', 0.01);

% Settle under the nominal controller (as mg_simulate does).
x = zeros(nx, 1);
x(n+1:2*n)   = p.load_p;
x(2*n+1:3*n) = p.load_q;
if p.use_inner_loops, x(5*n+1:6*n) = p.v_nom; end
quiet = atk;  quiet.type = 'Normal';
[~, Xs] = ode23tb(@(t, xx) mg_dynamics(t, xx, zeros(2*n, 1), p, quiet), [0 3.0], x, opts);
x = Xs(end, :).';

u = zeros(2*n, 1);
T = zeros(N, 1);  F = zeros(N, n);  P = zeros(N, n);  TR = zeros(N, n);
t = 0;  collapsed = false;  heldSet = false;
for k = 1:N
    t1 = t + dt;
    % Integrate one control period with the action held. Split at the onset so
    % the solver sees the discontinuity, and so DoS captures xi AT onset, as
    % the Python injector does.
    if atk.t_start > t && atk.t_start <= t1
        x = advance(p, atk, u, x, t, atk.t_start, opts);
        if strcmp(atk.type, 'DoS') && ~heldSet
            atk.held = x(3*n+1:4*n);
            heldSet = true;
        end
        x = advance(p, atk, u, x, atk.t_start, t1, opts);
    else
        x = advance(p, atk, u, x, t, t1, opts);
    end
    t = t1;

    [~, y] = mg_dynamics(t, x, u, p, atk);
    V  = y(1:n);            f  = y(n+1:2*n);       Pk = y(2*n+1:3*n);
    xi = y(5*n+1:6*n);      ze = y(6*n+1:7*n);
    xr = y(7*n+1:8*n);      zr = y(8*n+1:9*n);

    T(k) = t;  F(k, :) = f.';  P(k, :) = Pk.';
    TR(k, :) = min(max(1 + u(1:n), 0), 1).';

    if any(V < 0.5) || any(V > 1.5) || any(abs(f - 50) > 0.5)
        collapsed = true;
        T = T(1:k);  F = F(1:k, :);  P = P(1:k, :);  TR = TR(1:k, :);
        break
    end

    if useRule
        deg = sum(p.adj, 1).';
        inn_xi = (p.adj.' * xi - deg .* xr) ./ max(deg, 1) / p.xi_lim;
        inn_ze = (p.adj.' * ze - deg .* zr) ./ max(deg, 1) / p.zeta_lim;
        a = zeros(2*n, 1);
        a(abs(inn_xi) > th.tau_xi) = -1;
        a(n + find(abs(inn_ze) > th.tau_zeta)) = -1;
        a = min(max(a, u - th.rate_max), u + th.rate_max);      % rate limit
        a(1:n)     = connectivity_floor(a(1:n), th);
        a(n+1:2*n) = connectivity_floor(a(n+1:2*n), th);
        u = a;
    end
end

% Metrics: same definitions as scripts/25_matlab_validation_scenarios.py
post = T >= atk.t_start;
fdev = F(post, :) - 50;
share = std(P(post, :), 0, 2) ./ (mean(abs(P(post, :)), 2) + 1e-9);
res.f_rms_mHz = 1e3 * sqrt(mean(fdev(:).^2));
res.f_max_mHz = 1e3 * max(abs(fdev(:)));
res.p_share   = mean(share);
res.collapsed = collapsed;
iso = all(TR(:, atk.targets) < 0.5, 2) & post;
if any(iso)
    res.latency_ms = 1e3 * (T(find(iso, 1)) - atk.t_start);
else
    res.latency_ms = NaN;
end
end

function x = advance(p, atk, u, x, t0, t1, opts)
if t1 - t0 < 1e-12, return; end
[~, X] = ode23tb(@(t, xx) mg_dynamics(t, xx, u, p, atk), [t0 t1], x, opts);
x = X(end, :).';
end

function block = connectivity_floor(block, th)
% Same as ResilientMicrogridEnv._enforce_connectivity.
k = th.min_trusted;
trust = min(max(1 + block, 0), 1);
ok = trust >= th.trust_ok;
deficit = k - sum(ok);
if deficit <= 0, return; end
[~, order] = sort(-trust);
order = order(~ok(order));
order = order(1:min(deficit, numel(order)));
block(order) = max(block(order), th.trust_ok - 1);
end
