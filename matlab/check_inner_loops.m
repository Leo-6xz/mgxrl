%% check_inner_loops.m
% Tier A diagnostic: are the inner loops actually exercised, or does Tier A
% test discretisation only?
%
% WHY THIS IS A SEPARATE TEST FROM validate_fidelity.m
% Tier A compares Python (fwd-Euler, 1 ms, algebraic V) against MATLAB
% (ode23tb, inner voltage loop) on the SAME equations. It reported f_RMSE from
% 4e-8 to 5e-5 Hz, which reads as a pass. But two different things could
% produce that:
%
%   (a) the inner loops are active and the reduced model tracks them well, or
%   (b) the inner loops are so fast relative to the 10 ms secondary timescale
%       that they contribute nothing, in which case Tier A tested
%       discretisation and NOTHING ELSE.
%
% Those have different consequences for the paper. Under (b) the claim
% "neglecting the inner loops at the secondary timescale is safe" is untested
% by Tier A, and the paper must say so rather than implying the comparison
% covered it.
%
% The diagnostic block at the bottom of validate_fidelity.m suggests rerunning
% with the flag off and comparing against the Python trajectory. That conflates
% two effects: any change would be a mixture of the inner loops mattering and
% the reduced model's own error. This compares the MATLAB reference AGAINST
% ITSELF with the flag on and off, which isolates the question completely -- it
% does not involve the Python plant at all.
%
% DECISION RULE
%   max|f1 - f2| at or near solver noise  -> the inner loops are not exercised.
%       Tier A tests discretisation only. Say that in the paper, and either
%       drop the inner-loop claim or support it with Tier B.
%   max|f1 - f2| comparable to the Tier A f_RMSE or larger -> they ARE
%       exercised, Tier A's agreement is meaningful, and the claim stands.
%
% Classes are chosen for dynamic content. Normal and DoS both sit at the same
% fixed point and agreed at 1e-11 in the Tier A table, so they test equilibrium
% rather than dynamics and are useless here.
%
% Run (MATLAB, from the matlab/ directory):
%   check_inner_loops

clear; clc;
addpath(fileparts(mfilename('fullpath')));

p1 = mg_params();

% GUARD. In MATLAB, assigning to a field that does not exist CREATES it
% silently. If mg_params or mg_simulate never reads use_inner_loops, the run
% below would produce an exact zero difference and read as "the inner loops do
% not matter" when in fact the flag did nothing at all. That failure mode is
% indistinguishable from the real result, so check first and stop if absent.
if ~isfield(p1, 'use_inner_loops')
    error(['mg_params() has no field use_inner_loops. This test cannot ', ...
           'distinguish "flag has no effect" from "inner loops have no ', ...
           'effect". Add the field and confirm mg_dynamics reads it before ', ...
           'running this.']);
end
fprintf('use_inner_loops present, default = %d\n\n', p1.use_inner_loops);

p2 = p1;  p2.use_inner_loops = false;

T = 12.0;  dt_log = 0.010;
classes = {'Ramp','SlowRamp','Sinusoid'};
res = table();

for k = 1:numel(classes)
    name = classes{k};
    a1 = local_attack_config(name, p1);
    a2 = local_attack_config(name, p2);

    r1 = mg_simulate(a1, p1, T, dt_log);   % inner loops ON
    r2 = mg_simulate(a2, p2, T, dt_log);   % inner loops OFF

    m = min(size(r1.f,1), size(r2.f,1));
    df = max(abs(r1.f(1:m,:) - r2.f(1:m,:)), [], 'all');
    dV = max(abs(r1.V(1:m,:) - r2.V(1:m,:)), [], 'all');
    dP = max(abs(r1.P(1:m,:) - r2.P(1:m,:)), [], 'all');
    rf = sqrt(mean((r1.f(1:m,:) - r2.f(1:m,:)).^2, 'all'));

    % The comparison that matters: this difference against the Tier A f_RMSE
    % for the same class. Read them side by side from results/fidelity_tierA.csv.
    res = [res; table(string(name), df, rf, dV, dP, ...
        'VariableNames', {'Attack','max_df_Hz','rmse_df_Hz', ...
                          'max_dV_pu','max_dP_pu'})]; %#ok<AGROW>

    fprintf('%-10s max|df| = %.3e Hz   rmse = %.3e Hz   max|dV| = %.3e pu\n', ...
            name, df, rf, dV);
end

resdir = fullfile(fileparts(mfilename('fullpath')), '..', 'results');
if ~exist(resdir,'dir'), mkdir(resdir); end
writetable(res, fullfile(resdir,'tierA_inner_loop_check.csv'));
disp(res);

fprintf(['\nCompare max_df_Hz against f_RMSE_Hz in results/fidelity_tierA.csv\n', ...
         'for the same class (Tier A reported 4e-8 to 5e-5 Hz).\n', ...
         '  much SMALLER -> inner loops contribute less than the Python-MATLAB\n', ...
         '     gap itself. Tier A tested discretisation; say so.\n', ...
         '  COMPARABLE or larger -> the inner loops are exercised and the\n', ...
         '     agreement in Tier A is evidence that neglecting them is safe.\n']);

function atk = local_attack_config(name, p)
%LOCAL_ATTACK_CONFIG  Copy of the config in validate_fidelity.m, which defines
% it as a local function and therefore cannot be called from here. If the
% defaults in that file change, change them here too -- they must stay matched
% or this diagnostic is run on a different attack than Tier A was.
atk.type      = name;
atk.t_start   = 1.5;
atk.targets   = 1;
atk.bias_xi   = 0.3;    atk.bias_zeta = 0.01;
atk.slope     = 0.5;
atk.amp_xi    = 0.4;    atk.amp_zeta = 0.012;
atk.freq      = 2.0;
atk.amp       = 0.06;   atk.alpha    = 0.02;
atk.held      = zeros(p.n,1);

switch name
    case 'SlowRamp', atk.slope = 0.004;
    case 'Stealth',  atk.targets = [1 2 3]; atk.freq = 0.7;
end
end
