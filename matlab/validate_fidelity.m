%% validate_fidelity.m
% Fidelity validation of the reduced-order RL training plant.
%
% WHY THREE TIERS
% The RL agent is trained on a reduced-order averaged model, because a 2 us
% switching plant makes RL infeasible: one 10 ms control step = 5000 solver
% steps, so a 2M-step training run is ~1e10 solver steps. That choice is
% standard and defensible ONLY with evidence that the reduced model reproduces
% the reference dynamics in the regime the agent operates in. Without it, the
% likely question -- "your agent never saw the real plant" -- has no
% answer.
%
% But note what each tier can and cannot establish:
%
%   TIER A  Python (fwd-Euler, 1 ms, algebraic V)  vs  MATLAB (ode23tb, inner
%           voltage loop). SAME equations, so this is NOT a test of the model
%           structure. It tests exactly two things: discretisation error, and
%           whether neglecting the inner loops at the secondary timescale was
%           safe. Both are real questions; neither is the big one.
%
%   TIER B  vs a switching-level Simscape model at 2 us. Tests the averaging
%           assumption. Must be built in the Simulink GUI with Specialized
%           Power Systems blocks -- see BUILD NOTES at the bottom of this file.
%           Do this after Tier A passes, not before.
%
%   TIER C  vs the actual IEEE Dataport measurements. THIS IS THE ONLY TIER
%           THAT IS NOT SELF-REFERENTIAL, and it is the one to lead with in the
%           paper. Run scripts/06_validate_against_dataset.py. If Tier A and B
%           pass but C fails, the model is internally consistent and wrong.
%
% Acceptance criteria (report the table):
%   frequency trajectory RMSE       < 0.01 Hz
%   voltage trajectory RMSE         < 0.002 pu
%   attack-signature correlation    > 0.95   (Tier A)   > 0.90 (Tier B/C)

clear; clc;
addpath(fileparts(mfilename('fullpath')));

p = mg_params();
attacks = {'Normal','Additive','Ramp','SlowRamp','Sinusoid','Stealth','DoS'};
T = 12.0;  dt_log = 0.010;
resdir = fullfile(fileparts(mfilename('fullpath')), '..', 'results');
if ~exist(resdir,'dir'), mkdir(resdir); end

results = table();

for k = 1:numel(attacks)
    name = attacks{k};
    atk  = mg_attack_config(name, p);

    % ---------- reference trajectory ----------
    ref = mg_simulate(atk, p, T, dt_log);

    % ---------- reduced-order trajectory from the Python plant ----------
    % Generate first:  python scripts/05_export_reference_traj.py --out results/
    f_red = fullfile(resdir, sprintf('reduced_%s.csv', name));
    if ~isfile(f_red)
        warning('Missing %s -- run scripts/05_export_reference_traj.py first.', f_red);
        continue
    end
    R = readmatrix(f_red);            % [t, V(1:10), f(1:10), P(1:10), Q(1:10)]
    m = min(size(R,1), numel(ref.t));

    red_V = R(1:m, 1+(1:p.n));
    red_f = R(1:m, 1+p.n+(1:p.n));
    red_P = R(1:m, 1+2*p.n+(1:p.n));

    f_rmse = sqrt(mean((ref.f(1:m,:) - red_f).^2, 'all'));
    V_rmse = sqrt(mean((ref.V(1:m,:) - red_V).^2, 'all'));
    P_rmse = sqrt(mean((ref.P(1:m,:) - red_P).^2, 'all'));

    % Signature correlation is computed on the DEVIATION from nominal, not on
    % the raw trajectory. Raw trajectories are dominated by the steady-state
    % operating point and correlate at >0.99 even for completely wrong attack
    % models -- a check that always passes tells you nothing.
    dev_ref = mean(ref.f(1:m,:),2) - p.f_nom;
    dev_red = mean(red_f,2)        - p.f_nom;
    if std(dev_ref) > 1e-9 && std(dev_red) > 1e-9
        % corrcoef (base MATLAB) rather than corr (Statistics and Machine
        % Learning Toolbox). This keeps Tier A runnable on a bare MATLAB
        % licence -- no toolboxes required until Phase 7.
        C = corrcoef(dev_ref, dev_red);
        sig_corr = C(1,2);
    else
        sig_corr = NaN;   % Normal case: no deviation to correlate
    end

    verdict = "PASS";
    if f_rmse > 0.01 || V_rmse > 0.002 || (~isnan(sig_corr) && sig_corr < 0.95)
        verdict = "REVIEW";
    end

    results = [results; table(string(name), f_rmse, V_rmse, P_rmse, sig_corr, verdict, ...
        'VariableNames', {'Attack','f_RMSE_Hz','V_RMSE_pu','P_RMSE_pu', ...
                          'SignatureCorr','Verdict'})]; %#ok<AGROW>

    fprintf('%-10s f=%.5f Hz  V=%.5f pu  P=%.5f pu  corr=%.4f  %s\n', ...
            name, f_rmse, V_rmse, P_rmse, sig_corr, verdict);
end

writetable(results, fullfile(resdir,'fidelity_tierA.csv'));
disp(results);

%% Diagnostic: isolate the cause of any discrepancy
% If a row shows REVIEW, rerun that attack with the inner loops disabled. If
% the discrepancy vanishes, the inner loops matter at the secondary timescale
% and the Python plant needs a voltage lag added. If it persists, the cause is
% discretisation and you should reduce the Python plant's dt (or switch it to
% RK4) rather than changing the model structure.
%
%   p2 = mg_params(); p2.use_inner_loops = false;
%   ref2 = mg_simulate(mg_attack_config('SlowRamp',p2), p2, T, dt_log);

%% BUILD NOTES -- Tier B (switching-level Simscape model)
% Only after Tier A passes. Outline, in the Simulink GUI:
%   1. Specialized Power Systems > powergui, discrete, sample time 2e-6.
%   2. One "Universal Bridge" (IGBT/diode, 3 arms) per DG, PWM-driven, with an
%      LC output filter (L = 0.15 pu, C sized for a 1 kHz corner) and feeder
%      impedance matching p.r_line / p.x_line.
%   3. Per DG: abc->dq0 (PLL-synchronised), cascaded PI current and voltage
%      regulators, then P-f / Q-V droop feeding the voltage reference.
%   4. Secondary layer: implement eqs. (3)-(6) in a single vectorised MATLAB
%      Function block operating on all 10 DGs at a 10 ms rate (Rate Transition
%      blocks at the boundary). Do NOT build ten separate subsystems -- the
%      vectorised form is what lets you reuse mg_dynamics and keeps the two
%      implementations comparable.
%   5. Attack injection: place mg_attack.m in a MATLAB Function block on the
%      consensus signal path, between the DG outputs and the neighbour inputs.
%      It must sit on the COMMUNICATION path, not on the local state.
%   6. Log V, I, P, Q, f per DG to the workspace, then decimate by 5000 with
%      block-averaging (not subsampling) before comparing.
%
% Expect Tier B to show larger P_RMSE than Tier A because of switching ripple.
% That is not a failure. Compare the block-averaged signals, and report the
% ripple magnitude separately.

function atk = mg_attack_config(name, p)
%MG_ATTACK_CONFIG  Attack parameters, matching mgxrl/plant/attacks.py defaults.
atk.type      = name;
atk.t_start   = 1.5;
atk.targets   = 1;
atk.bias_xi   = 0.3;    atk.bias_zeta = 0.01;
atk.slope     = 0.5;                       % Ramp: rad/s per s
atk.amp_xi    = 0.4;    atk.amp_zeta = 0.012;
atk.freq      = 2.0;
atk.amp       = 0.06;   atk.alpha    = 0.02;
atk.held      = zeros(p.n,1);

switch name
    case 'SlowRamp', atk.slope = 0.004;              % pu per s
    case 'Stealth',  atk.targets = [1 2 3]; atk.freq = 0.7;
end
end
