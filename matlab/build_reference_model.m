function mdl = build_reference_model(mdl)
%BUILD_REFERENCE_MODEL  Construct the Simulink model programmatically.
%
% This exists because the dataset author never released their Simulink model.
% Both validate_fidelity.m (Tier B) and deploy_policy.m referenced a model that
% does not exist on your machine; this attempts to build it.
%
% DESIGN CHOICE: one VECTORISED MATLAB Function block holding all 10 DGs, not
% ten subsystems. Reasons, in order of importance:
%   1. It calls mg_dynamics.m directly, so the Simulink model and the ODE
%      reference are literally the same code. Two hand-built implementations
%      will diverge, and you will spend a week finding out where.
%   2. ~8 blocks instead of ~200, so it can be built and diffed in a script
%      rather than clicked together and screenshotted.
%   3. rlSimulinkEnv attaches cleanly to a single RL Agent block.
%
% Run:  mdl = build_reference_model(); open_system(mdl)

if nargin < 1, mdl = 'Microgrid_RL_Model'; end
if bdIsLoaded(mdl), close_system(mdl, 0); end
new_system(mdl);

p  = mg_params();
n  = p.n;
nx = 5*n + p.use_inner_loops*n;
nu = 2*n;
ny = 9*n;   % V,f,P,Q,I,xi,zeta + xi_rx,zeta_rx (the innovation needs the
            % received values; see mg_dynamics.m)

set_param(mdl, 'Solver','ode23tb', 'StopTime','12.0', ...
               'RelTol','1e-8', 'AbsTol','1e-10', 'MaxStep','0.01');

% ---- plant: dx/dt and measurements from the shared dynamics function ----
plant = [mdl '/Plant'];
add_block('simulink/User-Defined Functions/MATLAB Function', plant, ...
          'Position',[220 100 340 200]);
script = sprintf([ ...
 'function [dx, y] = plantFcn(x, u, t)\n' ...
 '%%#codegen\n' ...
 'p   = mg_params();\n' ...
 'atk = getAttackConfig();\n' ...
 '[dx, y] = mg_dynamics(t, x, u, p, atk);\n' ...
 'end\n']);
setChartScript(plant, script);

% ---- integrator ----
intg = [mdl '/States'];
add_block('simulink/Continuous/Integrator', intg, 'Position',[400 120 440 160]);
x0 = zeros(nx,1);
x0(n+1:2*n)   = p.load_p;
x0(2*n+1:3*n) = p.load_q;
if p.use_inner_loops, x0(5*n+1:6*n) = p.v_nom; end
set_param(intg, 'InitialCondition', mat2str(x0));

% ---- clock ----
add_block('simulink/Sources/Clock', [mdl '/Clock'], 'Position',[100 200 120 220]);

% ---- RL agent ----
% The agent outputs 20 TRUST OFFSETS, which mg_dynamics converts to per-source
% weights trust = min(max(1 + a, 0), 1) on the consensus graph. It does NOT
% output a derivative residual -- that was the first action space and it was
% abandoned. a = 0 gives trust = 1 everywhere, which IS the published consensus
% controller, so the residual structure and its nominal-case stability argument
% survive; wiring the output into the integrators instead would fail silently.
agent = [mdl '/RL Agent'];
try
    add_block('rl_lib/RL Agent', agent, 'Position',[220 260 340 320]);
catch
    warning(['RL Agent block unavailable -- building open-loop. ' ...
             'Install Reinforcement Learning Toolbox for closed-loop use.']);
    add_block('simulink/Sources/Constant', agent, 'Position',[220 260 340 320], ...
              'Value', mat2str(zeros(nu,1)));
end

% ---- observation / reward / isdone for the RL interface ----
obs = [mdl '/Observation'];
add_block('simulink/User-Defined Functions/MATLAB Function', obs, ...
          'Position',[520 240 640 320]);
% The per-step frame must match ResilientMicrogridEnv._observe EXACTLY, in
% order and in scaling: V, f, P, Q, xi, zeta, and the per-source consensus
% innovation on both channels -- 8 quantities per DG, not 6. The policy is fed
% whatever this produces; a mismatch does not error, it silently permutes the
% inputs. The innovation needs the RECEIVED values, so mg_dynamics exposes
% xi_rx / zeta_rx through y (indices 7n+1 : 9n).
%
% NOTE ON HISTORY. Python stacks history_len = 4 frames. This block emits ONE
% frame; place a Tapped Delay (4 taps, 0.01 s) between it and the RL Agent so
% the agent receives [t-3; t-2; t-1; t] in that order. Without the delay line
% the observation is a quarter of the expected width and importNetworkFromONNX
% will accept it.
setChartScript(obs, sprintf([ ...
 'function [o, r, d] = obsFcn(y, u)\n' ...
 '%%#codegen\n' ...
 'n = 10;\n' ...
 'p = mg_params();\n' ...
 'V = y(1:n); f = y(n+1:2*n); P = y(2*n+1:3*n); Q = y(3*n+1:4*n);\n' ...
 'xi = y(5*n+1:6*n); ze = y(6*n+1:7*n);\n' ...
 'xi_rx = y(7*n+1:8*n); ze_rx = y(8*n+1:9*n);\n' ...
 'deg = max(sum(p.adj,1).'', 1);\n' ...
 'inn_xi = (p.adj.''*xi - sum(p.adj,1).''.*xi_rx) ./ deg;\n' ...
 'inn_ze = (p.adj.''*ze - sum(p.adj,1).''.*ze_rx) ./ deg;\n' ...
 'o = [V-1.0; (f-50)*10; P-0.5; Q-0.2; xi/p.xi_lim; ze/p.zeta_lim; ...\n' ...
 '     inn_xi/p.xi_lim; inn_ze/p.zeta_lim];\n' ...
 '%% Reward mirrors mgxrl/envs/resilient_env.py. The control cost is\n' ...
 '%% CONNECTIVITY LOSS mean(1 - trust), not mean(u.^2): under trust weights\n' ...
 '%% every u >= 0 is full trust and costs nothing, so charging u.^2 taxes a\n' ...
 '%% harmless half of the action space and understates real isolation.\n' ...
 '%% The old 0.10 coefficient is also the value that made saturation free and\n' ...
 '%% collapsed the first policy; w_action is 1.0.\n' ...
 'refs = [5.0e-4; 1.8e-3; 6.0e-3; 1.8e-1];\n' ...
 'wts  = [3.0; 2.0; 1.5; 1.0];\n' ...
 'met  = [sqrt(mean((f-50).^2)); sqrt(mean((V-1).^2)); ...\n' ...
 '        std(P)/(mean(abs(P))+1e-9); std(Q)/(mean(abs(Q))+1e-9)];\n' ...
 'trust = min(max(1 + u, 0), 1);\n' ...
 'iso = mean(1 - trust);\n' ...
 'viol = 50.0 * (any(V < 0.95) || any(V > 1.05));\n' ...
 'r = 9.0 - sum(wts .* log1p((met./refs).^2)) - 1.0*iso - 6.0*(iso > 0.5) - viol;\n' ...
 'd = any(V < 0.5) || any(V > 1.5) || any(abs(f-50) > 0.5);\n' ...
 'end\n']));

% ---- logging ----
add_block('simulink/Sinks/To Workspace', [mdl '/LogY'], ...
          'Position',[520 100 600 140], 'VariableName','y_log', ...
          'SaveFormat','Structure With Time', 'SampleTime','0.01');

% ---- wiring ----
add_line(mdl,'Plant/1','States/1','autorouting','on');
add_line(mdl,'States/1','Plant/1','autorouting','on');
add_line(mdl,'RL Agent/1','Plant/2','autorouting','on');
add_line(mdl,'Clock/1','Plant/3','autorouting','on');
add_line(mdl,'Plant/2','LogY/1','autorouting','on');
add_line(mdl,'Plant/2','Observation/1','autorouting','on');
add_line(mdl,'RL Agent/1','Observation/2','autorouting','on');
if strcmp(get_param(agent,'BlockType'),'SubSystem')
    add_line(mdl,'Observation/1','RL Agent/1','autorouting','on');
    add_line(mdl,'Observation/2','RL Agent/2','autorouting','on');
    add_line(mdl,'Observation/3','RL Agent/3','autorouting','on');
end

Simulink.BlockDiagram.arrangeSystem(mdl);
save_system(mdl);
fprintf('Built %s: %d states, %d actions, %d measurements.\n', mdl, nx, nu, ny);
fprintf('Attack scenario is read from getAttackConfig.m -- edit that to switch.\n');
fprintf(['DoS: getAttackConfig returns atk.held = zeros, which is a placeholder.\n' ...
         'Set it from the settled xi before sim() or DoS will be far more\n' ...
         'severe here than in Python. mg_simulate.m does this automatically.\n']);
fprintf(['Observation: place a Tapped Delay (4 taps, 0.01 s) between the\n' ...
         'Observation block and the RL Agent -- Python stacks history_len = 4.\n']);
end

% -------------------------------------------------------------------------
function setChartScript(blk, script)
%SETCHARTSCRIPT  Set the code inside a MATLAB Function block.
% The Stateflow API is the only supported way to do this programmatically;
% set_param has no equivalent property.
r = sfroot;
c = r.find('-isa','Stateflow.EMChart','Path',blk);
if isempty(c)
    error('Could not locate the MATLAB Function chart at %s', blk);
end
c.Script = script;
end
