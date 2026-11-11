%% deploy_policy.m
% Export the Python-trained policy into Simulink for high-fidelity validation.
%
% Pipeline:  PPO (PyTorch) -> ONNX -> importNetworkFromONNX -> Simulink block
%
% Final results should be produced HERE, on the switching model -- not on the
% reduced-order training plant. Train fast, validate faithfully, report the
% faithful numbers. If performance degrades on the switching model, report that
% honestly and quantify the gap; a measured sim-to-sim gap is a legitimate
% result and reviewers trust it far more than numbers that are suspiciously
% identical across fidelities.

clear; clc;

% Build the model first if it does not exist:
mdl = build_reference_model();
onnxFile = '../runs/ppo/actor.onnx';   % see scripts/07_export_onnx.py
net = importNetworkFromONNX(onnxFile, 'InputDataFormats', 'BC');

% Read the dimension from the network rather than hard-coding it. The literal
% 240 was history_len(4) x 6 quantities x 10 DGs; the observation now carries
% 8 quantities per DG (the per-source consensus innovation was added on both
% channels), so it is 320, and it changes again if the detector goes in. A
% wrong obsInfo does not error -- importNetworkFromONNX accepts it and the
% policy receives misaligned inputs.
obsDim  = net.Layers(1).InputSize;
obsDim  = obsDim(end);
fprintf('ONNX input dimension: %d (scripts/07_export_onnx.py prints this)\n', obsDim);
obsInfo = rlNumericSpec([obsDim 1], 'Name', 'observation');
actInfo = rlNumericSpec([20 1], 'Name', 'trust_offset', ...
                        'LowerLimit', -1, 'UpperLimit', 1);

% VecNormalize. The policy was trained on normalised observations; feeding raw
% ones produces confident nonsense rather than an obvious failure. Load the
% statistics exported alongside the network and apply them BEFORE the network.
statsFile = strrep(onnxFile, '.onnx', '_obsnorm.json');
assert(isfile(statsFile), ['Missing %s -- re-run scripts/07_export_onnx.py ' ...
       'with --vecnorm runs/ppo/vecnormalize.pkl'], statsFile);
st = jsondecode(fileread(statsFile));
normalizeObs = @(o) min(max((o - st.mean(:)) ./ sqrt(st.var(:) + st.epsilon), ...
                            -st.clip_obs), st.clip_obs);

actor  = rlContinuousDeterministicActor(net, obsInfo, actInfo);
policy = rlDeterministicActorPolicy(actor);
generatePolicyFunction(policy, 'FunctionName', 'residualPolicy', ...
                       'PolicyName', 'PIRLPolicy');

% ACTION SEMANTICS -- the most important line in this file.
% The 20 outputs are TRUST OFFSETS, not derivative residuals. In Simulink,
% normalize the observation with normalizeObs above, call residualPolicy(o),
% and feed the result into mg_dynamics as u, which applies
%     trust_j = min(max(1 + a_j, 0), 1)
% as a per-source weight on the consensus graph. Do NOT sum the output into
% the xi/zeta integrators: that is the abandoned first action space, and it
% fails silently -- the simulation runs, the numbers look plausible, and the
% deployed controller is not the one that was trained.
% The residual structure is preserved either way: a = 0 gives trust = 1
% everywhere, which IS the published consensus controller, and that is what
% the nominal-case stability argument rests on.

env = rlSimulinkEnv('Microgrid_RL_Model', 'Microgrid_RL_Model/RL Agent', ...
                    obsInfo, actInfo);
fprintf('Validating on the switching model...\n');
exp = sim(env, policy, rlSimulationOptions('MaxSteps', 600, 'NumSimulations', 20));
fprintf('Mean return on high-fidelity plant: %.3f\n', ...
        mean(arrayfun(@(e) sum(e.Reward.Data), exp)));
