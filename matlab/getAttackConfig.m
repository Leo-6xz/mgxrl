function atk = getAttackConfig()
%GETATTACKCONFIG  Attack scenario for the Simulink model.
%#codegen
%
% Separated from build_reference_model.m so scenarios can be swapped without
% rebuilding the model. Values mirror mgxrl/plant/attacks.py defaults.
%
% Set via:  assignin('base','ATTACK_TYPE','Ramp')  before sim(), or edit here.

atk.type      = 'Ramp';
atk.t_start   = 1.5;
atk.targets   = 1;
atk.bias_xi   = 0.3;    atk.bias_zeta = 0.01;
atk.slope     = 0.5;
atk.amp_xi    = 0.4;    atk.amp_zeta  = 0.012;
atk.freq      = 2.0;
atk.amp       = 0.06;   atk.alpha     = 0.02;
% DoS: the value the frozen signal is held at. zeros(10,1) is a PLACEHOLDER
% and is wrong for a live simulation -- the settled xi is around 0.1-0.2 rad/s,
% so freezing at 0 is a far larger perturbation than the attack being modelled
% and DoS will look much more severe than in Python. mg_simulate.m overrides
% this by integrating to t_start and capturing xi there. Anything driving the
% Simulink model must do the same before sim(); see the note in
% build_reference_model.m. script
atk.held      = zeros(10,1);
atk.held_is_placeholder = true;
end
