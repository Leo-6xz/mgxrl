%% fit_noise_from_dataset.m
% Fit measurement-noise statistics from the Dataport CSVs and write them into
% the reduced-order plant config.
%
% This is the third legitimate use of the dataset (after detector training and
% fidelity benchmarking): the reduced-order model has no switching, so its
% noise floor must be borrowed from data rather than invented. Taking the
% residual after detrending a steady-state Normal window gives a defensible
% estimate and lets you write "noise statistics calibrated to the published
% dataset" instead of "noise added for realism".

root = 'C:\research\mgxrl\Dataset';        
d = dir(fullfile(root,'Normal','*.csv'));
assert(~isempty(d), 'No CSVs found in %s', fullfile(root,'Normal'));
T = readtable(fullfile(d(1).folder, d(1).name));

% Skip the startup transient. The release begins from ZERO initial conditions
% and the voltage settles around 0.014 s, so the first samples are an
% energisation event whose "noise" is the transient.
t0    = 0.05;
keep  = T{:,1} >= t0 & T{:,1} < 0.45;    % steady state, before onset at 0.500
win   = T(keep, :);
assert(height(win) > 20000, 'Too few steady-state samples.');

% V1/V2/V3 and I1/I2/I3 are INSTANTANEOUS 50 Hz phase waveforms, not
% magnitudes -- dataio.classify_ac_columns detects this from |mean|/RMS. The
% previous line took std(x - movmean(x, 5000)); 5000 samples at 2 us is 10 ms,
% HALF a 50 Hz cycle, so the moving average does not remove the carrier and the
% residual is dominated by the ~141 V waveform rather than by measurement
% noise. The estimate was wrong by several orders of magnitude.
%
% Estimate the voltage noise from the cycle-by-cycle RMS ENVELOPE instead: RMS
% over an integer number of cycles removes the carrier exactly, and what varies
% between cycles is what we want.
fs        = 1/2e-6;
n_cycle   = round(fs/50);                       % samples per 50 Hz cycle
n_full    = floor(height(win)/n_cycle)*n_cycle;
Vrms      = sqrt(mean(reshape(win.V1(1:n_full).^2, n_cycle, []), 1)).';
sigma_V   = std(Vrms - movmean(Vrms, 21)) / mean(Vrms);   % relative -> pu

% P, Q and f are already-computed smooth quantities, so detrending works
% directly. Use a window of several cycles so the secondary-control dynamics
% are removed but the sample-to-sample noise is not.
detrended = @(x) x - movmean(x, n_cycle*5);
sigma_P   = std(detrended(win.PDG1)) / mean(abs(win.PDG1));   % relative -> pu
sigma_Q   = std(detrended(win.QDG1)) / mean(abs(win.QDG1));
sigma_f   = std(detrended(win.fDG1));                          % already Hz

fprintf('noise_v: %.3e\nnoise_p: %.3e\nnoise_q: %.3e\nnoise_f: %.3e\n', ...
        sigma_V, sigma_P, sigma_Q, sigma_f);
fprintf(['\nPaste these into configs/microgrid.yaml AND into\n' ...
         'MicrogridParams in src/mgxrl/plant/microgrid.py -- the YAML is\n' ...
         'read only by scripts/08_sync_matlab_params.py, so editing it\n' ...
         'alone changes MATLAB and leaves Python untouched.\n']);
