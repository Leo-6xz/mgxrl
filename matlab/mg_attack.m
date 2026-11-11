function [xi_rx, zeta_rx] = mg_attack(t, xi, zeta, atk)
%MG_ATTACK  Corrupt the COMMUNICATED consensus signals (eqs. 9-17).
%
% As in the Python implementation, the corruption is applied to what neighbors
% receive, not to the attacked DG's local state. Corrupting local state models
% a sensor attack; corrupting the link models a communication attack, and only
% the latter reproduces the propagation through the consensus graph that
% Section III-C describes. Getting this wrong produces attack signatures that
% look plausible in isolation and fail the dataset correlation check.

xi_rx = xi;  zeta_rx = zeta;
if t < atk.t_start || strcmp(atk.type,'Normal')
    return
end
te  = t - atk.t_start;
tgt = atk.targets;

switch atk.type
    case 'Additive'      % (9)-(10)
        xi_rx(tgt)   = xi_rx(tgt)   + atk.bias_xi;
        zeta_rx(tgt) = zeta_rx(tgt) + atk.bias_zeta;

    case 'Ramp'          % (11)
        xi_rx(tgt)   = xi_rx(tgt)   + atk.slope * te;

    case 'SlowRamp'      % (12)  r_s << r
        zeta_rx(tgt) = zeta_rx(tgt) + atk.slope * te;

    case 'Sinusoid'      % (13)-(14)
        s = sin(2*pi*atk.freq*te);
        xi_rx(tgt)   = xi_rx(tgt)   + atk.amp_xi   * s;
        zeta_rx(tgt) = zeta_rx(tgt) + atk.amp_zeta * s;

    case 'Stealth'       % (15)-(16)
        base = atk.amp * (0.6*sin(2*pi*atk.freq*te) + ...
                          0.4*sin(2*pi*0.31*atk.freq*te + 1.1));
        for k = 1:numel(tgt)
            ph = 1.0 - 0.1*(k-1);
            xi_rx(tgt(k))   = xi_rx(tgt(k))   + base*ph;
            zeta_rx(tgt(k)) = zeta_rx(tgt(k)) + atk.alpha*base*ph;
        end

    case 'DoS'           % (17)  freeze at last received value
        xi_rx(tgt) = atk.held(tgt);
end
end
