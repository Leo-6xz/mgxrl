function Y = mg_ybus(p)
%MG_YBUS  Ring admittance matrix with constant-impedance loads folded in.
n = p.n;
yl = 1/complex(p.r_line, p.x_line);
Y  = zeros(n,n);
for i = 1:n
    j = mod(i,n) + 1;
    Y(i,i) = Y(i,i) + yl;   Y(j,j) = Y(j,j) + yl;
    Y(i,j) = Y(i,j) - yl;   Y(j,i) = Y(j,i) - yl;
end
% Constant-impedance load at 1.0 pu: Y_load = (P - jQ)/|V|^2
Y = Y + diag((p.load_p*p.load_scale) - 1j*(p.load_q*p.load_scale));
end
