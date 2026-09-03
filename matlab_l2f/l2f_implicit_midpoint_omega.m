function omega_next = l2f_implicit_midpoint_omega(omega, torque, inertia, dt, iterations)
%L2F_IMPLICIT_MIDPOINT_OMEGA Match env_l2f implicit angular update.

if nargin < 5
    iterations = 4;
end

batch = size(omega, 1);
omega_next = zeros(size(omega));
for i = 1:batch
    w = omega(i, :);
    tau = torque(i, :);
    j = inertia(i, :);
    gyro = cross_row(w, w .* j);
    wn = w + dt * ((tau - gyro) ./ max(j, 1.0e-12));
    for k = 1:iterations
        wm = 0.5 * (w + wn);
        residual = wn - w - dt * ((tau - cross_row(wm, wm .* j)) ./ max(j, 1.0e-12));
        jac = eye(3) + (0.5 * dt) * (gyro_jacobian(wm, j) ./ max(j(:), 1.0e-12));
        delta = jac \ residual(:);
        wn = wn - delta(:).';
    end
    omega_next(i, :) = wn;
end
end

function c = cross_row(a, b)
c = [
    a(2) * b(3) - a(3) * b(2), ...
    a(3) * b(1) - a(1) * b(3), ...
    a(1) * b(2) - a(2) * b(1)
];
end

function jac = gyro_jacobian(w, inertia)
wx = w(1); wy = w(2); wz = w(3);
jx = inertia(1); jy = inertia(2); jz = inertia(3);
jac = [
    0, (jz - jy) * wz, (jz - jy) * wy;
    (jx - jz) * wz, 0, (jx - jz) * wx;
    (jy - jx) * wy, (jy - jx) * wx, 0
];
end
