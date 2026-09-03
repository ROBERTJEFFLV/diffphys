function r = l2f_so3_exp(phi)
%L2F_SO3_EXP Exponential map on SO(3).

phi = phi(:);
theta_sq = sum(phi .* phi);
theta = sqrt(theta_sq);
k = l2f_skew(phi);
k2 = k * k;
if theta_sq < 1.0e-8
    theta_sq2 = theta_sq * theta_sq;
    a = 1.0 - theta_sq / 6.0 + theta_sq2 / 120.0;
    b = 0.5 - theta_sq / 24.0 + theta_sq2 / 720.0;
else
    a = sin(theta) / max(theta, 1.0e-4);
    b = (1.0 - cos(theta)) / max(theta_sq, 1.0e-8);
end
r = eye(3) + a * k + b * k2;
end
