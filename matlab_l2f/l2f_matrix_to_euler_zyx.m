function euler = l2f_matrix_to_euler_zyx(rotation)
%L2F_MATRIX_TO_EULER_ZYX Return [roll pitch yaw] for 3x3xN rotation.

batch = size(rotation, 3);
euler = zeros(batch, 3);
for i = 1:batch
    r = rotation(:, :, i);
    roll = atan2(r(3, 2), r(3, 3));
    pitch = -asin(min(max(r(3, 1), -1.0), 1.0));
    yaw = atan2(r(2, 1), r(1, 1));
    euler(i, :) = [roll, pitch, yaw];
end
end
