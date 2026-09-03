function k = l2f_skew(v)
%L2F_SKEW Skew-symmetric matrix for a 3-vector.

v = v(:);
k = [
    0, -v(3), v(2);
    v(3), 0, -v(1);
    -v(2), v(1), 0
];
end
