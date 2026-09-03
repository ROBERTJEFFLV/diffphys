function values = l2f_linspace(a, b, n)
%L2F_LINSPACE Local replacement for linspace on broken MATLAB paths.

if n <= 1
    values = a;
    return;
end
values = zeros(1, n);
step = (b - a) / (n - 1);
for i = 1:n
    values(i) = a + (i - 1) * step;
end
end
