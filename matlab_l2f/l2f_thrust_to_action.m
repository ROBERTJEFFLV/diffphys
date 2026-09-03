function action = l2f_thrust_to_action(thrust, c0, c1, c2)
%L2F_THRUST_TO_ACTION Invert thrust polynomial per rotor.

t_minus = c0 - c1 + c2;
t_plus = c0 + c1 + c2;
min_thrust = min(t_minus, t_plus);
max_thrust = max(t_minus, t_plus);
thrust = min(max(thrust, min_thrust), max_thrust);
eps_value = 1.0e-8;
action = zeros(size(thrust));
for i = 1:numel(thrust)
    if abs(c2(i)) > eps_value
        discr = max(c1(i) * c1(i) - 4 * c2(i) * (c0(i) - thrust(i)), 0.0);
        action(i) = (-c1(i) + sqrt(discr)) / (2 * c2(i));
    elseif abs(c1(i)) > eps_value
        action(i) = (thrust(i) - c0(i)) / c1(i);
    else
        action(i) = 0.0;
    end
end
action(~isfinite(action)) = 0.0;
action = min(max(action, -1.0), 1.0);
end
