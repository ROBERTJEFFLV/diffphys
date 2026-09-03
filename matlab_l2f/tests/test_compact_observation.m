function tests = test_compact_observation
%TEST_COMPACT_OBSERVATION Verify 22D/25D/legacy40 observation semantics.
tests = functiontests(localfunctions);
end

function testDimensionsAndAffineLegacyDuplicate(testCase)
state = fixture_state();
integral = [0.1, -0.2, 0.3; -0.4, 0.5, -0.6];
compact = l2f_observation(state, 'compact22', integral);
with_integral = l2f_observation(state, 'integral25', integral);
legacy = l2f_observation(state, 'legacy40', integral);
verifySize(testCase, compact, [2 22]);
verifySize(testCase, with_integral, [2 25]);
verifySize(testCase, legacy, [2 40]);
verifyEqual(testCase, with_integral(:, 19:21), integral, 'AbsTol', 0);
expected_first = [ ...
    1 2 3, 0.1 0.2 0.3, ...
    1 0 0 0 1 0 0 0 1, ...
    0.4 0.5 0.6, 0.1 -0.2 0.3, 0.1 0.2 0.3 0.4];
verifyEqual(testCase, with_integral(1, :), expected_first, 'AbsTol', 0);
duplicate = legacy(:, 19:36);
duplicate(:, [7 11 15]) = duplicate(:, [7 11 15]) + 1.0;
verifyEqual(testCase, duplicate, legacy(:, 1:18), 'AbsTol', 0);
end

function testIntegralClampLeakAndObservedPosition(testCase)
integral = ones(2, 3);
observed = [2 -3 0.5; -2 3 -0.5];
updated = l2f_update_position_integral(integral, observed, 0.1, 1.0, 0.5);
expected = min(max(0.95 .* integral + 0.1 .* observed, -1.0), 1.0);
verifyEqual(testCase, updated, expected, 'AbsTol', 1.0e-15);
end

function testWorldIntegralRotatesIntoBodyFrame(testCase)
state = fixture_state();
integral = [1 0 0; -0.4 0.5 -0.6];
body_observation = l2f_observation(state, 'integral25', integral, 'body');
verifyEqual(testCase, body_observation(1, 19:21), [1 0 0], 'AbsTol', 0);
verifyEqual(testCase, body_observation(2, 19:21), [0.5 0.4 -0.6], 'AbsTol', 1.0e-15);
end

function testIntegralInputMultiplierOnlyScalesFeature(testCase)
state = fixture_state();
integral = [1 0 0; -0.4 0.5 -0.6];
unscaled = l2f_observation(state, 'integral25', integral, 'body', 1.0);
scaled = l2f_observation(state, 'integral25', integral, 'body', 4.0);
verifyEqual(testCase, scaled(:, 1:18), unscaled(:, 1:18), 'AbsTol', 0);
verifyEqual(testCase, scaled(:, 19:21), 4.0 .* unscaled(:, 19:21), 'AbsTol', 0);
verifyEqual(testCase, scaled(:, 22:25), unscaled(:, 22:25), 'AbsTol', 0);
end

function state = fixture_state()
state = struct();
state.position = [1 2 3; -1 -2 -3];
state.velocity = [0.1 0.2 0.3; -0.1 -0.2 -0.3];
state.rotation = zeros(3, 3, 2);
state.rotation(:, :, 1) = eye(3);
state.rotation(:, :, 2) = [0 -1 0; 1 0 0; 0 0 1];
state.omega = [0.4 0.5 0.6; -0.4 -0.5 -0.6];
state.previous_action = [0.1 0.2 0.3 0.4; -0.1 -0.2 -0.3 -0.4];
end
