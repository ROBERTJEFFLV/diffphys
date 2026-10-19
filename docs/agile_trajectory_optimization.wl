(* Offline symbolic derivation for the Wolfram tool.
   The production planner runs in PyTorch and has no Wolfram dependency.
   l is half-span, b is the transverse bend magnitude, t is motion duration.
   Normalize the unchanged position_limit B to 1 for the shape optimization.
   This shape optimum is conditional on the chosen path family and clearances;
   individual-rotor and RK4 motor feasibility are checked separately in Python.
   Current defaults request 105 degrees and H500 at 100 Hz. There is no fixed
   final hold interval: after the motion duration, keep commanding the endpoint.
   Configured angles lie in [0,180) degrees; actual screened angles are reported.
*)

ClearAll[u, q, l, b, t, s, theta, eta, timeLaw, spatialPath, path,
  midpoint, arcLength, snapEnergy, reducedSnap, optimalBend,
  tiltActual, tiltDerivative, tiltExpectedDerivative, tiltAssumptions,
  along, curvature, lateralBend, verticalBend, gravity,
  transverseForceSquared, transverseSquareCompletion, transverseLowerBound,
  projectionNormSquared, forceAssumptions,
  axis, heading, ex, ey, ez, z1, z2, z3, projection,
  projectionFirst, projectionSecond, raw, x1, x2, x3, rawNorm, unit,
  normFirst, normSecond, unitFirst, unitSecond, angle, rollRotation, vee];

timeLaw = 126 u^5 - 420 u^6 + 540 u^7 - 315 u^8 + 70 u^9;
spatialPath = {l (2 q - 1), 4 b q (1 - q)};
path = spatialPath /. q -> timeLaw;

midpoint = FullSimplify[
  {path, D[path, u]/t, D[path, {u, 2}]/t^2} /. u -> 1/2];
(* {{0,b}, {315 l/(64 t),0}, {0,-99225 b/(2048 t^2)}} *)

arcLength = FullSimplify[
  Integrate[Sqrt[D[spatialPath, q].D[spatialPath, q]], {q, 0, 1},
    Assumptions -> {l > 0, b > 0}]];
(* Equivalent stable form:
   Sqrt[l^2+4 b^2] + l^2 ArcSinh[2 b/l]/(2 b).
   The straight-path limit is 2 l; length is independent of duration.
*)

snapEnergy = Factor[
  Integrate[Expand[D[path, {u, 4}].D[path, {u, 4}]], {u, 0, 1}]/t^7];
(* 2419200 (120285164 b^2 + 8402199 l^2)/(30808063 t^7) *)

(* At a fixed requested midpoint tilt and thrust ratio, t^2 is proportional
   to b. Remove the positive constant independent of b from the snap cost.
   The user-requested longer half-span is 0.70 B. Cap b at 0.38 B so the
   middle Bezier control point has norm <=0.76 B, inside the 0.80 B margin.
*)
reducedSnap = (120285164 b^2 + 8402199 l^2)/b^(7/2);
optimalBend = Minimize[
  {reducedSnap /. l -> 7/10, 1/5 <= b <= 19/50}, b, Reals];
(* Exact minimizing bend: b -> 19/50. Its derivative is strictly negative
   for positive l,b; this is not a claim about arbitrary trajectory families.
*)

(* With acceleration scale s=(T_ideal/T)^2, use the full two-argument angle:
     horizontal = s eta Sin[theta]
     vertical   = 1+s (eta Cos[theta]-1).
   Wolfram ArcTan[x,y] is the angle of {x,y}; its argument order is the reverse
   of Python atan2(y,x). This expression stays valid through horizontal thrust
   and into negative vertical thrust. A one-argument arctangent or tangent
   quotient would lose the distinction between acute and obtuse tilts.
   Strict monotonicity holds for 0<theta<Pi; theta=0 is the upright endpoint
   for 0<s<=1. Thus the shortest feasible tested duration gives the largest
   actual midpoint tilt among the tested durations, including above 90 degrees.
*)
tiltAssumptions = {s > 0, eta > 0, 0 < theta < Pi};
tiltActual = ArcTan[1+s (eta Cos[theta]-1), s eta Sin[theta]];
tiltDerivative = FullSimplify[
  D[tiltActual, s], Assumptions -> tiltAssumptions];
tiltExpectedDerivative = eta Sin[theta] /
  ((s eta Sin[theta])^2+(1+s (eta Cos[theta]-1))^2);

(* Projected reference-heading frame:
     b3 = f/Norm[f], x = e-(e.b3)b3, b1 = x/Norm[x], b2 = Cross[b3,b1].
   This is a projected-heading convention, not fixed Euler yaw at every time.
   At the midpoint f.e=0, so x=e even at exactly 90 degrees or above.

   In the orthonormal {e,n,up} basis the S9 curve has
     f = {along, curvature lateralBend, gravity+curvature verticalBend}.
   A projected-heading singularity would require f parallel to e, meaning the
   final two components both vanish. Completing the square proves this cannot
   occur if gravity>0 and lateralBend!=0. The latter holds for an active curved
   maneuver with 0<theta<Pi. The bound can approach zero near theta=Pi; it is
   not a uniform conditioning or actuator-feasibility guarantee near 180 degrees.
   Straight, upright or zero-amplitude references retain numerical norm checks.
*)
forceAssumptions = {
  Element[{along, curvature, lateralBend, verticalBend, gravity}, Reals],
  gravity > 0, lateralBend != 0};
transverseForceSquared =
  (curvature lateralBend)^2+(gravity+curvature verticalBend)^2;
transverseLowerBound =
  gravity^2 lateralBend^2/(lateralBend^2+verticalBend^2);
transverseSquareCompletion = (
  (lateralBend^2+verticalBend^2) *
    (curvature+gravity verticalBend/(lateralBend^2+verticalBend^2))^2 +
  transverseLowerBound);
projectionNormSquared =
  transverseForceSquared/(along^2+transverseForceSquared);

(* Analytic first and second derivatives of the projected heading. The three
   components below are generic functions of time; e is constant. No assumption
   of positive world-vertical thrust enters either identity.
*)
axis = {z1[u], z2[u], z3[u]};
heading = {ex, ey, ez};
projection = heading-(heading.axis) axis;
projectionFirst =
  -(heading.D[axis, u]) axis-(heading.axis) D[axis, u];
projectionSecond = (
  -(heading.D[axis, {u, 2}]) * axis -
  2 * (heading.D[axis, u]) * D[axis, u] -
  (heading.axis) * D[axis, {u, 2}]);

(* Derivative identities for the normalization helper used first for the thrust
   axis and then for the projected heading. Its domain is a real nonzero vector.
   Together/Simplify prove the algebraic residuals are zero without selecting
   any heading chart or finite-difference step.
*)
raw = {x1[u], x2[u], x3[u]};
rawNorm = Sqrt[raw.raw];
unit = raw/rawNorm;
normFirst = unit.D[raw, u];
normSecond =
  (D[raw, u].D[raw, u]+raw.D[raw, {u, 2}]-normFirst^2)/rawNorm;
unitFirst = (D[raw, u]-unit normFirst)/rawNorm;
unitSecond =
  (D[raw, {u, 2}]-unit normSecond-2 unitFirst normFirst)/rawNorm;

(* Exact cross-90-degree roll fixture. For
     f = g {0,-Sin[angle[u]],Cos[angle[u]]}, e={1,0,0},
   projected heading produces Rx[angle[u]]. The antisymmetric part of
   Transpose[R].R'' gives angular acceleration because R'^T R' is symmetric.
   Expected body rates are {angle'[u],0,0} and {angle''[u],0,0}.
*)
rollRotation = {{1, 0, 0},
  {0, Cos[angle[u]], -Sin[angle[u]]},
  {0, Sin[angle[u]], Cos[angle[u]]}};
vee[m_] := {m[[3, 2]]-m[[2, 3]], m[[1, 3]]-m[[3, 1]],
  m[[2, 1]]-m[[1, 2]]}/2;

<|
  "DefaultRequestedTiltDegrees" -> 105,
  "DefaultHorizonStepsAt100Hz" -> 500,
  "FinalTargetExtension" -> "Keep the endpoint commanded for t>=T; no fixed hold interval",
  "MidpointPositionVelocityAcceleration" -> midpoint,
  "EndpointTimeDerivatives" -> Table[
    {D[timeLaw, {u, k}] /. u -> 0, D[timeLaw, {u, k}] /. u -> 1},
    {k, 1, 4}],
  "ArcLength" -> arcLength,
  "IntegratedSquaredSnap" -> snapEnergy,
  "ReducedSnapDerivative" -> Factor[D[reducedSnap, b]],
  "ExactBoundedBendOptimum" -> Last[optimalBend],
  "FullRangeMidpointTilt" -> tiltActual,
  "TiltDerivativeInAccelerationScale" -> tiltDerivative,
  "TiltDerivativeIdentityResidual" -> FullSimplify[
    tiltDerivative-tiltExpectedDerivative, Assumptions -> tiltAssumptions],
  "TiltDerivativeStrictlyPositive" -> FullSimplify[
    tiltExpectedDerivative > 0, Assumptions -> tiltAssumptions],
  "UprightEndpointTilt" -> FullSimplify[
    tiltActual /. theta -> 0, Assumptions -> {eta > 0, 0 < s <= 1}],
  "ExactHorizontalThrustTilt" -> FullSimplify[
    tiltActual /. {theta -> Pi/2, s -> 1, eta -> 6/5}],
  "TransverseForceSquareCompletion" -> transverseSquareCompletion,
  "SquareCompletionIdentityResidual" -> FullSimplify[
    transverseForceSquared-transverseSquareCompletion,
    Assumptions -> forceAssumptions],
  "TransverseForcePositiveLowerBound" -> transverseLowerBound,
  "TransverseLowerBoundStrictlyPositive" -> FullSimplify[
    transverseLowerBound > 0, Assumptions -> forceAssumptions],
  "ProjectedHeadingNormSquared" -> projectionNormSquared,
  "ProjectionFirstSecondDerivativeResiduals" -> {
    Simplify[D[projection, u]-projectionFirst],
    Simplify[D[projection, {u, 2}]-projectionSecond]},
  "NormalizationFirstSecondDerivativeResiduals" -> {
    Simplify[Together[D[unit, u]-unitFirst]],
    Simplify[Together[D[unit, {u, 2}]-unitSecond]]},
  "ExactRollBodyRateAndAcceleration" -> FullSimplify[{
    vee[Transpose[rollRotation].D[rollRotation, u]],
    vee[Transpose[rollRotation].D[rollRotation, {u, 2}]]}],
  "TimeDilationDerivativeScales" -> {s, s^(3/2), s^2},
  "GeometryScalingDerivativeScales" -> {s, s, s},
  "UncappedPathLengthRatio" -> N[
    (Sqrt[(7/10)^2+4 (19/50)^2]
      +(7/10)^2 ArcSinh[2 (19/50)/(7/10)]/(2 (19/50))) /
    (Sqrt[(9/20)^2+4 (1/5)^2]
      +(9/20)^2 ArcSinh[2 (1/5)/(9/20)]/(2 (1/5))), 12]
|>
