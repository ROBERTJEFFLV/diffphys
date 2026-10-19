(* Offline derivation evaluated with the Wolfram tool on 2026-10-19.
   The production planner runs in PyTorch and has no Wolfram dependency.
   l is half-span, b is the transverse bend magnitude, t is motion duration.
   Normalize the unchanged position_limit B to 1 for the shape optimization.
   This shape optimum is conditional on the chosen path family and clearances;
   individual-rotor and RK4 motor feasibility are checked separately in Python.
*)

ClearAll[u, q, l, b, t, s, theta, eta, timeLaw, spatialPath, path,
  midpoint, arcLength, snapEnergy, reducedSnap, optimalBend, tiltDerivative];

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

(* With acceleration scale s=(T_ideal/T)^2 and positive vertical thrust,
   tan(theta_actual)=s eta sin(theta)/(1+s(eta cos(theta)-1)).
   The derivative is positive, so the shortest feasible tested duration
   also gives the largest midpoint tilt on that tested fixed-geometry grid.
*)
tiltDerivative = FullSimplify[
  D[s eta Sin[theta]/(1+s (eta Cos[theta]-1)), s]];

<|
  "MidpointPositionVelocityAcceleration" -> midpoint,
  "EndpointTimeDerivatives" -> Table[
    {D[timeLaw, {u, k}] /. u -> 0, D[timeLaw, {u, k}] /. u -> 1},
    {k, 1, 4}],
  "ArcLength" -> arcLength,
  "IntegratedSquaredSnap" -> snapEnergy,
  "ReducedSnapDerivative" -> Factor[D[reducedSnap, b]],
  "ExactBoundedBendOptimum" -> Last[optimalBend],
  "TiltDerivativeInAccelerationScale" -> tiltDerivative,
  "TimeDilationDerivativeScales" -> {s, s^(3/2), s^2},
  "GeometryScalingDerivativeScales" -> {s, s, s},
  "UncappedPathLengthRatio" -> N[
    (Sqrt[(7/10)^2+4 (19/50)^2]
      +(7/10)^2 ArcSinh[2 (19/50)/(7/10)]/(2 (19/50))) /
    (Sqrt[(9/20)^2+4 (1/5)^2]
      +(9/20)^2 ArcSinh[2 (1/5)/(9/20)]/(2 (1/5))), 12]
|>
