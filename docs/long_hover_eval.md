# Long moving-target EVAL with surface-force pulses

This is an independent, inference-only environment. It does not alter training,
the Actor, its optimizer, the training loss, or the training source-hash contract.
`response_long_eval.py` provides `LongHoverEval`: construction initializes one
continuous episode, `step()` advances it and `run()` completes it.

## Run

```bash
python tools/evaluate_response_long.py @configs/response_long_eval.args \
    --checkpoint runs/pulsed_recovery/seed7/best.pt \
    --work-dir runs/long_hover_eval/seed20260927
```

The default evaluates every airframe in the checkpoint's two fixed EVAL banks,
usually 256. To select original pooled scene IDs, append `--scene-ids 196 254`.
Sampling always constructs the full bank and schedules before selecting IDs;
choosing one scene does not change its airframe, noise tape, targets or pulses.
The default device is CPU to avoid competing with ongoing GPU training. CUDA is
available through `--device cuda`. The output directory must be new or empty.

The current evaluator uses the upstream Gaussian-observation protocol and its
three-frame velocity history. Older bounded-noise checkpoints are not silently
converted: use their archived evaluator sources for exact reruns. Existing
trajectory exports and the switchable replay UI remain usable without loading
an Actor checkpoint. The v2 evaluator version distinguishes these sensor semantics
from the original v1 results; it does not change the arena/target/pulse protocol.

## Exact protocol

- 100 Hz, 6000 control transitions, 60 seconds maximum per airframe.
- Arena: x/y in [-2,2] m, z in [0,4] m. The geometric origin is at the floor
  center; there is no simulated floor contact or wall collision response.
- Start: (0,0,2) m, level attitude, zero velocity and angular velocity, motors at
  their analytic static hover state, matching previous command, zero GRU memory.
- Six independently sampled target positions: updates at t=0,10,20,30,40,50 s.
  Targets lie in x/y [-1.5,1.5], z [0.5,3.5] m. The same route is used for every
  airframe. This is a 3 x 3 x 3 m target cube concentric with the 4 m arena:
  both centers are (0,0,2) m. There is no target interpolation or controller warm-up.
- Only the Actor's position observation is translated by the current target.
  The physical state remains in world coordinates. GRU memory, motors and
  measurement history are continuous when a target changes.
- 60 impulses of **finite duration**: on [k,k+0.1) s for k=0,...,59; zero force
  on the remaining 0.9 s. Peak force magnitude is exactly 0.2*m*9.81 N. It is not
  an instantaneous velocity jump, nor a 20%-weight force for the entire second.
- Each pulse samples an independent isotropic world-space force direction and
  an independent body-fixed surface point. Direction and material point are held
  for that pulse; the point moves in world space as the aircraft rotates.
- Gaussian sensor noise and 10-30 ms velocity delay are retained from the
  compatible checkpoint; there is no additive command error in this protocol.
  All three past velocity samples remain continuous. Original constant external
  force, torque and TRAIN force-at-point pulse tapes are replaced, not added to
  the EVAL surface force. TRAIN's random-arm Gaussian pulses and this fixed-
  magnitude surface-force test have different distributions.
- First strict COM position exit from the arena marks failure. Nonfinite states
  also mark failure, separately. Failed rows are frozen; no reset or continued
  Actor evaluation occurs. Frozen padding is excluded from flight metrics.
  Failure at the final step is still failure. No speed or omega termination gate
  is introduced. A COM exactly on a boundary is inside; an arm or propeller
  extending outside does not trigger failure while the COM is inside. This is
  a COM containment test, not a full-body collision test.

## Surface geometry and wrench

There is no CAD shell mesh in the simulator. The explicitly declared approximation
is a closed body-frame rectangular envelope centered at the COM. Half dimensions:

```
hx = max(abs(rotor_positions.x)) + 0.12 * arm_length
hy = max(abs(rotor_positions.y)) + 0.12 * arm_length
hz = 0.10 * arm_length
```

These geometry ratios are configurable and are stored along with each airframe's
actual dimensions in the manifest. The box is a force-application proxy, not a
claim about the true fuselage/propeller surface. It does not change mass or inertia.

Select faces in proportion to area (4*hy*hz, 4*hx*hz, 4*hx*hy for the respective
pairs); sample the two remaining coordinates uniformly. Thus every equal area
patch has the same probability. Equal probability for all six faces would be
incorrect for this thin box. No point is placed at the COM.

For a body point r and world force F, at each RK4 stage:

```
world acceleration += F / mass
body torque += cross(r, R(q).transpose() @ F)
world application point = COM + R(q) @ r
```

The EVAL-only `PointForceSimulator` retains the motor curve, joint RK4,
normalization, gyro term and clamps of the production simulator. It evaluates
the point torque at all four intermediate attitudes. Holding a body torque
constant for an entire second would be physically incorrect for a world-fixed
force. Tests compare COM forcing bitwise with production, and point forcing with
an independent NumPy RK4 calculation.

## Reports and reproduction

The output contains an immutable `checkpoint.pt`, `manifest.json`, the exact
runtime sources under `source/`, complete `initial_state.pt` including immutable
noise tapes, and `schedule.pt` containing targets, surface points, face IDs and
world forces. All saved PyTorch inputs are dictionaries of tensors suitable for
`torch.load(..., weights_only=True)`. `trajectory.pt` retains state histories,
actions, valid masks, applied force/body-torque samples and memory norms.

`summary.json` reports survival, failure time/reason, tracking-error/velocity/omega
RMS, and all six target segments for each scene. An uncompleted target is never
reported as successful; later unattempted targets remain in the denominator.

An additional **reporting-only** hovered flag requires the complete ten-second
segment and its last two seconds entirely within: position error norm <=0.20 m,
speed norm <=0.20 m/s, omega norm <=0.50 rad/s. Thresholds are explicit config
values, not extra termination or training gates. Final error and tail RMS are
also supplied so this boolean does not hide the actual result.

Re-run using the frozen checkpoint, same argument file, seed and scene IDs, with
a new work directory. The manifest records source hashes, model hash, precision
and PyTorch version. CPU/CUDA or precision changes may change floating-point
trajectories; do not claim cross-device bitwise equivalence.

This evaluation does not authorize deployment and does not prove stability for
all possible force directions, surface points, airframes or target sequences.

## Switchable flight replay

The player reads the saved trajectories; it does not run the Actor or simulator.
It includes all arena exits and 13 completed flights selected at evenly spaced
mass ranks, rather than selecting the best tracking scores. Completion means
surviving the full episode, not meeting the strict hovering criterion.

```bash
python tools/play_response_long.py --run-dir runs/long_hover_eval/seed20260927
```

Rendering requires NumPy and Pygame. If Pygame is in a separate Python environment,
export with the training Python using the same command plus `--export-only`, then
run that other Python with `tools/play_response_long.py --replay
runs/long_hover_eval/seed20260927/playback/playlist.json`. Rendering the exported
file does not import PyTorch. The export records original scene IDs, selection
criteria and the source trajectory hash.

Click either column of aircraft to switch, or press Left/Right. Space pauses;
Up/Down seeks five seconds; F jumps to five seconds before failure (or the end
of a completed flight); R restarts; A toggles automatic cycling; +/- changes
speed. The timeline can be clicked or dragged. Drag the main view to rotate,
use the wheel to zoom, and C to reset the camera. S saves `playback/capture.png`.

The arena and target cube remain visible alongside a magnified aircraft inset.
The purple marker is the actual sampled material point; the arrow shows the
world force direction while a pulse is active. Arrow lengths are illustrative,
with the force magnitude in newtons displayed separately. Failed scenes end at
the first failure frame; frozen padding is never shown as continued flight.
