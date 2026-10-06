# Read-only training monitor and saved-flight replay

This GUI extends the uploaded long-EVAL Pygame workflow, not the training loop.
`tools/monitor_response_training.py` is a separate process that reads existing
small log records and opens a saved short or long replay only on request.
The existing 3D projection, aircraft inset, surface-force arrow, timeline and
playback controls remain in that player.

## Start the monitor (training may already be running)

Use a graphical desktop Python. GUI dependencies are optional; they do not need
to be installed in the training environment:

```bash
python -m pip install -r requirements-gui.txt
python tools/monitor_response_training.py \
    --run-dir runs/pulsed_recovery_b2048/seed7
```

A missing directory or log is a waiting state. The monitor does not create the
training directory, start training or load a checkpoint. It can verify the
trainer/supervisor PID registered in `processes.json` against its Linux process
identity and run directory; it never guesses a PID from log age.
No trainer restart, new work directory or checkpoint migration is needed for
this GUI-only change. All files listed in `response_training.SOURCE_FILES` are
unchanged, as are training configs, Actor input and environment/noise versions.

The window shows task objective, position/velocity/omega RMS, the recorded
post-group/pre-global-clip gradient norm, update duration, recorded CUDA peak
allocation, motor saturation and fixed-EVAL first-exit fraction. TRAIN and EVAL
keep their actual update numbers: EVAL is not interpolated to look freshly
measured between scheduled evaluations. A long-EVAL replay is not substituted
for these fixed-EVAL metrics.

Space or **Pause reader** pauses GUI polling, NOT training. **Log y** / L toggles
positive-only logarithmic objective/gradient charts. Closing either GUI window
does not signal the trainer. The window is resizable. **Last saved status** is
explicitly a historical `summary.json` value; log age is not a liveness check.
Long EVAL/checkpoint writes can produce a quiet interval without a training crash.

## Training status and errors

The persistent status bar distinguishes running, stopped, interrupted, failed,
and unavailable status. On a new exit or failure, the details panel opens
automatically with the last completed update, TRAIN/EVAL metrics, recorded
exit reason/code, and original error text. Press **E** or **Status / error** to
open or close it; scroll with the mouse wheel or Page Up / Page Down. Escape
closes the panel first. Long messages wrap instead of disappearing behind an
ellipsis. Pausing metric curves does not disable exit monitoring.

The monitor reads `summary.json`, the launcher's `training_exit.json`,
`active_launch.json` and `processes.json`. A new launch's timestamp and update
progress prevent an old failure from being reported as a current crash.
Errors absent from the structured records can be read from the last 32 KiB of
`training.stdout.log`; an older traceback followed by successful updates is
not reported as a new error. If a registered process vanishes without an exit
record, the GUI says so explicitly. A signal exit is identified by its code;
it is not automatically called an OOM. Legacy logs without process metadata
are shown with unknown process state.

These files are optional launcher metadata, not a new training contract. The
GUI stays read-only and never changes checkpoints, restarts training, sends
signals to the trainer or silently diagnoses an unknown exit. All changes are
outside `response_training.SOURCE_FILES`; training's source binding remains
unchanged. Closing the GUI does not stop training.

## Open your existing exported replay

```bash
python tools/monitor_response_training.py \
    --run-dir runs/pulsed_recovery_b2048/seed7 \
    --replay runs/long_hover_eval/seed20260927/playback/playlist.json
```

Select an entry and click **Open replay** (or Enter). The uploaded 3D player opens
as a separate window, at a 20 FPS render cap; its simulated-time playback still
uses the saved 100 Hz samples. No policy or simulator is imported. A second
player is not launched while the first is open. The window shows the **recorded
checkpoint update**, not the current training update, and never automatically
switches model or flight while you inspect it.

For multiple exported evaluations, use `--replay-root runs/long_hover_eval` and
**Refresh list** / R. The scan checks that directory and its immediate children
for `playlist.json` or `playback/playlist.json`. It is bounded to 128 directory
entries and 64 replays and does not recursively traverse all checkpoints. Pass
`--replay` repeatedly for exact files outside that layout. Replay scanning is
also repeated every five seconds, so a newly saved short replay appears without
restarting the monitor. R forces an immediate refresh.

## Save the latest ordinary 5-second EVAL replay

An already-running trainer cannot load a new save hook. This optional sidecar
reads each committed checkpoint, recreates the same fixed EVAL pool, performs one
additional H500 forward pass, and saves all EVAL scenes with position, velocity,
orientation, angular velocity, actions, termination masks, and the exact
episode-constant and transient forces used by the physics step:

```bash
python tools/short_eval_replay.py \
    --run-dir runs/pulsed_recovery_b2048/seed7 --device cuda
```

It keeps only `short_eval/playlist.json` and one `trajectory_<update>.npz` in
the run directory. The watcher launches a short-lived CUDA child only after a
new EVAL/checkpoint pair; it releases that child's GPU context after export.
The 5-second player draws the constant force at the center of mass and each
active pulse at its body-frame application point. Labels show newtons and the
force-to-weight ratio; arrow length is a schematic display scale. Terminated
scenes show no later force application. Older short replays without force data
remain readable and say that force telemetry was not recorded.
This ordinary fixed EVAL uses the training protocol's Gaussian episode force
and random-arm pulses; it does not silently switch to the separate 60-second
EVAL's once-per-second, 20%-weight surface-pulse protocol.
It checks the checkpoint's full source hash and compares recomputed metrics to
the logged fixed EVAL at the same update. A source or metric mismatch fails the
export without changing training. `--once` exports the current committed Actor
one time, including when its update has not yet been written to `evaluation.jsonl`.

Pass the exported playlist to the monitor alongside any historical long EVAL:

```bash
python tools/monitor_response_training.py \
    --run-dir runs/pulsed_recovery_b2048/seed7 \
    --replay-root runs/long_hover_eval \
    --replay runs/pulsed_recovery_b2048/seed7/short_eval/playlist.json
```

On the RTX 4060 Ti with 256 EVAL scenes, H500 and float32, one actual export
at update 2700 took 3.7 seconds total, including 2.5 seconds to reconstruct
the EVAL trajectory. The compressed array was 7.7 MB plus a 0.14 MB playlist.
With a roughly 6.3-second training update and one EVAL per 50 updates, this is
about 1.2% additional wall time. These measurements are specific to that run.
At update 2800 the exporter reported 95 MB peak PyTorch CUDA allocation; the
short-lived export process exited afterward, leaving only the trainer on GPU.
Saving only the latest replay bounds disk use; keeping every EVAL would grow
storage by roughly 7.8 MB per checkpoint at the measured size.

The existing player now pages its two aircraft columns (13 per page), so many
arena exits do not overlap the telemetry. Export includes all arena exits and
**up to** the requested number of completed flights, sampled by mass rank. Zero
completed flights no longer prevents replaying available arena exits. Selection
does not include nonfinite-failure scenes; it preserves the original player's
arena-exit/completion scope. Counts, duration and force labels use saved metadata
instead of hard-coded `13 exits + 13 completed` / `60 s` labels.

## When only saved long-EVAL tensor files exist

Conversion is a separate, explicit command, with a Python that has PyTorch:

```bash
python tools/play_response_long.py \
    --run-dir runs/long_hover_eval/seed20260927 --export-only
```

This reads the finished EVAL's `trajectory.pt`, `schedule.pt`, `initial_state.pt`,
`manifest.json` and `summary.json` and exports NumPy arrays. It loads those tensors
on CPU, but still costs CPU time, disk I/O and RAM. The monitor does **not** start
this conversion, re-export on checkpoint updates, or invoke the long evaluator.
Export only after the EVAL output is complete; do not overwrite a playlist while
a new player is opening it. A running player has already loaded its own copy.

When only a checkpoint exists, there is no recorded long flight to play. Use
the existing `tools/evaluate_response_long.py` explicitly after training or on
a separate machine, then export it. **Even CPU EVAL competes for CPU/RAM on the
training host**; the short-EVAL sidecar above is a separate, explicitly enabled
exception with its measured cost.
The long-EVAL force/target/arena protocol remains exactly as uploaded and is
not the same distribution as training's random-arm Gaussian pulses.

## Resource isolation and limits

- No changes to TRAIN, the trainer's scheduled EVAL, rollout, loss, sampling,
  logging frequency or optimizer. There is no trainer-side queue/callback. The
  optional sidecar repeats one fixed EVAL and copies its recorded arrays to CPU.
- Default log polling is once per second (never faster). Each log reads at most
  256 KiB per poll plus small file-position anchors. Unchanged logs require only
  file metadata checks, not rereading their contents. A cold start reads only
  the tail; the UI explicitly says when an older prefix is omitted.
- Each curve retains at most 2000 reduced records, adjustable via `--max-points`
  in [100,10000]. Large nested diagnostic tables are not cached. Pixel envelopes
  retain minimum/maximum samples rather than averaging away gradient spikes.
- Partial final JSONL records wait for a newline. Malformed complete records are
  counted and ignored. Atomic resume rewrites, ordinary truncation and rollback
  clear stale future curve entries. No locks or writes are made to training logs.
- GUI event handling is capped at 10 Hz; charts repaint on polling/input rather
  than rendering an animation every frame. The renderer uses a software display,
  not a CUDA/OpenGL context. Launched replay processes have CUDA hidden and BLAS
  thread counts set to one. The dashboard does not poll `latest.pt`; the optional
  replay sidecar polls the checkpoint on CPU.

This removes direct training-path overhead; it does **not** assert zero total
machine overhead. A desktop compositor, CPU renderer and file reads still use
resources. For the strongest isolation, run the GUI on another computer using
copies/synced versions of the three small log/summary files and exported JSON/NPZ
pairs. A graphical desktop is required; a headless SSH shell alone is insufficient.
`--poll-seconds 2` reduces the read/redraw rate further. No network listener,
authentication layer or remote command execution is added.

## Verification

`tests/test_training_status.py` covers OOM text, normal/interrupted/signal exits,
resumed runs, missing/partial metadata, bounded traceback reads, PID identity
and a visible failure panel without changing run files.
`tests/test_training_monitor.py` exercises append/partial-line/rotation handling,
cache/read limits, spike retention, old-summary semantics, manual CPU-only replay
launch, limited scanning, and blocked Torch/trainer imports. SDL dummy-display
smoke tests render both windows without a GPU or Actor. CI installs Pygame so
these GUI tests cannot silently skip; smoke screenshots use synthetic fixture
data, not claimed flight results. `tests/test_short_eval_replay.py` verifies that
the exporter reproduces the normal rollout, keeps terminal padding frozen and
replaces the previous replay. Existing physical/gradient/resume tests remain.

Optional bounded headless check (requires Pygame):

```bash
SDL_VIDEODRIVER=dummy python tools/monitor_response_training.py \
    --run-dir /path/to/run --screenshot /tmp/monitor.png
```

CPU correctness and import isolation do not measure throughput on your GPU host.
