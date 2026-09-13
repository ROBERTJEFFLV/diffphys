# DiffPhys response motor control

A response encoder, GRU memory and motor controller learn from differentiable
quadrotor flight. The deployable Actor consumes a 25-element observation and
outputs four motor commands. Simulator dynamics and motor truth do not enter
its observation.

## Production files

| File | Responsibility |
| --- | --- |
| `env_l2f.py` | PyTorch rigid-body physics, motor response and scenario dynamics |
| `response_policy.py` | Response encoder, GRU and motor controller |
| `response_task.py` | Initial states, continuous rollout, task/CVaR loss and metrics |
| `response_adjoints.py` | Exact full-flight gradients with retained graph or window recomputation |
| `response_training.py` | Adam, fixed development evaluation, checkpoints and resume |
| `response_execution.py` | Exit status classification |
| `tools/train_response_control.py` | Train, profile and evaluate CLI |
| `configs/response_phase1_single_airframe.args` | Nominal-airframe H500/W50 configuration |

The default `--backprop-mode full` flies H500 once with its entire computation
graph, selects pooled CVaR weights once and differentiates the same objective.
H50 chunks preserve absolute time and continuous physical/recurrent state; they
do not detach the graph or replay the flight. `--backprop-mode windowed` retains
the exact reverse-window implementation for limited memory and numerical
comparisons. Both modes apply finite checks, optional AGC, global clipping and
one persistent Adam update. All implicit-solve statuses from forward and backward
are checked together before Adam; physical integration is unchanged.

The current defaults use seed 7, four pooled banks of 128 states (512 concurrent
TRAIN trajectories), H500/W50,
`gradient-scale=0.1`, Adam `lr=3e-4`, clip 10 and AGC disabled. Nominal dynamics
and zero external force are fixed; position, velocity, attitude and omega vary.
The CLI also supports `--scenario-mode physical-fit`, retaining outer 4x4
thrust-to-weight / roll-authority sampling and its existing disturbances.
Fixed EVAL pools two banks of 128 states (256 trajectories). Periodic EVAL and
checkpoint saving both run every 50 updates. Initial and final/interrupted
checkpoints still protect the run independently of that periodic schedule.

## Run

Python, PyTorch and NumPy are required. CUDA training uses the installed PyTorch
CUDA runtime; no project native extension is required. Run training only with
an explicit budget:

```bash
python3 tools/train_response_control.py $(cat configs/response_phase1_single_airframe.args)
```

Use a fresh work directory for a new experiment. `--mode profile` performs at
most one update of the same trainer. `--resume PATH` requires matching source
and configuration. `--mode evaluate --checkpoint PATH` evaluates fixed development
states. See [the task, state and checkpoint contract](docs/response_control_v1.md)
for commands and explicit migration requirements.

Runtime recomputation consistency in windowed mode, finite checks, failure recovery, atomic saving,
RNG restoration and fixed development evaluation are part of production.
Development scores select best checkpoints; finite loss increases do not veto
individual updates. Reports, model files and local audit evidence are not code
and must not be deleted or published as part of source cleanup.

## Source history and limits

This project derives from the DiffPhys L2F / RLtools CUDA simulator work and the
`diffphysDrone` recurrent-control flow. The original reference source and its
notices, older baselines, optional CUDA extensions and standalone test
and diagnostic tools remain available at commit
`76b3a02857e122fbfdef5ece5d0ae7dbf98a870b`. Source/provenance notices and the local
`物理配置/` calibration history are retained. No new license is asserted here.

Removed APIs and historical pickled module objects require their matching old
source. Source cleanup invalidates strict checkpoint source bindings; do not
edit hashes to bypass this. Explicitly audited migration must retain named Adam
state, RNG and the update index to qualify as optimizer continuation.

Exact gradients can still explode over long horizons. Numerical equivalence
and reproducible saving do not demonstrate reliable hover, cross-airframe
performance or real-flight safety. This code does not authorize deployment.
