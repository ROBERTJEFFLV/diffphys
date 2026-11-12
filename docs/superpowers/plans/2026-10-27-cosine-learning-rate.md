# Cosine learning rate implementation plan

**Goal:** Use PyTorch CosineAnnealingLR to decrease Adam LR from 1e-3 to 1e-5 over 10000 successful updates, then hold the minimum.

**Architecture:** Keep the existing constant LR default. Add opt-in CLI flags, bind the schedule to checkpoints, and save/restore its state. Advance only after the existing Adam transaction and audit have succeeded; audit capsules continue to describe the actual Adam step at its recorded LR.

**Constraints:** Preserve the Actor, mission, loss, simulator, sampling and gradient clipping. Old source/config checkpoints remain ineligible for exact resume. The user selected a fresh random Actor and Adam, with no checkpoint import. No delegation. Keep 3072-scene training running until stopped.

**Review focus:** First-update LR; update versus evaluation counts; no rebound after T_max; exact resume including Adam/RNG/schedule; numerical or audit failure must not advance the schedule.

## Steps

- [x] Add failing deterministic integration checks in tests/test_lr_schedule.py for the curve, minimum hold, exact resume, rollback, audit replay and incompatible configuration rejection.
- [x] Add CLI validation in tools/train_response_control.py and minimal scheduler binding/state/step/log support in response_training.py.
- [x] Run the new checks and the CPU suite; review only this task's diff. Check that physics/Actor/loss sources are unchanged. Results: 15 scheduler checks; 688 passed and 25 CUDA skips in the CPU suite.
- [x] Add configs/gradient_repaired_cosine_flight.args and document the new schedule. Preserve the old run and OOM evidence. Launch a new run from scratch, then verify live CUDA updates, LR and GUI. Evidence: reports/cosine_lr_20261027/verification.json and scope_review.json.

The user has already authorized implementation and ongoing local training; execute in this thread without a separate approval stage.
