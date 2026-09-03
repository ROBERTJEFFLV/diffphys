# Archived shooting experiments

`multiple_shooting.py` and `tools/train_multiple_shooting_h1000.py` implement
the earlier persistent-boundary penalty experiment.  They remain at their
historical import paths so old checkpoints and reports stay reproducible, but
the structured-policy trainer must not import them as its optimizer.

`strict_multiple_shooting.py` is likewise retained as a compatibility path.
Its canonical description is now `checkpointed_exact_bptt.py`: segmented
recomputation with exact full-BPTT derivatives, not full-space shooting.

New work lives in `full_space_shooting.py` and uses trajectory-local boundary
variables plus trust-region steps.
