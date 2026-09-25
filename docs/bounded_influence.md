# Historical per-scene bounded-influence experiment

The expensive per-scene VJP implementation is archived at `17cfc625`.
Cost-RMS normalization at `3ea9fe8` is also superseded.
This branch now uses **physical-group parameter-gradient normalization**:
keep the physical groups (at least 32 initial scenes each), compute each
GROUP's gradient, normalize its norm to a common median, then average.
There is no per-scene norm calculation and no cost normalization.
See [the current algorithm, limits and tests](physics_group_balance.md).
