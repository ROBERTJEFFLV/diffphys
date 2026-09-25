# Superseded experiment

The expensive per-scene gradient-clipping implementation was replaced on this
branch by [physical-group score balancing](physics_group_balance.md).
The original implementation, proof and measurements remain at commit
`17cfc62515cd2b500d0d00185c793bc68624af0c` in this repository's history.

Current training uses physics-based group membership, detached group RMS,
and one ordinary full-horizon backward. It does NOT promise per-scene gradient
bounds and does NOT accept the obsolete --contribution-* flags.
