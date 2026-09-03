"""Diagnostics-only helpers for the legacy Q2 controller and simulator.

Nothing in this package is imported by the deployment forward path.  The
helpers intentionally consume simulator-only quantities only after the policy
has produced its deployable action.
"""

from .physics import (
    BranchCounterfactuals,
    BranchWrenchDecomposition,
    SteadyStateFeasibility,
    branch_counterfactual_actions,
    branch_wrench_decomposition,
    command_to_next_motor,
    motor_to_thrust,
    steady_state_feasibility,
    thrust_to_wrench,
)
from .time_weighting import dense_tracking_time_weights

__all__ = (
    "BranchCounterfactuals",
    "BranchWrenchDecomposition",
    "SteadyStateFeasibility",
    "branch_counterfactual_actions",
    "branch_wrench_decomposition",
    "command_to_next_motor",
    "dense_tracking_time_weights",
    "motor_to_thrust",
    "steady_state_feasibility",
    "thrust_to_wrench",
)
