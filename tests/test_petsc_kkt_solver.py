from __future__ import annotations

from dataclasses import replace
import json

import pytest
import torch

import full_space_shooting as shooting
import petsc_kkt_solver as backend
from tools.validate_petsc_kkt import artificial_problem, dense_oracle


def test_reduced_kkt_matches_dense_and_is_symmetric_with_nonuniform_scales():
    problem = artificial_problem()
    scale = torch.logspace(-1, 1, 13, dtype=torch.float64)
    layout = backend.ReducedKKTLayout.from_problem(problem, boundary_scale=scale, constraint_scale=scale)
    context = backend.TorchKKTContext(problem, 0.3, layout)
    matrix, rhs, _ = dense_oracle(problem, 0.3)
    assert layout.size == 108
    assert layout.n_constraint == 24
    u = torch.linspace(-1, 1, layout.size, dtype=torch.float64)
    v = torch.cos(u * 3)
    torch.testing.assert_close(context.raw_rhs, rhs)
    torch.testing.assert_close(context.matvec(u), context.scaling * (matrix @ (context.scaling * u)))
    torch.testing.assert_close(u @ context.matvec(v), v @ context.matvec(u))
    theta, boundary, multiplier = layout.recover(u)
    assert bool((boundary[problem.fixed_boundary_mask] == 0).all())
    assert bool((multiplier[problem.fixed_boundary_mask] == 0).all())
    torch.testing.assert_close(layout.pack_primal(theta, boundary), u[:layout.n_primal] * layout.primal_scale)
    torch.testing.assert_close(layout.pack_constraint(multiplier), u[layout.n_primal:] / layout.constraint_scale)


def test_spd_preconditioner_is_fixed_and_rejects_invalid_diagonal():
    diagonal = torch.tensor([1., 2., 3.])
    pc = backend.FixedSPDPreconditioner(diagonal)
    diagonal.zero_()
    assert bool((pc.inverse_diag > 0).all())
    for value in [0., -1., float("nan"), float("inf")]:
        with pytest.raises(ValueError, match="strictly positive"):
            backend.FixedSPDPreconditioner(torch.tensor([1., value]))


def test_curvature_probes_are_deterministic_spd_and_preserve_sampling_rng():
    problem = artificial_problem()
    layout = backend.ReducedKKTLayout.from_problem(problem)
    context = backend.TorchKKTContext(problem, 1., layout)
    state = torch.get_rng_state().clone()
    first = backend.curvature_diagonal(context, 8)
    second = backend.curvature_diagonal(context, 8)
    assert torch.equal(first, second)
    assert torch.equal(state, torch.get_rng_state())
    assert bool(torch.isfinite(first).all() and (first > 0).all())


def test_uncertified_direction_cannot_reach_trust_region_or_legacy_cg(monkeypatch):
    problem = artificial_problem()
    visits = []
    def failed(*args, **kwargs):
        return backend.PetscKKTResult(
            torch.ones_like(problem.theta), torch.ones_like(problem.boundaries),
            torch.zeros_like(problem.boundaries), False, 3, 1., 1., 1., 2, True,
        )
    def no_cg(*args, **kwargs):
        raise AssertionError("PETSc path invoked legacy CG")
    monkeypatch.setattr(backend, "solve_petsc_minres", failed)
    monkeypatch.setattr(shooting, "_cg_solve", no_cg)
    def actions(theta, boundary):
        visits.append(theta.detach().clone())
        return theta[:4]
    result = shooting.solve_joint_sqp_step(problem, action_evaluator=actions)
    assert not result.accepted
    assert not result.linear_solver_converged
    assert result.linear_solver_reason == 2  # Positive PETSc flag alone is insufficient.
    torch.testing.assert_close(result.theta, problem.theta)
    assert all(torch.equal(value, problem.theta) for value in visits)


def test_dtype_mismatch_fails_before_dlpack_export():
    pytest.importorskip("petsc4py")
    from petsc4py import PETSc
    other = torch.float32 if PETSc.ScalarType.__name__ == "float64" else torch.float64
    with pytest.raises(ValueError, match="ScalarType"):
        backend.require_petsc(torch.zeros(1, dtype=other))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_dlpack_storage_alias_and_mat_work_vector_type(device):
    pytest.importorskip("petsc4py")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    storage = torch.arange(8, device=device, dtype=torch.float64)
    petsc = backend.require_petsc(storage)
    backend._synchronize(storage.device)
    vec = matrix = work = None
    try:
        vec = petsc.Vec().createWithDLPack(storage, comm=petsc.COMM_SELF)
        alias = torch.utils.dlpack.from_dlpack(vec.toDLPack(mode="rw"))
        assert alias.data_ptr() == storage.data_ptr()
        vec.scale(2.)
        backend._synchronize(storage.device)
        torch.testing.assert_close(storage, torch.arange(8, device=device, dtype=torch.float64) * 2)
        alias.add_(1.)
        backend._synchronize(storage.device)
        assert vec.sum() == 64.
        matrix = petsc.Mat().createPython([8, 8], context=object(), comm=petsc.COMM_SELF)
        matrix.setVecType("cuda" if device == "cuda" else "seq")
        matrix.setUp()
        work = matrix.createVecRight()
        work.set(0.)
        tensor = torch.utils.dlpack.from_dlpack(work.toDLPack(mode="r"))
        assert tensor.device == storage.device
        del tensor, alias
    finally:
        for handle in (work, matrix, vec):
            if handle is not None:
                handle.destroy()


def test_fixed_constraint_with_nonzero_residual_is_not_silently_dropped():
    pytest.importorskip("petsc4py")
    problem = artificial_problem()
    boundaries = problem.boundaries.clone()
    boundaries[..., -1] += 0.25
    with pytest.raises(ValueError, match="nonzero residual"):
        backend.solve_petsc_minres(replace(problem, boundaries=boundaries))


def test_certified_petsc_sqp_step_uses_no_cg_and_keeps_fixed_clock(monkeypatch):
    pytest.importorskip("petsc4py")
    problem = artificial_problem()
    def forbidden(*args, **kwargs):
        raise AssertionError("PETSc path called legacy CG, including multiplier refit")
    monkeypatch.setattr(shooting, "_cg_solve", forbidden)
    step = shooting.solve_joint_sqp_step(problem, damping=1., kkt_rtol=1.0e-9,
                                         kkt_atol=1.0e-14, kkt_max_iterations=1000)
    assert step.linear_solver_converged
    assert step.accepted
    assert step.merit_after < step.merit_before
    assert step.stationarity_multiplier_kind == "raw-full-step"
    torch.testing.assert_close(step.boundaries[problem.fixed_boundary_mask],
                                problem.boundaries[problem.fixed_boundary_mask], rtol=0, atol=0)


def test_nonfinite_solve_rejects_response_proposal_and_preserves_json_evidence(monkeypatch):
    pytest.importorskip("petsc4py")
    from env_l2f import L2FParams, L2FSimulator
    from response_policy import ResponseMotorPolicy, ResponsePolicyConfig
    from response_shooting import task_shooting_step
    from response_task import TaskLossConfig
    policy = ResponseMotorPolicy(ResponsePolicyConfig(memory_dim=2, hidden_dim=4)).double()
    simulator = L2FSimulator(L2FParams())
    initial = simulator.reset(2, device=torch.device("cpu"), dtype=torch.float64)
    original = {name: value.clone() for name, value in policy.state_dict().items()}
    def failed(problem, **kwargs):
        return backend.PetscKKTResult(
            torch.full_like(problem.theta, float("nan")),
            torch.full_like(problem.boundaries, float("nan")),
            torch.full_like(problem.boundaries, float("nan")),
            False, 4, float("nan"), float("nan"), float("nan"), -9, False,
        )
    monkeypatch.setattr(backend, "solve_petsc_minres", failed)
    evidence, _ = task_shooting_step(policy, simulator, initial, initial,
                                     TaskLossConfig(prediction_weight=0),
                                     segment_steps=2, segments=2, debug_solver=True)
    assert not evidence["accepted"]
    assert evidence["linear_solver_relative_residual"] is None
    assert evidence["solver_debug"]["linear_solver_breakdown"]
    json.dumps(evidence, allow_nan=False)
    assert all(torch.equal(original[name], value) for name, value in policy.state_dict().items())
