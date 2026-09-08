"""A/B dense-oracle validation of the reduced PETSc backend; no training."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from env_l2f import L2FParams, L2FSimulator
from full_space_shooting import BoundaryLayout, FullSpaceProblem
from petsc_kkt_solver import ReducedKKTLayout, TorchKKTContext, require_petsc, solve_petsc_minres
from response_policy import ResponseMotorPolicy, ResponsePolicyConfig
from response_shooting import make_problem
from response_task import TaskLossConfig


def artificial_problem(device="cpu"):
    generator = torch.Generator(device=device).manual_seed(71)
    def random(*shape):
        return torch.randn(*shape, generator=generator, device=device, dtype=torch.float64)
    initial = random(1, 13)
    initial[..., -1] = 0
    theta = random(60) * 0.1
    boundaries = random(2, 1, 13) * 0.1
    boundaries[..., -1] = torch.tensor([[1.], [2.]], device=device)
    a, b = random(12, 12) * 0.05, random(12, 60) * 0.05
    target = random(2, 1, 12)
    def segment(node, parameters):
        return torch.cat((node[..., :12] @ a.T + parameters @ b.T, node[..., -1:] + 1), -1)
    def task(starts, ends, parameters):
        return torch.cat(((ends[..., :12] - target).reshape(-1), parameters))
    fixed = torch.zeros_like(boundaries, dtype=torch.bool)
    fixed[..., -1] = True
    return FullSpaceProblem(initial, boundaries, theta, segment, BoundaryLayout(13),
                            task_residual=task, fixed_boundary_mask=fixed)


def tiny_response_problem(device="cpu"):
    torch.manual_seed(7)
    simulator = L2FSimulator(L2FParams())
    initial = simulator.reset(2, device=torch.device(device), dtype=torch.float64)
    initial = replace(initial,
        position=initial.position.new_tensor(((.3, -.2, .4), (-.4, .1, .2))),
        velocity=initial.velocity.new_tensor(((.1, .2, -.1), (-.2, .1, .2))),
        rotation=torch.eye(3, device=device, dtype=torch.float64).repeat(2, 1, 1),
        omega=initial.omega.new_tensor(((.2, -.1, .3), (-.1, .2, -.2))),
        motor=torch.zeros_like(initial.motor), previous_action=torch.zeros_like(initial.previous_action),
    )
    policy = ResponseMotorPolicy(ResponsePolicyConfig(memory_dim=2, hidden_dim=4)).to(
        device=device, dtype=torch.float64
    )
    problem, codec, _ = make_problem(policy, simulator, initial, TaskLossConfig(steady_steps=4),
                                     segment_steps=2, segments=2)
    # Nonzero dynamic defects test the constraint RHS and multiplier sign.
    perturbation = torch.linspace(-1., 1., problem.boundaries.numel(), device=device,
                                  dtype=torch.float64).reshape_as(problem.boundaries) * 1.0e-4
    perturbation[problem.fixed_boundary_mask] = 0
    problem = replace(problem, boundaries=problem.boundaries.detach() + perturbation)
    return problem, codec.characteristic_scales()


def dense_oracle(problem, damping):
    """Explicit Jacobians in original coordinates, independent of KKT matvec."""
    theta_count = problem.theta.numel()
    free = (torch.ones_like(problem.boundaries, dtype=torch.bool) if problem.fixed_boundary_mask is None
            else ~torch.broadcast_to(problem.fixed_boundary_mask, problem.boundaries.shape))
    base = torch.cat((problem.theta.detach().flatten(), problem.boundaries.detach()[free]))
    def unpack(value):
        boundary = problem.boundaries.detach().clone()
        boundary[free] = value[theta_count:]
        return value[:theta_count].reshape_as(problem.theta), boundary
    def task(value):
        theta, boundary = unpack(value)
        return problem.task_values(boundary, theta)
    def constraint(value):
        theta, boundary = unpack(value)
        return problem.defects(boundary, theta)[free]
    jr = torch.autograd.functional.jacobian(task, base, vectorize=True).detach()
    c = torch.autograd.functional.jacobian(constraint, base, vectorize=True).detach()
    h = jr.T @ jr + damping * torch.eye(base.numel(), device=base.device, dtype=base.dtype)
    matrix = torch.cat((torch.cat((h, c.T), 1),
                        torch.cat((c, c.new_zeros(c.shape[0], c.shape[0])), 1)), 0)
    rhs = -torch.cat((jr.T @ task(base).detach(), constraint(base).detach()))
    return matrix, rhs, torch.linalg.solve(matrix, rhs)


def check_problem(problem, *, damping=1.0, boundary_scale=None, label="A"):
    started = time.monotonic()
    scales = dict(parameter_scale=problem.theta.detach().abs().clamp_min(1.0e-2),
                  boundary_scale=boundary_scale, constraint_scale=boundary_scale)
    layout = ReducedKKTLayout.from_problem(problem, **scales)
    context = TorchKKTContext(problem, damping, layout)
    matrix, rhs, oracle = dense_oracle(problem, damping)
    direction = torch.linspace(-.8, 1., layout.size, device=rhs.device, dtype=rhs.dtype)
    product_error = float((context.raw_matvec(direction) - matrix @ direction).norm()
                          / (matrix @ direction).norm())
    result = solve_petsc_minres(problem, damping=damping, rtol=1.0e-10, atol=1.0e-14,
                                max_iterations=2000, **scales)
    solved = torch.cat((layout.pack_primal(result.dtheta, result.dboundary),
                        layout.pack_constraint(result.multipliers)))
    relative_error = float((solved - oracle).norm() / oracle.norm().clamp_min(1.0e-12))
    relative_residual = float((matrix @ solved - rhs).norm() / rhs.norm())
    output = {
        "stage": label, "device": str(problem.theta.device), "dtype": str(problem.theta.dtype),
        "n_primal": layout.n_primal, "n_constraint": layout.n_constraint,
        "removed_primal": problem.theta.numel() + problem.boundaries.numel() - layout.n_primal,
        "removed_constraints": problem.boundaries.numel() - layout.n_constraint,
        "matvec_relative_error": product_error,
        "solution_relative_error": relative_error, "dense_relative_residual": relative_residual,
        "true_relative_residual": result.relative_residual,
        "scaled_relative_residual": result.scaled_relative_residual,
        "petsc_reason": result.reason, "iterations": result.iterations,
        "certified": result.converged, "finite": result.finite,
        "preconditioner": result.preconditioner, "curvature_probes": result.curvature_probes,
        "seconds": time.monotonic() - started,
    }
    output["passed"] = bool(result.converged and product_error < 1.0e-9
                            and relative_residual < 1.0e-8 and relative_error < 1.0e-6)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    petsc = require_petsc(torch.zeros(1, device=args.device, dtype=torch.float64))
    results = []
    results.append(check_problem(artificial_problem(args.device), label="A"))
    print(json.dumps(results[-1]), flush=True)
    if results[-1]["passed"]:
        problem, scale = tiny_response_problem(args.device)
        results.append(check_problem(problem, damping=1., boundary_scale=scale, label="B"))
        print(json.dumps(results[-1]), flush=True)
    from response_training import source_hash
    report = {"petsc_version": petsc.Sys.getVersion(), "torch_version": str(torch.__version__),
              "source_sha256": source_hash(), "results": results,
              "passed": len(results) == 2 and all(row["passed"] for row in results),
              "training_executed": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
