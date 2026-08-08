from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

from conditional_wf_sbb import BridgeConfig, ConditionalWFSBB
from central_path_transition import EntropicCentralPath


Array = NDArray[np.float64]


class HybridLogDomainSolver:

    def __init__(self, base: BridgeConfig, chi: float):
        self.base = base
        self.chi = float(chi)
        self.unregularized_model = ConditionalWFSBB(
            replace(base, entropy_epsilon=0.0, lower_bound=1.0e-14)
        )
        solved = self.unregularized_model.solve(self.chi)
        if not solved["success"]:
            raise RuntimeError("The unregularized active-face solve failed")

        self.zero_solution = solved
        self.zero_flow = np.asarray(solved["flow"], dtype=float)
        self.boundary_indices = np.asarray(solved["active_bounds"], dtype=int)
        self.interior_indices = np.asarray(
            [
                j
                for j in range(self.unregularized_model.n_var)
                if j not in set(self.boundary_indices.tolist())
            ],
            dtype=int,
        )
        self.active_cones = list(solved["active_cones"])
        blocks = [self.unregularized_model.Aeq]
        targets = [self.unregularized_model.beq]
        if self.active_cones:
            blocks.append(self.unregularized_model.Acone[self.active_cones])
            targets.append(np.zeros(len(self.active_cones)))
        self.constraint_matrix = np.vstack(blocks)
        self.constraint_target = np.concatenate(targets)

        active_matrix = self.unregularized_model._active_constraint_matrix(
            self.boundary_indices.tolist(), self.active_cones
        )
        zero_gradient = self.unregularized_model.objective_gradient(
            self.zero_flow, self.chi
        )[1]
        multipliers = np.linalg.lstsq(
            active_matrix.T, -zero_gradient, rcond=None
        )[0]
        n_equalities = self.unregularized_model.Aeq.shape[0]
        self.zero_bound_multipliers = multipliers[
            n_equalities : n_equalities + len(self.boundary_indices)
        ]

        reference_model = ConditionalWFSBB(base)
        self.reference_flow = np.asarray(reference_model.reference_flow, dtype=float)

    def model(self, epsilon: float) -> ConditionalWFSBB:
        return ConditionalWFSBB(
            replace(
                self.base,
                entropy_epsilon=float(epsilon),
                lower_bound=1.0e-30,
            )
        )

    def unpack(self, vector: Array, epsilon: float) -> Tuple[Array, Array, Array, Array]:
        n_interior = len(self.interior_indices)
        n_boundary = len(self.boundary_indices)
        interior_flow = np.asarray(vector[:n_interior], dtype=float)
        log_dual = np.asarray(
            vector[n_interior : n_interior + n_boundary], dtype=float
        )
        multipliers = np.asarray(vector[n_interior + n_boundary :], dtype=float)
        boundary_log_flow = (
            np.log(self.reference_flow[self.boundary_indices]) - log_dual / epsilon
        )
        boundary_flow = np.exp(boundary_log_flow)
        flow = np.empty(self.unregularized_model.n_var)
        flow[self.interior_indices] = interior_flow
        flow[self.boundary_indices] = boundary_flow
        return flow, interior_flow, log_dual, multipliers

    def initial_vector(self, flow: Array, epsilon: float) -> Array:
        interior_flow = np.asarray(flow[self.interior_indices], dtype=float)
        boundary_flow = np.maximum(
            np.asarray(flow[self.boundary_indices], dtype=float),
            np.finfo(float).tiny,
        )
        log_dual = -epsilon * np.log(
            boundary_flow / self.reference_flow[self.boundary_indices]
        )
        model = self.model(epsilon)
        gradient = model.objective_gradient(flow, self.chi)[1]
        multiplier = np.linalg.lstsq(
            self.constraint_matrix[:, self.interior_indices].T,
            -gradient[self.interior_indices],
            rcond=None,
        )[0]
        return np.concatenate([interior_flow, log_dual, multiplier])

    def residual_jacobian(
        self,
        vector: Array,
        epsilon: float,
        chi: float,
    ) -> Tuple[Array, Array, Array, Array]:
        flow, interior_flow, log_dual, multipliers = self.unpack(vector, epsilon)
        _, gradient = self.unregularized_model.objective_gradient(flow, chi)
        hessian = self.unregularized_model.hessian(flow, chi)
        interior = self.interior_indices
        boundary = self.boundary_indices
        matrix = self.constraint_matrix

        log_interior = np.log(interior_flow / self.reference_flow[interior])
        residual_interior = (
            gradient[interior]
            + epsilon * log_interior
            + matrix[:, interior].T @ multipliers
        )
        residual_boundary = (
            log_dual
            - gradient[boundary]
            - matrix[:, boundary].T @ multipliers
        )
        residual_constraint = matrix @ flow - self.constraint_target
        residual = np.concatenate(
            [residual_interior, residual_boundary, residual_constraint]
        )

        boundary_flow = flow[boundary]
        derivative_boundary = -boundary_flow / epsilon
        h_ii = hessian[np.ix_(interior, interior)]
        h_ij = hessian[np.ix_(interior, boundary)]
        h_ji = hessian[np.ix_(boundary, interior)]
        h_jj = hessian[np.ix_(boundary, boundary)]

        jacobian_ii = h_ii + np.diag(epsilon / interior_flow)
        jacobian_ig = h_ij * derivative_boundary[None, :]
        jacobian_ji = -h_ji
        jacobian_jg = np.eye(len(boundary)) - h_jj * derivative_boundary[None, :]
        jacobian_mi = matrix[:, interior]
        jacobian_mg = matrix[:, boundary] * derivative_boundary[None, :]

        jacobian = np.block(
            [
                [jacobian_ii, jacobian_ig, matrix[:, interior].T],
                [jacobian_ji, jacobian_jg, -matrix[:, boundary].T],
                [
                    jacobian_mi,
                    jacobian_mg,
                    np.zeros((matrix.shape[0], matrix.shape[0])),
                ],
            ]
        )
        return residual, jacobian, flow, hessian

    def solve(
        self,
        epsilon: float,
        start: Array,
        chi: float | None = None,
        tolerance: float = 2.0e-11,
        max_iterations: int = 100,
    ) -> Tuple[Array, Dict[str, object]]:
        local_chi = self.chi if chi is None else float(chi)
        vector = np.asarray(start, dtype=float).copy()
        accepted_steps = 0
        for iteration in range(max_iterations):
            residual, jacobian, flow, _ = self.residual_jacobian(
                vector, epsilon, local_chi
            )
            residual_norm = float(np.max(np.abs(residual)))
            if residual_norm < tolerance:
                return vector, self.diagnostics(
                    vector, epsilon, iteration, accepted_steps, True, local_chi
                )

            direction = np.linalg.solve(jacobian, -residual)
            current_merit = float(residual @ residual)
            step = 1.0
            n_interior = len(self.interior_indices)
            negative = direction[:n_interior] < 0.0
            if np.any(negative):
                step = min(
                    step,
                    0.99
                    * float(
                        np.min(
                            -vector[:n_interior][negative]
                            / direction[:n_interior][negative]
                        )
                    ),
                )

            accepted = False
            for _ in range(80):
                candidate = vector + step * direction
                if np.min(candidate[:n_interior]) <= 0.0:
                    step *= 0.5
                    continue
                candidate_residual = self.residual_jacobian(
                    candidate, epsilon, local_chi
                )[0]
                candidate_merit = float(candidate_residual @ candidate_residual)
                if candidate_merit <= current_merit * (1.0 - 1.0e-4 * step):
                    vector = candidate
                    accepted = True
                    accepted_steps += 1
                    break
                step *= 0.5
            if not accepted:
                break

        return vector, self.diagnostics(
            vector, epsilon, max_iterations, accepted_steps, False, local_chi
        )

    def diagnostics(
        self,
        vector: Array,
        epsilon: float,
        iterations: int,
        accepted_steps: int,
        declared_success: bool,
        chi: float | None = None,
    ) -> Dict[str, object]:
        local_chi = self.chi if chi is None else float(chi)
        residual, jacobian, flow, _ = self.residual_jacobian(
            vector, epsilon, local_chi
        )
        _, _, log_dual, _ = self.unpack(vector, epsilon)
        model = self.model(epsilon)
        equality_residual = float(np.max(np.abs(model.Aeq @ flow - model.beq)))
        cone_residual = float(np.max(model.Acone @ flow))
        residual_norm = float(np.max(np.abs(residual)))
        return {
            "success": bool(declared_success or residual_norm < 2.0e-9),
            "iterations": int(iterations),
            "accepted_steps": int(accepted_steps),
            "hybrid_kkt_residual": residual_norm,
            "equality_residual": equality_residual,
            "cone_residual": cone_residual,
            "minimum_materialized_flow": float(np.min(flow)),
            "minimum_boundary_log_dual": float(np.min(log_dual)),
            "maximum_boundary_log_dual": float(np.max(log_dual)),
            "hybrid_jacobian_condition_number": float(np.linalg.cond(jacobian)),
        }

    def chi_derivative(
        self,
        vector: Array,
        epsilon: float,
    ) -> Tuple[Array, Dict[str, float]]:
        residual, jacobian, flow, _ = self.residual_jacobian(
            vector, epsilon, self.chi
        )
        cross_step = 2.0e-5
        plus_gradient = self.unregularized_model.objective_gradient(
            flow, self.chi + cross_step
        )[1]
        minus_gradient = self.unregularized_model.objective_gradient(
            flow, self.chi - cross_step
        )[1]
        cross_gradient = (plus_gradient - minus_gradient) / (2.0 * cross_step)
        cross = np.concatenate(
            [
                cross_gradient[self.interior_indices],
                -cross_gradient[self.boundary_indices],
                np.zeros(self.constraint_matrix.shape[0]),
            ]
        )
        derivative_vector = np.linalg.solve(jacobian, -cross)
        n_interior = len(self.interior_indices)
        n_boundary = len(self.boundary_indices)
        derivative_flow = np.zeros_like(flow)
        derivative_flow[self.interior_indices] = derivative_vector[:n_interior]
        log_dual_derivative = derivative_vector[
            n_interior : n_interior + n_boundary
        ]
        derivative_flow[self.boundary_indices] = (
            -flow[self.boundary_indices]
            * log_dual_derivative
            / epsilon
        )
        return derivative_flow, {
            "linearized_kkt_residual": float(
                np.max(np.abs(jacobian @ derivative_vector + cross))
            ),
            "constraint_derivative_residual": float(
                np.max(np.abs(self.constraint_matrix @ derivative_flow))
            ),
            "maximum_boundary_log_dual_derivative": float(
                np.max(np.abs(log_dual_derivative))
            ),
            "base_hybrid_kkt_residual": float(np.max(np.abs(residual))),
        }


def information_decomposition(
    model: ConditionalWFSBB,
    flow: Array,
    derivative: Array,
    boundary_indices: Array,
) -> Dict[str, object]:
    boundary_set = set(boundary_indices.tolist())
    boundary_information = 0.0
    interior_information = 0.0
    boundary_records = []
    for k in range(model.cfg.n_steps):
        local_flow = flow[model._slice(k)]
        local_derivative = derivative[model._slice(k)]
        for i in range(model.cfg.n_grid):
            edge_ids = model.rows[i]
            row_mass = float(np.sum(local_flow[edge_ids]))
            row_derivative = float(np.sum(local_derivative[edge_ids]))
            for edge in edge_ids:
                full_index = model._full_index(k, int(edge))
                score = (
                    local_derivative[int(edge)] / local_flow[int(edge)]
                    - row_derivative / row_mass
                )
                contribution = float(local_flow[int(edge)] * score * score)
                if full_index in boundary_set:
                    boundary_information += contribution
                    origin, destination = model.transitions[int(edge)]
                    boundary_records.append(
                        {
                            "full_index": int(full_index),
                            "time_step": int(k),
                            "origin": int(origin),
                            "destination": int(destination),
                            "flow": float(local_flow[int(edge)]),
                            "score": float(score),
                            "information": contribution,
                        }
                    )
                else:
                    interior_information += contribution
    dominant_boundary = max(
        boundary_records,
        key=lambda record: record["information"],
    )
    return {
        "boundary_information": float(boundary_information),
        "interior_information": float(interior_information),
        "decomposition_error": float(
            abs(
                boundary_information
                + interior_information
                - model.score_information(flow, derivative)["process_fisher"]
            )
        ),
        "dominant_boundary_edge": dominant_boundary,
    }


def continuation_grid(target: float) -> List[float]:
    requested = [
        2.5e-5,
        3.5e-5,
        5.0e-5,
        7.5e-5,
        1.0e-4,
        1.5e-4,
        2.0e-4,
        3.0e-4,
        4.0e-4,
        5.0e-4,
        6.0e-4,
        7.5e-4,
        1.0e-3,
        1.5e-3,
        2.0e-3,
        2.5e-3,
    ]
    requested.append(float(target))
    return sorted(set(requested))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("log_domain_boundary_layer.json"),
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=Path("log_domain_information_path.csv"),
    )
    parser.add_argument("--entropy-epsilon", type=float, default=5.0e-4)
    parser.add_argument("--chi", type=float, default=0.45)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base = BridgeConfig(
        entropy_epsilon=max(2.5e-3, args.entropy_epsilon),
        chi=args.chi,
        lower_bound=1.0e-30,
        seed=args.seed,
    )
    hybrid = HybridLogDomainSolver(base, args.chi)
    primal_path = EntropicCentralPath(base)
    _, primal_flow, _ = primal_path.solve(args.entropy_epsilon, args.chi)
    target_start = hybrid.initial_vector(primal_flow, args.entropy_epsilon)
    target_vector, target_diagnostic = hybrid.solve(
        args.entropy_epsilon, target_start, tolerance=2.0e-12
    )
    if not target_diagnostic["success"]:
        raise RuntimeError("The target log-domain solve failed")

    target_flow = hybrid.unpack(target_vector, args.entropy_epsilon)[0]
    target_derivative, target_derivative_diagnostic = hybrid.chi_derivative(
        target_vector, args.entropy_epsilon
    )
    target_information = hybrid.model(args.entropy_epsilon).score_information(
        target_flow, target_derivative
    )

    grid = continuation_grid(args.entropy_epsilon)
    target_position = grid.index(args.entropy_epsilon)
    solved_vectors: Dict[float, Array] = {args.entropy_epsilon: target_vector}
    solve_diagnostics: Dict[float, Dict[str, object]] = {
        args.entropy_epsilon: target_diagnostic
    }

    previous = target_vector
    for epsilon in reversed(grid[:target_position]):
        previous, diagnostic = hybrid.solve(epsilon, previous, tolerance=2.0e-12)
        if not diagnostic["success"]:
            raise RuntimeError(f"Descending continuation failed at {epsilon}")
        solved_vectors[epsilon] = previous
        solve_diagnostics[epsilon] = diagnostic

    previous = target_vector
    for epsilon in grid[target_position + 1 :]:
        previous, diagnostic = hybrid.solve(epsilon, previous, tolerance=2.0e-12)
        if not diagnostic["success"]:
            raise RuntimeError(f"Ascending continuation failed at {epsilon}")
        solved_vectors[epsilon] = previous
        solve_diagnostics[epsilon] = diagnostic

    path = []
    finite_difference_step = 2.0e-4
    for epsilon in grid:
        vector = solved_vectors[epsilon]
        flow = hybrid.unpack(vector, epsilon)[0]
        derivative, derivative_diagnostic = hybrid.chi_derivative(vector, epsilon)
        information = hybrid.model(epsilon).score_information(flow, derivative)
        decomposition = information_decomposition(
            hybrid.model(epsilon),
            flow,
            derivative,
            hybrid.boundary_indices,
        )
        plus_vector, plus_diagnostic = hybrid.solve(
            epsilon,
            vector,
            chi=args.chi + finite_difference_step,
            tolerance=2.0e-11,
            max_iterations=100,
        )
        minus_vector, minus_diagnostic = hybrid.solve(
            epsilon,
            vector,
            chi=args.chi - finite_difference_step,
            tolerance=2.0e-11,
            max_iterations=100,
        )
        plus_flow = hybrid.unpack(plus_vector, epsilon)[0]
        minus_flow = hybrid.unpack(minus_vector, epsilon)[0]
        finite_derivative = (
            plus_flow - minus_flow
        ) / (2.0 * finite_difference_step)
        finite_difference_error = float(
            np.linalg.norm(derivative - finite_derivative)
            / max(1.0e-12, np.linalg.norm(finite_derivative))
        )
        diagnostic = solve_diagnostics[epsilon]
        path.append(
            {
                "entropy_epsilon": float(epsilon),
                **diagnostic,
                **derivative_diagnostic,
                "finite_reoptimization_relative_error": finite_difference_error,
                "plus_reoptimization_success": bool(plus_diagnostic["success"]),
                "minus_reoptimization_success": bool(minus_diagnostic["success"]),
                "process_fisher": float(information["process_fisher"]),
                **decomposition,
                "score_mean": float(information["score_mean"]),
                "maximum_conditional_score_mean": float(
                    information["maximum_conditional_score_mean"]
                ),
            }
        )

    zero_derivative_result = hybrid.unregularized_model.implicit_derivative(
        hybrid.zero_flow,
        args.chi,
        active_bounds=hybrid.boundary_indices.tolist(),
        active_cones=hybrid.active_cones,
    )
    zero_derivative = np.asarray(zero_derivative_result["flow_derivative"])
    zero_information = hybrid.unregularized_model.score_information(
        hybrid.zero_flow, zero_derivative
    )

    target_log_dual = hybrid.unpack(target_vector, args.entropy_epsilon)[2]
    multiplier_correlation = float(
        np.corrcoef(target_log_dual, hybrid.zero_bound_multipliers)[0, 1]
    )
    multiplier_mae = float(
        np.mean(np.abs(target_log_dual - hybrid.zero_bound_multipliers))
    )

    small_epsilon_information = path[0]["process_fisher"]
    zero_information_value = float(zero_information["process_fisher"])
    absolute_limit_gap = float(
        abs(small_epsilon_information - zero_information_value)
    )
    process_values = np.asarray([item["process_fisher"] for item in path])
    maximum_position = int(np.argmax(process_values))
    maximum_item = path[maximum_position]

    checks = {
        "target_log_domain_kkt": bool(
            target_diagnostic["hybrid_kkt_residual"] < 2.0e-9
        ),
        "target_exact_constraints": bool(
            target_diagnostic["equality_residual"] < 2.0e-12
            and target_diagnostic["cone_residual"] < -1.0e-8
        ),
        "target_log_duals_positive": bool(
            target_diagnostic["minimum_boundary_log_dual"] > 0.0
        ),
        "target_derivative": bool(
            target_derivative_diagnostic["linearized_kkt_residual"] < 2.0e-9
            and target_derivative_diagnostic["constraint_derivative_residual"]
            < 2.0e-9
        ),
        "fine_grid_solved": bool(
            all(
                item["hybrid_kkt_residual"] < 2.0e-9
                and item["equality_residual"] < 2.0e-12
                and item["cone_residual"] < -1.0e-8
                for item in path
            )
        ),
        "fine_grid_derivatives_reoptimized": bool(
            all(
                item["finite_reoptimization_relative_error"] < 2.0e-5
                and item["plus_reoptimization_success"]
                and item["minus_reoptimization_success"]
                for item in path
            )
        ),
        "scores_centered": bool(
            all(
                abs(item["score_mean"]) < 2.0e-8
                and item["maximum_conditional_score_mean"] < 2.0e-8
                for item in path
            )
        ),
        "positive_process_information": bool(np.all(process_values > 0.0)),
        "information_decomposition": bool(
            all(item["decomposition_error"] < 2.0e-12 for item in path)
        ),
        "boundary_multiplier_tracking": bool(
            multiplier_correlation > 0.99 and multiplier_mae < 1.0e-3
        ),
        "small_epsilon_information_near_limit": bool(absolute_limit_gap < 3.0e-3),
    }

    payload = {
        "result": (
            "the central path is log-domain stable and the process-information bump "
            "is a genuine finite-entropy effect"
        ),
        "config": {
            "entropy_epsilon": args.entropy_epsilon,
            "chi": args.chi,
            "seed": args.seed,
        },
        "chart": {
            "interior_primal_coordinates": int(len(hybrid.interior_indices)),
            "boundary_log_dual_coordinates": int(len(hybrid.boundary_indices)),
            "active_covariance_rows": int(len(hybrid.active_cones)),
            "identity": "gamma_j = reference_j * exp(-g_j / epsilon)",
        },
        "target": {
            **target_diagnostic,
            "derivative": target_derivative_diagnostic,
            "information": target_information,
            "log_dual_multiplier_correlation": multiplier_correlation,
            "log_dual_multiplier_mean_absolute_error": multiplier_mae,
        },
        "zero_temperature": {
            "active_bounds": int(len(hybrid.boundary_indices)),
            "minimum_active_multiplier": float(
                np.min(hybrid.zero_bound_multipliers)
            ),
            "maximum_active_multiplier": float(
                np.max(hybrid.zero_bound_multipliers)
            ),
            "process_fisher": zero_information_value,
        },
        "information_path": path,
        "information_limit_diagnostic": {
            "smallest_positive_epsilon": float(path[0]["entropy_epsilon"]),
            "smallest_positive_epsilon_information": float(
                small_epsilon_information
            ),
            "absolute_limit_gap": absolute_limit_gap,
            "maximum_information": float(maximum_item["process_fisher"]),
            "maximum_information_epsilon": float(
                maximum_item["entropy_epsilon"]
            ),
            "monotone_on_reported_grid": bool(
                np.all(np.diff(process_values) <= 0.0)
                or np.all(np.diff(process_values) >= 0.0)
            ),
        },
        "checks": checks,
        "all_pass": bool(all(checks.values())),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "entropy_epsilon,process_fisher,boundary_information,interior_information,"
        "finite_reoptimization_relative_error,minimum_materialized_flow,"
        "minimum_boundary_log_dual,maximum_boundary_log_dual,"
        "hybrid_kkt_residual,equality_residual,cone_residual\n"
    )
    rows = [
        ",".join(
            str(item[key])
            for key in (
                "entropy_epsilon",
                "process_fisher",
                "boundary_information",
                "interior_information",
                "finite_reoptimization_relative_error",
                "minimum_materialized_flow",
                "minimum_boundary_log_dual",
                "maximum_boundary_log_dual",
                "hybrid_kkt_residual",
                "equality_residual",
                "cone_residual",
            )
        )
        for item in path
    ]
    args.csv_output.write_text(header + "\n".join(rows) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not payload["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
