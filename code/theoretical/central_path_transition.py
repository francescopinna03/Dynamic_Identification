from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import null_space

from conditional_wf_sbb import BridgeConfig, ConditionalWFSBB


Array = NDArray[np.float64]


class EntropicCentralPath:
    def __init__(self, base: BridgeConfig):
        self.base = base
        reference_model = ConditionalWFSBB(base)
        self.tangent = null_space(reference_model.Aeq)

    def model(self, entropy_epsilon: float) -> ConditionalWFSBB:
        return ConditionalWFSBB(
            replace(
                self.base,
                entropy_epsilon=float(entropy_epsilon),
                lower_bound=1.0e-30,
            )
        )

    def projected_newton(
        self,
        model: ConditionalWFSBB,
        flow: Array,
        chi: float,
        tolerance: float = 2.0e-11,
        max_iterations: int = 500,
    ) -> Tuple[Array, Dict[str, object]]:
        z = np.asarray(flow, dtype=float).copy()
        tangent = self.tangent
        accepted_steps = 0
        for iteration in range(max_iterations):
            value, gradient = model.objective_gradient(z, chi)
            reduced_gradient = tangent.T @ gradient
            residual = float(np.max(np.abs(reduced_gradient)))
            if residual < tolerance:
                return z, {
                    "success": True,
                    "iterations": iteration,
                    "accepted_steps": accepted_steps,
                    "reduced_gradient_residual": residual,
                }

            hessian = model.hessian(z, chi)
            reduced_hessian = tangent.T @ hessian @ tangent
            reduced_direction = np.linalg.solve(reduced_hessian, -reduced_gradient)
            direction = tangent @ reduced_direction
            slope = float(gradient @ direction)

            alpha = 1.0
            decreasing = direction < 0.0
            if np.any(decreasing):
                alpha = min(alpha, 0.995 * float(np.min(z[decreasing] / -direction[decreasing])))
            cone_value = model.Acone @ z
            cone_direction = model.Acone @ direction
            tightening = cone_direction > 0.0
            if np.any(tightening):
                alpha = min(
                    alpha,
                    0.995 * float(np.min(-cone_value[tightening] / cone_direction[tightening])),
                )

            accepted = False
            for _ in range(120):
                candidate = z + alpha * direction
                if candidate.min() <= 0.0 or np.max(model.Acone @ candidate) >= 0.0:
                    alpha *= 0.5
                    continue
                candidate_value, candidate_gradient = model.objective_gradient(candidate, chi)
                candidate_residual = float(
                    np.max(np.abs(tangent.T @ candidate_gradient))
                )
                objective_acceptance = candidate_value <= value + 1.0e-4 * alpha * slope
                residual_acceptance = candidate_residual <= residual * (1.0 - 1.0e-4 * alpha)
                if objective_acceptance or residual_acceptance:
                    z = candidate
                    accepted = True
                    accepted_steps += 1
                    break
                alpha *= 0.5
            if not accepted:
                break

        final_residual = float(
            np.max(np.abs(tangent.T @ model.objective_gradient(z, chi)[1]))
        )
        return z, {
            "success": final_residual < max(2.0e-8, 20.0 * tolerance),
            "iterations": max_iterations,
            "accepted_steps": accepted_steps,
            "reduced_gradient_residual": final_residual,
        }

    def solve(self, target_epsilon: float, chi: float) -> Tuple[ConditionalWFSBB, Array, Dict[str, object]]:
        start_epsilon = max(2.5e-3, target_epsilon)
        start_model = self.model(start_epsilon)
        initialized = start_model.solve(chi)
        if not initialized["success"]:
            raise RuntimeError("The regular homotopy initialization failed")
        z = np.asarray(initialized["flow"], dtype=float)

        if target_epsilon < start_epsilon:
            schedule = np.geomspace(start_epsilon, target_epsilon, 25)[1:]
        else:
            schedule = np.asarray([target_epsilon])
        history = []
        for epsilon in schedule:
            model = self.model(float(epsilon))
            final_step = bool(abs(float(epsilon) - target_epsilon) < 1.0e-16)
            tolerance = 2.0e-12 if final_step else 2.0e-8
            z, diagnostic = self.projected_newton(
                model,
                z,
                chi,
                tolerance=tolerance,
            )
            history.append(
                {
                    "entropy_epsilon": float(epsilon),
                    "minimum_flow": float(z.min()),
                    "objective": float(model.objective_gradient(z, chi)[0]),
                    **diagnostic,
                }
            )

        target_model = self.model(target_epsilon)
        z, final_diagnostic = self.projected_newton(
            target_model,
            z,
            chi,
            tolerance=2.0e-12,
            max_iterations=800,
        )
        return target_model, z, {
            "initialization": {key: value for key, value in initialized.items() if key != "flow"},
            "homotopy": history,
            "final": final_diagnostic,
        }


def kkt_derivative(
    model: ConditionalWFSBB,
    flow: Array,
    chi: float,
    cross_gradient: Array,
) -> Tuple[Array, Dict[str, float]]:
    hessian = model.hessian(flow, chi)
    equality = model.Aeq
    kkt = np.block(
        [
            [hessian, equality.T],
            [equality, np.zeros((equality.shape[0], equality.shape[0]))],
        ]
    )
    derivative = np.linalg.solve(
        kkt,
        -np.concatenate([cross_gradient, np.zeros(equality.shape[0])]),
    )[: model.n_var]
    tangent = null_space(equality)
    reduced_hessian = tangent.T @ hessian @ tangent
    return derivative, {
        "kkt_condition_number": float(np.linalg.cond(kkt)),
        "reduced_hessian_condition_number": float(np.linalg.cond(reduced_hessian)),
        "reduced_hessian_min_eigenvalue": float(np.linalg.eigvalsh(reduced_hessian).min()),
        "constraint_derivative_residual": float(np.max(np.abs(equality @ derivative))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("central_path_transition.json"))
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
    central = EntropicCentralPath(base)
    model, flow, homotopy = central.solve(args.entropy_epsilon, args.chi)

    neighborhood = []
    for epsilon_ratio in (0.90, 0.95, 1.00, 1.05, 1.10):
        local_epsilon = args.entropy_epsilon * epsilon_ratio
        local_model = central.model(local_epsilon)
        local_flow, local_diagnostic = central.projected_newton(
            local_model,
            flow,
            args.chi,
            tolerance=2.0e-12,
            max_iterations=800,
        )
        neighborhood.append(
            {
                "entropy_epsilon": float(local_epsilon),
                "minimum_flow": float(local_flow.min()),
                "equality_residual": float(
                    np.max(np.abs(local_model.Aeq @ local_flow - local_model.beq))
                ),
                "cone_residual": float(np.max(local_model.Acone @ local_flow)),
                **local_diagnostic,
            }
        )

    chi_step = 2.0e-4
    chi_cross_step = 2.0e-5
    chi_cross = (
        model.objective_gradient(flow, args.chi + chi_cross_step)[1]
        - model.objective_gradient(flow, args.chi - chi_cross_step)[1]
    ) / (2.0 * chi_cross_step)
    chi_derivative, chi_kkt = kkt_derivative(model, flow, args.chi, chi_cross)
    flow_plus_chi, plus_chi_diagnostic = central.projected_newton(
        model, flow, args.chi + chi_step, tolerance=2.0e-12, max_iterations=800
    )
    flow_minus_chi, minus_chi_diagnostic = central.projected_newton(
        model, flow, args.chi - chi_step, tolerance=2.0e-12, max_iterations=800
    )
    finite_chi_derivative = (flow_plus_chi - flow_minus_chi) / (2.0 * chi_step)
    chi_derivative_error = float(
        np.linalg.norm(chi_derivative - finite_chi_derivative)
        / max(1.0, np.linalg.norm(finite_chi_derivative))
    )

    epsilon_step = 1.0e-6
    epsilon_cross = np.log(flow / model.reference_flow)
    epsilon_derivative, epsilon_kkt = kkt_derivative(
        model, flow, args.chi, epsilon_cross
    )
    plus_epsilon_model = central.model(args.entropy_epsilon + epsilon_step)
    minus_epsilon_model = central.model(args.entropy_epsilon - epsilon_step)
    flow_plus_epsilon, plus_epsilon_diagnostic = central.projected_newton(
        plus_epsilon_model, flow, args.chi, tolerance=2.0e-12, max_iterations=800
    )
    flow_minus_epsilon, minus_epsilon_diagnostic = central.projected_newton(
        minus_epsilon_model, flow, args.chi, tolerance=2.0e-12, max_iterations=800
    )
    finite_epsilon_derivative = (
        flow_plus_epsilon - flow_minus_epsilon
    ) / (2.0 * epsilon_step)
    epsilon_derivative_error = float(
        np.linalg.norm(epsilon_derivative - finite_epsilon_derivative)
        / max(1.0, np.linalg.norm(finite_epsilon_derivative))
    )

    zero_model = ConditionalWFSBB(
        replace(base, entropy_epsilon=0.0, lower_bound=1.0e-14)
    )
    zero_solution = zero_model.solve(args.chi)
    zero_flow = np.asarray(zero_solution["flow"])
    active_bounds = list(zero_solution["active_bounds"])
    active_matrix = zero_model._active_constraint_matrix(active_bounds, [])
    zero_gradient = zero_model.objective_gradient(zero_flow, args.chi)[1]
    zero_multipliers = np.linalg.lstsq(
        active_matrix.T, -zero_gradient, rcond=None
    )[0]
    active_multipliers = zero_multipliers[zero_model.Aeq.shape[0] :]
    scaled_log_flows = -args.entropy_epsilon * np.log(
        flow[active_bounds] / model.reference_flow[active_bounds]
    )
    multiplier_correlation = float(
        np.corrcoef(active_multipliers, scaled_log_flows)[0, 1]
    )
    multiplier_mean_absolute_error = float(
        np.mean(np.abs(active_multipliers - scaled_log_flows))
    )

    floor_model = ConditionalWFSBB(
        replace(base, entropy_epsilon=args.entropy_epsilon, lower_bound=1.0e-12)
    )
    floor_solution = floor_model.solve(args.chi)
    floor_flow = np.asarray(floor_solution["flow"])
    floor_objective_gap = float(
        floor_model.objective_gradient(floor_flow, args.chi)[0]
        - model.objective_gradient(flow, args.chi)[0]
    )

    equality_residual = float(np.max(np.abs(model.Aeq @ flow - model.beq)))
    cone_residual = float(np.max(model.Acone @ flow))
    reduced_gradient_residual = float(
        np.max(np.abs(central.tangent.T @ model.objective_gradient(flow, args.chi)[1]))
    )
    information = model.score_information(flow, chi_derivative)

    checks = {
        "positive_interior_flow": bool(flow.min() > 0.0),
        "old_floor_crossed": bool(flow.min() < 1.0e-12),
        "exact_flow_equalities": bool(equality_residual < 2.0e-12),
        "strict_cone_slack": bool(cone_residual < -1.0e-8),
        "central_kkt": bool(reduced_gradient_residual < 2.0e-9),
        "positive_reduced_hessian": bool(chi_kkt["reduced_hessian_min_eigenvalue"] > 0.0),
        "chi_derivative": bool(chi_derivative_error < 2.0e-5),
        "entropy_derivative": bool(epsilon_derivative_error < 2.0e-4),
        "stable_positive_neighborhood": bool(
            all(
                item["minimum_flow"] > 0.0
                and item["cone_residual"] < -1.0e-8
                and item["reduced_gradient_residual"] < 2.0e-9
                for item in neighborhood
            )
        ),
        "score_centered": bool(abs(information["score_mean"]) < 2.0e-8),
        "process_information_positive": bool(information["process_fisher"] > 0.0),
        "zero_face_strict_complementarity": bool(active_multipliers.min() > 0.0),
        "exponential_boundary_layer": bool(
            multiplier_correlation > 0.99 and multiplier_mean_absolute_error < 1.0e-3
        ),
        "floor_objective_gap_small": bool(abs(floor_objective_gap) < 1.0e-8),
    }

    payload = {
        "result": "the apparent positive-entropy face transition is a numerical-floor artifact",
        "config": {
            "entropy_epsilon": args.entropy_epsilon,
            "chi": args.chi,
            "seed": args.seed,
            "old_numerical_floor": 1.0e-12,
        },
        "central_solution": {
            "objective": float(model.objective_gradient(flow, args.chi)[0]),
            "minimum_flow": float(flow.min()),
            "equality_residual": equality_residual,
            "cone_residual": cone_residual,
            "reduced_gradient_residual": reduced_gradient_residual,
            "positive_flow_count": int(np.sum(flow > 0.0)),
            "flow_count": int(flow.size),
        },
        "homotopy": homotopy,
        "local_entropy_neighborhood": neighborhood,
        "chi_derivative": {
            **chi_kkt,
            "finite_reoptimization_relative_error": chi_derivative_error,
            "plus_reoptimization": plus_chi_diagnostic,
            "minus_reoptimization": minus_chi_diagnostic,
        },
        "entropy_derivative": {
            **epsilon_kkt,
            "finite_reoptimization_relative_error": epsilon_derivative_error,
            "derivative_norm": float(np.linalg.norm(epsilon_derivative)),
            "plus_reoptimization": plus_epsilon_diagnostic,
            "minus_reoptimization": minus_epsilon_diagnostic,
        },
        "zero_temperature_face": {
            "active_bounds": len(active_bounds),
            "minimum_active_multiplier": float(active_multipliers.min()),
            "maximum_active_multiplier": float(active_multipliers.max()),
            "scaled_log_multiplier_correlation": multiplier_correlation,
            "scaled_log_multiplier_mean_absolute_error": multiplier_mean_absolute_error,
        },
        "old_floor_diagnostic": {
            "reported_success": bool(floor_solution["success"]),
            "apparent_active_bounds": len(floor_solution["active_bounds"]),
            "minimum_candidate_multiplier": floor_solution["minimum_active_multiplier"],
            "objective_gap_from_central_solution": floor_objective_gap,
        },
        "information": information,
        "checks": checks,
        "all_pass": bool(all(checks.values())),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not payload["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
