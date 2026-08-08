from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import sqrtm
from scipy.optimize import minimize


Array = np.ndarray


def load_design_module():
    path = Path(__file__).with_name("design_games.py")
    spec = importlib.util.spec_from_file_location("design_games", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the design module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def bures_squared(a: Array, b: Array) -> float:
    middle = sqrtm(sqrtm(b) @ a @ sqrtm(b))
    value = np.trace(a) + np.trace(b) - 2.0 * np.trace(middle)
    return float(np.real_if_close(value))


def check_auction_contraction() -> dict:
    design = load_design_module()
    result = design.auction_contraction_certificate(
        1.8, 0.2, 3, 3, 0.03
    )
    expected_diameter = 791.0 / 1200.0
    expected_modulus = 2373.0 / 8000.0
    assert abs(result["reduced_affine_diameter"] - expected_diameter) < 1e-12
    assert abs(result["contraction_modulus"] - expected_modulus) < 1e-12
    assert result["certified_unique"]
    return result


def check_wf_bures_feedback(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    dimension = 3
    raw = rng.normal(size=(dimension, dimension))
    reference = raw @ raw.T + 0.7 * np.eye(dimension)
    curvature_raw = rng.normal(size=(dimension, dimension))
    curvature = 0.12 * (curvature_raw + curvature_raw.T)
    rho = 2.0
    c_matrix = np.eye(dimension) + curvature / rho
    eigenvalues = np.linalg.eigvalsh(c_matrix)
    if eigenvalues.min() <= 0.25:
        curvature += (0.3 - eigenvalues.min()) * rho * np.eye(dimension)
        c_matrix = np.eye(dimension) + curvature / rho
    c_inverse = np.linalg.inv(c_matrix)
    analytic_covariance = c_inverse @ reference @ c_inverse

    reference_root = np.real_if_close(sqrtm(reference))

    def factor_objective(flat_factor: Array) -> float:
        factor = flat_factor.reshape(dimension, dimension)
        covariance = factor @ factor.T
        return float(
            0.5 * np.trace(curvature @ covariance)
            + 0.5 * rho * np.sum((factor - reference_root) ** 2)
        )

    def factor_gradient(flat_factor: Array) -> Array:
        factor = flat_factor.reshape(dimension, dimension)
        gradient = curvature @ factor + rho * (factor - reference_root)
        return np.asarray(gradient, dtype=float).reshape(-1)

    analytic_factor = c_inverse @ reference_root
    optimization = minimize(
        factor_objective,
        np.asarray(reference_root).reshape(-1),
        method="BFGS",
        jac=factor_gradient,
        options={"gtol": 1e-11, "maxiter": 5000},
    )
    numerical_factor = optimization.x.reshape(dimension, dimension)
    numerical_covariance = numerical_factor @ numerical_factor.T
    relative_gap = float(
        np.linalg.norm(numerical_covariance - analytic_covariance)
        / np.linalg.norm(analytic_covariance)
    )
    assert relative_gap < 2e-6
    assert optimization.success

    covariance_objective = float(
        0.5 * np.trace(curvature @ analytic_covariance)
        + 0.5 * rho * bures_squared(analytic_covariance, reference)
    )
    factor_value = factor_objective(analytic_factor.reshape(-1))
    assert abs(covariance_objective - factor_value) < 1e-8
    return {
        "minimum_curvature_eigenvalue": float(
            np.linalg.eigvalsh(c_matrix).min()
        ),
        "relative_covariance_gap": relative_gap,
        "covariance_objective": covariance_objective,
        "optimizer_success": bool(optimization.success),
    }


def check_singular_jet() -> dict:
    p0 = np.array([0.20, 0.30, 0.50])
    h = np.array([0.10, -0.04, -0.06])
    order = 2
    information = float(np.sum(h * h / p0))
    target = information / (2.0 * math.factorial(order) ** 2)
    ratios = []
    for t_value in (0.08, 0.05, 0.03, 0.02):
        pt = p0 + (t_value ** order / math.factorial(order)) * h
        divergence = float(np.sum(pt * np.log(pt / p0)))
        ratios.append(divergence / t_value ** (2 * order))
    relative_error = abs(ratios[-1] - target) / target
    assert relative_error < 2e-4
    return {
        "order": order,
        "jet_information": information,
        "kl_coefficient_target": target,
        "scaled_kl_values": ratios,
        "last_relative_error": relative_error,
    }


def check_terminal_copula_assembly() -> dict:
    belief_terminal = np.array([0.4, 0.6])
    depth_terminal = np.array([0.7, 0.3])
    terminal_copula = np.array([[0.32, 0.08], [0.38, 0.22]])
    assert np.allclose(terminal_copula.sum(axis=1), belief_terminal)
    assert np.allclose(terminal_copula.sum(axis=0), depth_terminal)

    belief_path_given_terminal = np.array([
        [0.8, 0.2],
        [0.1, 0.9],
    ])
    depth_path_given_terminal = np.array([
        [0.75, 0.25],
        [0.2, 0.8],
    ])
    joint_paths = np.einsum(
        "mr,mi,rj->ij", terminal_copula,
        belief_path_given_terminal, depth_path_given_terminal
    )
    belief_path = belief_terminal @ belief_path_given_terminal
    depth_path = depth_terminal @ depth_path_given_terminal
    assert np.allclose(joint_paths.sum(axis=1), belief_path)
    assert np.allclose(joint_paths.sum(axis=0), depth_path)
    assert abs(joint_paths.sum() - 1.0) < 1e-14
    return {
        "joint_path_mass": float(joint_paths.sum()),
        "belief_marginal_max_error": float(np.max(np.abs(
            joint_paths.sum(axis=1) - belief_path
        ))),
        "depth_marginal_max_error": float(np.max(np.abs(
            joint_paths.sum(axis=0) - depth_path
        ))),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path,
                        default=Path("math_checks.json"))
    args = parser.parse_args(argv)

    report = {
        "release": "1.0",
        "seed": args.seed,
        "auction_contraction": check_auction_contraction(),
        "wf_bures_feedback": check_wf_bures_feedback(args.seed),
        "singular_jet": check_singular_jet(),
        "terminal_copula_assembly": check_terminal_copula_assembly(),
        "all_checks_pass": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
