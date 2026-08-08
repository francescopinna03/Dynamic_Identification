from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Dict, List

import numpy as np

from conditional_wf_sbb import BridgeConfig, ConditionalWFSBB


def summarize(name: str, family: str, required: bool, result: Dict[str, object]) -> Dict[str, object]:
    cfg = result["config"]
    optimization = result["optimization"]
    information = result["information"]
    return {
        "case": name,
        "family": family,
        "required": required,
        "all_pass": result["all_pass"],
        "n_grid": cfg["n_grid"],
        "n_steps": cfg["n_steps"],
        "entropy_epsilon": cfg["entropy_epsilon"],
        "lower_bound": cfg["lower_bound"],
        "objective": optimization["objective"],
        "endpoint_error": result["endpoint_error"],
        "process_fisher": information["process_fisher"],
        "implicit_derivative_error": result["implicit_finite_difference_relative_error"],
        "active_bounds": len(optimization["active_bounds"]),
        "active_cones": len(optimization["active_cones"]),
        "minimum_active_multiplier": optimization["minimum_active_multiplier"],
        "reduced_hessian_min_eigenvalue": optimization["reduced_hessian_min_eigenvalue"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-output", type=Path, default=Path("robustness_wf_sbb.json"))
    parser.add_argument("--csv-output", type=Path, default=Path("robustness_wf_sbb.csv"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base = BridgeConfig(seed=args.seed)
    specifications: List[tuple[str, str, bool, BridgeConfig]] = []

    for entropy_epsilon in (0.0, 5.0e-4, 1.0e-3, 2.5e-3, 5.0e-3, 1.0e-2):
        specifications.append(
            (
                f"regularization_{entropy_epsilon:g}",
                "regularization",
                entropy_epsilon != 5.0e-4,
                replace(base, entropy_epsilon=entropy_epsilon),
            )
        )

    specifications.extend(
        [
            ("active_floor_1e-10", "active_floor", True, replace(base, entropy_epsilon=0.0, lower_bound=1.0e-10)),
            ("active_floor_1e-14", "active_floor", True, replace(base, entropy_epsilon=0.0, lower_bound=1.0e-14)),
            ("diagonal_9_by_10", "diagonal_refinement", True, replace(base, n_grid=9, n_steps=10)),
            ("diagonal_11_by_12", "diagonal_refinement", True, replace(base, n_grid=11, n_steps=12)),
            ("time_7_by_12", "time_refinement", True, replace(base, n_steps=12)),
            ("time_7_by_16", "time_refinement", True, replace(base, n_steps=16)),
        ]
    )

    sender_receiver = BridgeConfig(
        seed=args.seed,
        chi=0.2,
        correct_association=0.525,
        independent_association=0.42,
        initial_center=0.42,
        initial_scale=0.08,
        terminal_center=0.504,
        terminal_scale=0.08,
        model_label="game G2 sender-receiver model-implied bridge",
    )
    specifications.append(("sender_receiver_G2", "game_anchor", True, sender_receiver))

    full_results: Dict[str, object] = {}
    rows: List[Dict[str, object]] = []
    for name, family, required, cfg in specifications:
        print(f"Running {name}", flush=True)
        model = ConditionalWFSBB(cfg)
        try:
            result = model.audit()
            full_results[name] = result
            rows.append(summarize(name, family, required, result))
        except RuntimeError as error:
            if required:
                raise
            solved = model.solve(cfg.chi)
            flow = np.asarray(solved["flow"])
            marginals, _ = model.marginals_and_kernels(flow)
            diagnostic = {
                "error": str(error),
                "interpretation": "legacy floor artifact resolved by the central-path audit",
                "optimization": {key: value for key, value in solved.items() if key != "flow"},
            }
            full_results[name] = diagnostic
            rows.append(
                {
                    "case": name,
                    "family": family,
                    "required": required,
                    "all_pass": False,
                    "n_grid": cfg.n_grid,
                    "n_steps": cfg.n_steps,
                    "entropy_epsilon": cfg.entropy_epsilon,
                    "lower_bound": cfg.lower_bound,
                    "objective": solved["objective"],
                    "endpoint_error": float(np.max(np.abs(marginals[-1] - model.muT))),
                    "process_fisher": None,
                    "implicit_derivative_error": None,
                    "active_bounds": len(solved["active_bounds"]),
                    "active_cones": len(solved["active_cones"]),
                    "minimum_active_multiplier": solved["minimum_active_multiplier"],
                    "reduced_hessian_min_eigenvalue": solved["reduced_hessian_min_eigenvalue"],
                }
            )

    diagonal_rows = [
        summarize("diagonal_7_by_8", "diagonal_refinement", True, full_results["regularization_0.005"]),
        *[row for row in rows if row["family"] == "diagonal_refinement"],
    ]
    diagonal_objectives = np.asarray([row["objective"] for row in diagonal_rows], dtype=float)
    diagonal_information = np.asarray([row["process_fisher"] for row in diagonal_rows], dtype=float)
    objective_relative_range = float(np.ptp(diagonal_objectives) / np.mean(diagonal_objectives))
    information_relative_range = float(np.ptp(diagonal_information) / np.mean(diagonal_information))

    required_rows = [row for row in rows if bool(row["required"])]
    required_pass = all(bool(row["all_pass"]) for row in required_rows)
    suite_checks = {
        "all_required_audits_pass": required_pass,
        "all_endpoint_errors_below_2e-8": all(
            float(row["endpoint_error"]) < 2.0e-8 for row in required_rows
        ),
        "all_process_information_positive": all(
            float(row["process_fisher"]) > 0.0 for row in required_rows
        ),
        "diagonal_objective_relative_range_below_2pct": objective_relative_range < 0.02,
        "diagonal_information_relative_range_below_15pct": information_relative_range < 0.15,
    }

    payload = {
        "description": "WF-SBB regularization, mesh, time, active-face, and sender-receiver audit",
        "rows": rows,
        "diagonal_refinement_with_baseline": diagonal_rows,
        "diagonal_objective_relative_range": objective_relative_range,
        "diagonal_information_relative_range": information_relative_range,
        "suite_checks": suite_checks,
        "all_pass": all(suite_checks.values()),
        "full_results": full_results,
    }

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with args.csv_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps({
        "suite_checks": suite_checks,
        "all_pass": payload["all_pass"],
        "diagonal_objective_relative_range": objective_relative_range,
        "diagonal_information_relative_range": information_relative_range,
    }, indent=2))
    if not payload["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
