from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


SHALLOW_TYPES = {"L1", "D1"}
DEEP_TYPES = {"L2", "L3", "E", "D2", "S"}
TYPE_TO_MACRO = {t: "shallow" for t in SHALLOW_TYPES}
TYPE_TO_MACRO.update({t: "deep" for t in DEEP_TYPES})
RANK_MAP = {"strong": 3, "medium": 2, "weak": 1, "degenerate": 0}


def ols_with_se(y: np.ndarray, x: np.ndarray) -> dict[str, float | int]:
    design = np.column_stack([np.ones(len(y)), x])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ beta
    sigma2 = float(residual @ residual) / (len(y) - design.shape[1])
    covariance = sigma2 * np.linalg.inv(design.T @ design)
    se = np.sqrt(np.diag(covariance))
    total = float((y - y.mean()) @ (y - y.mean()))
    r2 = 1.0 - float(residual @ residual) / total
    return {
        "intercept": float(beta[0]),
        "slope": float(beta[1]),
        "se_intercept": float(se[0]),
        "se_slope": float(se[1]),
        "t_intercept": float(beta[0] / se[0]),
        "t_slope": float(beta[1] / se[1]),
        "r_squared": r2,
        "n_games": int(len(y)),
    }


def prepare_cells(
    density: pd.DataFrame, types: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, bool | int | float]]:
    required_density = {"subject_id", "game_master", "type", "look_up", "density"}
    required_types = {"subject_id", "game_master", "treatment", "t_hat", "cat", "w_deep"}
    if missing := required_density.difference(density.columns):
        raise ValueError(f"density is missing columns {sorted(missing)}")
    if missing := required_types.difference(types.columns):
        raise ValueError(f"types is missing columns {sorted(missing)}")

    baseline = types.loc[
        types["treatment"].eq("Baseline"),
        ["subject_id", "game_master", "t_hat", "cat", "w_deep"],
    ].copy()
    key = ["subject_id", "game_master"]
    if baseline.duplicated(key).any():
        raise ValueError("fold-specific classifications have duplicate keys")

    expected_types = SHALLOW_TYPES | DEEP_TYPES
    observed_types = set(density["type"].unique())
    if observed_types != expected_types:
        raise ValueError(f"unexpected process types {sorted(observed_types)}")

    stats = (
        density.groupby(["game_master", "type", "look_up"])["density"]
        .agg(["mean", "std"])
        .rename(columns={"mean": "cell_mean", "std": "cell_std"})
        .reset_index()
    )
    cells = density.merge(stats, on=["game_master", "type", "look_up"], validate="many_to_one")
    cells["z_tgl"] = (
        (cells["density"] - cells["cell_mean"])
        / cells["cell_std"].clip(lower=1e-6)
    )
    cells = cells.merge(baseline, on=key, validate="many_to_one")
    cells["type_macro"] = cells["type"].map(TYPE_TO_MACRO)

    selected = cells.loc[
        cells["type"].eq(cells["t_hat"]),
        [*key, "look_up", "cat", "w_deep", "t_hat", "z_tgl"],
    ].rename(columns={"z_tgl": "normalized_lookup"})
    opposite = (
        cells.loc[cells["type_macro"].ne(cells["cat"])]
        .groupby([*key, "look_up"])["z_tgl"]
        .max()
        .rename("opposite_max")
        .reset_index()
    )
    selected = selected.merge(opposite, on=[*key, "look_up"], validate="one_to_one")
    selected["macro_margin"] = selected["normalized_lookup"] - selected["opposite_max"]

    diagnostics: dict[str, bool | int | float] = {
        "n_density_rows": int(len(density)),
        "n_baseline_subjects": int(baseline["subject_id"].nunique()),
        "n_games": int(baseline["game_master"].nunique()),
        "n_fold_classifications": int(len(baseline)),
        "n_selected_cells": int(len(selected)),
        "unique_fold_keys": not baseline.duplicated(key).any(),
        "complete_fold_panel": len(baseline) == 71 * 16,
        "complete_process_panel": len(density) == 71 * 16 * 7 * 2,
        "finite_scores": bool(
            np.isfinite(selected[["normalized_lookup", "macro_margin"]].to_numpy()).all()
        ),
        "maximum_normalization_mean_error": float(
            cells.groupby(["game_master", "type", "look_up"])["z_tgl"].mean().abs().max()
        ),
    }
    return selected, diagnostics


def directional_test(
    selected: pd.DataFrame,
    fisher: pd.DataFrame,
    outcome: str,
    look_up: str,
) -> tuple[dict[str, float | int | str], pd.DataFrame]:
    means = (
        selected.loc[selected["look_up"].eq(look_up)]
        .groupby(["game_master", "cat"])[outcome]
        .mean()
        .unstack()
    )
    if set(means.columns) != {"deep", "shallow"}:
        raise ValueError(f"missing depth class for {outcome} and {look_up}")
    means["contrast"] = means["deep"] - means["shallow"]
    table = means.reset_index().merge(
        fisher[["game_master", "tau_informativity", "se_tau", "I_tau_cond"]],
        on="game_master",
        validate="one_to_one",
    )
    table["information_rank"] = table["tau_informativity"].map(RANK_MAP)
    if table["information_rank"].isna().any():
        raise ValueError("unknown information category")
    regression = ols_with_se(
        table["contrast"].to_numpy(), table["information_rank"].to_numpy()
    )
    rho, p_value = spearmanr(-table["se_tau"].to_numpy(), table["contrast"].to_numpy())
    result: dict[str, float | int | str] = {
        "outcome": outcome,
        "look_up": look_up,
        **regression,
        "spearman_rho": float(rho),
        "spearman_p_value": float(p_value),
    }
    table.insert(1, "look_up", look_up)
    table.insert(2, "outcome", outcome)
    return result, table


def cell_agreement(
    density: pd.DataFrame, types: pd.DataFrame, value: str
) -> dict[str, float]:
    baseline = types.loc[
        types["treatment"].eq("Baseline"),
        ["subject_id", "game_master", "cat"],
    ]
    if value == "z_tgl":
        stats = (
            density.groupby(["game_master", "type", "look_up"])["density"]
            .agg(["mean", "std"])
            .reset_index()
        )
        work = density.merge(stats, on=["game_master", "type", "look_up"])
        work["z_tgl"] = (work["density"] - work["mean"]) / work["std"].clip(lower=1e-6)
    else:
        work = density.copy()
    work["search_cat"] = work["type"].map(TYPE_TO_MACRO)
    answer: dict[str, float] = {}
    for look_up in ["early", "late"]:
        subset = work.loc[work["look_up"].eq(look_up)]
        maxima = subset.loc[
            subset.groupby(["subject_id", "game_master"])[value].idxmax(),
            ["subject_id", "game_master", "search_cat"],
        ]
        joined = baseline.merge(maxima, on=["subject_id", "game_master"], validate="one_to_one")
        answer[look_up] = float(joined["cat"].eq(joined["search_cat"]).mean())
    return answer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--types", type=Path, required=True)
    parser.add_argument("--fisher", type=Path, required=True)
    parser.add_argument("--density", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    args = parser.parse_args(argv)

    types = pd.read_csv(args.types)
    fisher = pd.read_csv(args.fisher)
    density = pd.read_csv(args.density)
    selected, diagnostics = prepare_cells(density, types)

    results = []
    tables = []
    for look_up in ["early", "late"]:
        for outcome in ["normalized_lookup", "macro_margin"]:
            result, table = directional_test(selected, fisher, outcome, look_up)
            results.append(result)
            tables.append(table)

    diagnostics["raw_cell_agreement"] = cell_agreement(density, types, "density")
    diagnostics["normalized_cell_agreement"] = cell_agreement(density, types, "z_tgl")
    checks = {
        "unique_fold_keys": bool(diagnostics["unique_fold_keys"]),
        "complete_fold_panel": bool(diagnostics["complete_fold_panel"]),
        "complete_process_panel": bool(diagnostics["complete_process_panel"]),
        "finite_scores": bool(diagnostics["finite_scores"]),
        "normalization_centered": diagnostics["maximum_normalization_mean_error"] < 1e-12,
        "four_regressions": len(results) == 4,
        "sixteen_games_each": all(item["n_games"] == 16 for item in results),
        "finite_regressions": bool(
            np.isfinite(
                [[item["slope"], item["se_slope"], item["t_slope"], item["r_squared"]]
                 for item in results]
            ).all()
        ),
    }
    payload = {
        "model": "cross-fitted MouseLab process audit",
        "classification": "leave-one-game-out subject type",
        "diagnostics": diagnostics,
        "results": results,
        "checks": checks,
        "all_pass": all(checks.values()),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    pd.concat(tables, ignore_index=True).to_csv(args.csv_output, index=False)
    print(json.dumps(payload, indent=2))
    return 0 if payload["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
