from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import poisson


Array = np.ndarray


def softmax(x: Array) -> Array:
    z = np.asarray(x, dtype=float)
    z = z - np.max(z)
    e = np.exp(z)
    return e / e.sum()


def payoff_normalize(u: Array) -> tuple[Array, float]:
    u = np.asarray(u, dtype=float)
    span = float(np.max(u) - np.min(u))
    if not np.isfinite(span) or span <= 0:
        raise ValueError("payoff span must be positive")
    return u / span, span


def fisher_summary(I: Array) -> dict:
    eig = np.linalg.eigvalsh(I)
    rank = int(np.sum(eig > 1e-9))
    out = {
        "fisher": I.tolist(),
        "eigenvalues": eig.tolist(),
        "rank": rank,
    }
    if rank == I.shape[0]:
        inv = np.linalg.inv(I)
        out.update(
            inverse=inv.tolist(),
            trace_inverse=float(np.trace(inv)),
            standard_errors=np.sqrt(np.diag(inv)).tolist(),
        )
    return out


def sample_sizes(I_subject: Array, delta: float) -> dict:
    inv = np.linalg.inv(I_subject)
    p = I_subject.shape[0]
    return {
        "delta": float(delta),
        "average_mse": int(np.ceil(np.trace(inv) / (p * delta * delta))),
        "all_standard_errors": int(
            np.ceil(np.max(np.diag(inv)) / (delta * delta))
        ),
    }


def sr_posterior(pi: float, t_h: float, t_l: float) -> tuple[float, ...]:
    p_h = pi * t_h + (1 - pi) * (1 - t_l)
    p_l = 1 - p_h
    mu_h = pi * t_h / p_h
    mu_l = pi * (1 - t_h) / p_l
    return mu_h, mu_l, p_h, p_l


def sr_ccp(eta: Array, pi: float, t_h: float, t_l: float) -> Array:
    lam, chi, tau = np.asarray(eta, dtype=float)
    mu_h, mu_l, p_h, p_l = sr_posterior(pi, t_h, t_l)
    mu = np.array([mu_h, mu_l])
    tilted = (1 - chi) * mu + chi * pi
    index = 2 * tilted - 1
    strategic = 1 / (1 + np.exp(-lam * index))
    alpha = 1 - np.exp(-tau)
    q = alpha * strategic + (1 - alpha) * 0.5
    return np.array([q[0], q[1], p_h, p_l])


def sr_fisher(eta: Array, pi: float, t_h: float, t_l: float,
              eps: float = 1e-5) -> Array:
    base = sr_ccp(eta, pi, t_h, t_l)
    q = base[:2]
    weights = base[2:]
    derivative = np.empty((3, 2))
    for j in range(3):
        plus = np.asarray(eta, dtype=float).copy()
        minus = plus.copy()
        plus[j] += eps
        minus[j] -= eps
        derivative[j] = (
            sr_ccp(plus, pi, t_h, t_l)[:2]
            - sr_ccp(minus, pi, t_h, t_l)[:2]
        ) / (2 * eps)
    I = np.zeros((3, 3))
    for m in range(2):
        I += weights[m] * np.outer(derivative[:, m], derivative[:, m]) / (
            q[m] * (1 - q[m])
        )
    return I


def bc_payoff(n_actions: int, target: float) -> tuple[Array, float]:
    actions = np.linspace(0, 1, n_actions)
    raw = -(actions[:, None] - target * actions[None, :]) ** 2
    return payoff_normalize(raw)


def bc_terminal(lam: float, tau: float, n_actions: int, target: float,
                k_max: int = 24) -> Array:
    payoff, _ = bc_payoff(n_actions, target)
    levels = np.zeros((k_max + 1, n_actions))
    levels[0] = 1 / n_actions
    mass = poisson.pmf(np.arange(k_max + 1), tau)
    mass[-1] += max(0.0, 1.0 - mass.sum())
    for k in range(1, k_max + 1):
        lower = mass[:k]
        belief = lower @ levels[:k] / lower.sum()
        levels[k] = softmax(lam * (payoff @ belief))
    terminal = mass @ levels
    return terminal / terminal.sum()


def bc_fisher(eta: Array, n_actions: int, target: float,
              eps: float = 1e-5) -> Array:
    lam, _, tau = np.asarray(eta, dtype=float)
    base = bc_terminal(lam, tau, n_actions, target)
    derivative = np.zeros((3, n_actions))
    for j in (0, 2):
        plus = np.asarray(eta, dtype=float).copy()
        minus = plus.copy()
        plus[j] += eps
        minus[j] -= eps
        derivative[j] = (
            bc_terminal(plus[0], plus[2], n_actions, target)
            - bc_terminal(minus[0], minus[2], n_actions, target)
        ) / (2 * eps)
    I = np.zeros((3, 3))
    for a in range(n_actions):
        I += np.outer(derivative[:, a], derivative[:, a]) / base[a]
    return I


def signal_kernel(n_values: int, sigma: float) -> Array:
    kernel = np.full((n_values, n_values), sigma / (n_values - 1))
    np.fill_diagonal(kernel, 1 - sigma)
    return kernel


def auction_payoff(n_values: int, n_bids: int) -> tuple[Array, float]:
    values = np.linspace(0, 1, n_values)
    bids = np.linspace(0, 1, n_bids)
    own = bids[:, None, None]
    other = bids[None, :, None]
    value = values[None, None, :]
    raw = (own > other) * (value - own)
    raw = raw + 0.5 * (own == other) * (value - own)
    return payoff_normalize(raw)


def auction_utilities(policy: Array, chi: float, kernel: Array,
                      payoff: Array) -> Array:
    n_values = kernel.shape[0]
    prior = np.full(n_values, 1 / n_values)
    p_signal = prior @ kernel
    posterior = prior[None, :] * kernel.T / p_signal[:, None]
    bid_given_value = kernel @ policy
    correct_at_value = np.einsum("vj,ijv->iv", bid_given_value, payoff)
    correct = posterior @ correct_at_value.T
    marginal_bid = posterior @ bid_given_value
    cursed = np.einsum("sv,sj,ijv->si", posterior, marginal_bid, payoff)
    return (1 - chi) * correct + chi * cursed


def auction_qre(lam: float, chi: float, n_values: int, n_bids: int,
                sigma: float, start: Array | None = None,
                damping: float = 0.5, tolerance: float = 1e-10,
                max_iterations: int = 5000) -> tuple[Array, float, int]:
    kernel = signal_kernel(n_values, sigma)
    payoff, _ = auction_payoff(n_values, n_bids)
    if start is None:
        policy = np.full((n_values, n_bids), 1 / n_bids)
    else:
        policy = np.asarray(start, dtype=float)
        policy = policy / policy.sum(axis=1, keepdims=True)
    residual = np.inf
    for iteration in range(1, max_iterations + 1):
        utilities = auction_utilities(policy, chi, kernel, payoff)
        mapped = np.vstack([softmax(lam * row) for row in utilities])
        residual = float(np.max(np.abs(mapped - policy)))
        policy = (1 - damping) * policy + damping * mapped
        if residual <= tolerance:
            return policy, residual, iteration
    raise RuntimeError(f"auction QRE failed with residual {residual}")


def auction_multistart(lam: float, chi: float, n_values: int, n_bids: int,
                       sigma: float) -> dict:
    uniform = np.full((n_values, n_bids), 1 / n_bids)
    low = np.zeros_like(uniform)
    low[:, 0] = 1
    high = np.zeros_like(uniform)
    high[:, -1] = 1
    signal = np.zeros_like(uniform)
    for s in range(n_values):
        signal[s, round(s * (n_bids - 1) / (n_values - 1))] = 1
    solutions = []
    diagnostics = []
    for label, start in (("uniform", uniform), ("low", low),
                         ("high", high), ("signal", signal)):
        policy, residual, iterations = auction_qre(
            lam, chi, n_values, n_bids, sigma, start=start
        )
        diagnostics.append({
            "start": label,
            "residual": residual,
            "iterations": iterations,
        })
        if all(np.max(np.abs(policy - old)) > 1e-7 for old in solutions):
            solutions.append(policy)
    return {
        "distinct_fixed_points": len(solutions),
        "diagnostics": diagnostics,
        "max_pairwise_gap": float(max(
            [0.0] + [
                np.max(np.abs(a - b))
                for i, a in enumerate(solutions)
                for b in solutions[i + 1:]
            ]
        )),
    }


def auction_fisher(eta: Array, n_values: int, n_bids: int, sigma: float,
                   eps: float = 1e-5) -> Array:
    lam, chi, _ = np.asarray(eta, dtype=float)
    base, _, _ = auction_qre(lam, chi, n_values, n_bids, sigma)
    prior = np.full(n_values, 1 / n_values)
    p_signal = prior @ signal_kernel(n_values, sigma)
    derivative = np.zeros((3, n_values, n_bids))
    for j in (0, 1):
        plus = np.asarray(eta, dtype=float).copy()
        minus = plus.copy()
        plus[j] += eps
        minus[j] -= eps
        derivative[j] = (
            auction_qre(plus[0], plus[1], n_values, n_bids, sigma)[0]
            - auction_qre(minus[0], minus[1], n_values, n_bids, sigma)[0]
        ) / (2 * eps)
    I = np.zeros((3, 3))
    for s in range(n_values):
        for b in range(n_bids):
            I += p_signal[s] * np.outer(
                derivative[:, s, b], derivative[:, s, b]
            ) / base[s, b]
    return I


@dataclass(frozen=True)
class Candidate:
    game: str
    design: tuple
    fisher: Array


def valid_objective(I: Array) -> float:
    eig = np.linalg.eigvalsh(I)
    if eig[0] <= 1e-9:
        return np.inf
    return float(np.trace(np.linalg.inv(I)))


def grid_candidates(eta: Array) -> tuple[list[Candidate], ...]:
    g2 = [
        Candidate("G2", (pi, t), sr_fisher(eta, pi, t, t))
        for pi, t in product((0.60, 0.70, 0.80, 0.90),
                             (0.60, 0.70, 0.80, 0.90, 0.95))
    ]
    g3 = [
        Candidate("G3", (n, p), bc_fisher(eta, n, p))
        for n, p in product((5, 7, 9, 11),
                            (0.40, 0.50, 0.60, 2 / 3, 0.75, 0.85))
    ]
    g4 = [
        Candidate("G4", (v, b, sigma), auction_fisher(eta, v, b, sigma))
        for v in (3, 4, 5, 6)
        for b in sorted({max(3, v - 1), v, v + 2})
        for sigma in (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60)
    ]
    return g2, g3, g4


def best_grid_battery(eta: Array) -> tuple[dict, tuple[list[Candidate], ...]]:
    groups = grid_candidates(eta)
    best = None
    for c2, c3, c4 in product(*groups):
        I = c2.fisher + c3.fisher + c4.fisher
        objective = valid_objective(I)
        if best is None or objective < best[0]:
            best = (objective, c2, c3, c4, I)
    if best is None or not np.isfinite(best[0]):
        raise RuntimeError("no full-rank grid battery")
    objective, c2, c3, c4, I = best
    result = {
        "objective": objective,
        "design": {
            "G2": {"pi": c2.design[0], "t": c2.design[1]},
            "G3": {"actions": c3.design[0], "target": c3.design[1]},
            "G4": {
                "values": c4.design[0],
                "bids": c4.design[1],
                "sigma": c4.design[2],
            },
        },
        "games": {
            "G2": fisher_summary(c2.fisher),
            "G3": fisher_summary(c3.fisher),
            "G4": fisher_summary(c4.fisher),
        },
        "battery": fisher_summary(I),
        "sample_size": [sample_sizes(I, d) for d in (0.20, 0.10, 0.05)],
    }
    return result, groups


def continuous_refinement(eta: Array, grid: dict, seed: int) -> dict:
    n_actions = int(grid["design"]["G3"]["actions"])
    n_values = int(grid["design"]["G4"]["values"])
    n_bids = int(grid["design"]["G4"]["bids"])

    def objective(x: Array) -> float:
        pi, t, target, sigma = x
        try:
            I = (
                sr_fisher(eta, pi, t, t)
                + bc_fisher(eta, n_actions, target)
                + auction_fisher(eta, n_values, n_bids, sigma)
            )
            return valid_objective(I)
        except (FloatingPointError, RuntimeError, ValueError):
            return 1e12

    result = differential_evolution(
        objective,
        bounds=((0.55, 0.95), (0.55, 0.97), (0.30, 0.90), (0.03, 0.65)),
        seed=seed,
        popsize=8,
        maxiter=80,
        tol=1e-5,
        polish=True,
        workers=1,
    )
    pi, t, target, sigma = result.x
    I2 = sr_fisher(eta, pi, t, t)
    I3 = bc_fisher(eta, n_actions, target)
    I4 = auction_fisher(eta, n_values, n_bids, sigma)
    I = I2 + I3 + I4
    return {
        "success": bool(result.success),
        "message": str(result.message),
        "evaluations": int(result.nfev),
        "objective": float(result.fun),
        "design": {
            "G2": {"pi": float(pi), "t": float(t)},
            "G3": {"actions": n_actions, "target": float(target)},
            "G4": {
                "values": n_values,
                "bids": n_bids,
                "sigma": float(sigma),
            },
        },
        "games": {
            "G2": fisher_summary(I2),
            "G3": fisher_summary(I3),
            "G4": fisher_summary(I4),
        },
        "battery": fisher_summary(I),
        "sample_size": [sample_sizes(I, d) for d in (0.20, 0.10, 0.05)],
        "branch_audit": auction_multistart(
            eta[0], eta[1], n_values, n_bids, sigma
        ),
    }


def parameter_landscape(design: dict) -> list[dict]:
    rows = []
    for lam, tau in product((0.8, 1.2, 1.8, 2.5, 3.5),
                            (0.75, 1.5, 2.5, 3.5)):
        eta = np.array([lam, 0.20, tau])
        I = (
            sr_fisher(eta, design["G2"]["pi"], design["G2"]["t"],
                      design["G2"]["t"])
            + bc_fisher(eta, design["G3"]["actions"],
                        design["G3"]["target"])
            + auction_fisher(eta, design["G4"]["values"],
                             design["G4"]["bids"], design["G4"]["sigma"])
        )
        rows.append({
            "lambda": lam,
            "chi": 0.20,
            "tau": tau,
            "trace_inverse": valid_objective(I),
            "minimum_eigenvalue": float(np.linalg.eigvalsh(I)[0]),
        })
    return rows


def battery_progression(eta: Array, design: dict) -> list[dict]:
    matrices = {
        "G2": sr_fisher(
            eta, design["G2"]["pi"], design["G2"]["t"],
            design["G2"]["t"]
        ),
        "G3": bc_fisher(
            eta, design["G3"]["actions"], design["G3"]["target"]
        ),
        "G4": auction_fisher(
            eta, design["G4"]["values"], design["G4"]["bids"],
            design["G4"]["sigma"]
        ),
    }
    batteries = (
        ("G2",), ("G3",), ("G4",),
        ("G2", "G3"), ("G2", "G4"), ("G3", "G4"),
        ("G2", "G3", "G4"),
    )
    rows = []
    for games in batteries:
        I = sum((matrices[g] for g in games), start=np.zeros((3, 3)))
        summary = fisher_summary(I)
        rows.append({
            "games": list(games),
            "rank": summary["rank"],
            "minimum_eigenvalue": float(np.linalg.eigvalsh(I)[0]),
            "trace_inverse": (
                summary.get("trace_inverse")
                if summary["rank"] == 3 else None
            ),
        })
    return rows


def chi_landscape(design: dict) -> list[dict]:
    rows = []
    for chi in (0.0, 0.10, 0.20, 0.40, 0.60, 0.80):
        eta = np.array([1.80, chi, 2.50])
        I = (
            sr_fisher(eta, design["G2"]["pi"], design["G2"]["t"],
                      design["G2"]["t"])
            + bc_fisher(eta, design["G3"]["actions"],
                        design["G3"]["target"])
            + auction_fisher(eta, design["G4"]["values"],
                             design["G4"]["bids"], design["G4"]["sigma"])
        )
        rows.append({
            "lambda": 1.80,
            "chi": chi,
            "tau": 2.50,
            "trace_inverse": valid_objective(I),
            "minimum_eigenvalue": float(np.linalg.eigvalsh(I)[0]),
        })
    return rows


def primitive_slices(eta: Array, design: dict) -> dict[str, list[dict]]:
    def evaluate(local_design: dict) -> dict:
        I = (
            sr_fisher(eta, local_design["G2"]["pi"],
                      local_design["G2"]["t"], local_design["G2"]["t"])
            + bc_fisher(eta, local_design["G3"]["actions"],
                        local_design["G3"]["target"])
            + auction_fisher(eta, local_design["G4"]["values"],
                             local_design["G4"]["bids"],
                             local_design["G4"]["sigma"])
        )
        return {
            "trace_inverse": valid_objective(I),
            "minimum_eigenvalue": float(np.linalg.eigvalsh(I)[0]),
        }

    slices: dict[str, list[dict]] = {}
    specifications = {
        "G2_pi": ("G2", "pi", (0.55, 0.65, 0.75, 0.85, 0.95)),
        "G2_t": ("G2", "t", (0.55, 0.65, 0.75, 0.85, 0.97)),
        "G3_target": ("G3", "target", (0.30, 0.40, 0.50, 0.60, 0.75, 0.90)),
        "G4_sigma": ("G4", "sigma", (0.03, 0.05, 0.10, 0.20, 0.40, 0.65)),
    }
    for label, (game, key, values) in specifications.items():
        rows = []
        for value in values:
            local = {g: dict(v) for g, v in design.items()}
            local[game][key] = value
            rows.append({key: value, **evaluate(local)})
        slices[label] = rows
    return slices


def finite_difference_stability(eta: Array, design: dict) -> list[dict]:
    matrices = []
    for eps in (1e-4, 1e-5, 1e-6):
        I = (
            sr_fisher(eta, design["G2"]["pi"], design["G2"]["t"],
                      design["G2"]["t"], eps=eps)
            + bc_fisher(eta, design["G3"]["actions"],
                        design["G3"]["target"], eps=eps)
            + auction_fisher(eta, design["G4"]["values"],
                             design["G4"]["bids"], design["G4"]["sigma"],
                             eps=eps)
        )
        matrices.append((eps, I))
    reference = matrices[1][1]
    norm = np.linalg.norm(reference)
    return [{
        "step": eps,
        "relative_frobenius_difference_from_1e-5": float(
            np.linalg.norm(I - reference) / norm
        ),
        "trace_inverse": valid_objective(I),
        "minimum_eigenvalue": float(np.linalg.eigvalsh(I)[0]),
    } for eps, I in matrices]


def structural_checks(eta: Array, design: dict) -> dict:
    perfect_signal_chi_information = auction_fisher(
        eta, design["G4"]["values"], design["G4"]["bids"], 0.0
    )[1, 1]
    mu_h, mu_l, _, _ = sr_posterior(
        design["G2"]["pi"], design["G2"]["t"], design["G2"]["t"]
    )
    return {
        "G2_posterior_gap": float(mu_h - mu_l),
        "G2_prior_asymmetry": float(2 * design["G2"]["pi"] - 1),
        "G4_perfect_signal_chi_information": float(
            perfect_signal_chi_information
        ),
        "G4_perfect_signal_cursing_collapse": bool(
            abs(perfect_signal_chi_information) < 1e-12
        ),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("design_games_pre_certificate.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-continuous", action="store_true")
    args = parser.parse_args(argv)

    eta = np.array([1.80, 0.20, 2.50])
    grid, _ = best_grid_battery(eta)
    if args.skip_continuous:
        final = grid
        final["branch_audit"] = auction_multistart(
            eta[0], eta[1],
            grid["design"]["G4"]["values"],
            grid["design"]["G4"]["bids"],
            grid["design"]["G4"]["sigma"],
        )
    else:
        final = continuous_refinement(eta, grid, args.seed)

    payload = {
        "release": "pre-certificate",
        "seed": args.seed,
        "parameter_order": ["lambda", "chi", "tau"],
        "local_parameter_point": eta.tolist(),
        "payoff_normalization": "divide each payoff array by its span",
        "grid": grid,
        "refined": final,
        "parameter_landscape": parameter_landscape(final["design"]),
        "chi_landscape": chi_landscape(final["design"]),
        "battery_progression": battery_progression(eta, final["design"]),
        "primitive_slices": primitive_slices(eta, final["design"]),
        "finite_difference_stability": finite_difference_stability(
            eta, final["design"]
        ),
        "structural_checks": structural_checks(eta, final["design"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if final["branch_audit"]["distinct_fixed_points"] != 1:
        raise RuntimeError("G4 branch audit found multiple fixed points")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
