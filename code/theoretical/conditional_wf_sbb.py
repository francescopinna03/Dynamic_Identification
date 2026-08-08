from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import null_space, qr
from scipy.optimize import Bounds, LinearConstraint, minimize


Array = NDArray[np.float64]


@dataclass(frozen=True)
class BridgeConfig:
    n_grid: int = 7
    n_steps: int = 8
    epsilon: float = 0.15
    kappa: float = 0.05
    theta: float = 9.9
    rho: float = 1.0
    cone_lambda: float = 4.0
    entropy_epsilon: float = 5.0e-3
    chi: float = 0.45
    correct_association: float = 0.45
    independent_association: float = 0.36
    initial_center: float = 0.38
    initial_scale: float = 0.105
    terminal_center: float = 0.44
    terminal_scale: float = 0.115
    model_label: str = "generic cursed-association slice"
    lower_bound: float = 1.0e-12
    seed: int = 42


class ConditionalWFSBB:
    def __init__(self, cfg: BridgeConfig):
        self.cfg = cfg
        self.grid = np.linspace(cfg.epsilon, 1.0 - cfg.epsilon, cfg.n_grid)
        self.h = float(self.grid[1] - self.grid[0])
        self.dt = 1.0 / cfg.n_steps
        self.transitions: List[Tuple[int, int]] = []
        self.rows: List[np.ndarray] = []
        self.cols: List[np.ndarray] = []
        for i in range(cfg.n_grid):
            idx = []
            for j in range(max(0, i - 1), min(cfg.n_grid, i + 2)):
                idx.append(len(self.transitions))
                self.transitions.append((i, j))
            self.rows.append(np.asarray(idx, dtype=int))
        for j in range(cfg.n_grid):
            self.cols.append(
                np.asarray([e for e, (_, jj) in enumerate(self.transitions) if jj == j], dtype=int)
            )
        self.edges_per_step = len(self.transitions)
        self.n_var = cfg.n_steps * self.edges_per_step
        self.mu0 = self._discrete_normal(
            center=cfg.initial_center,
            scale=cfg.initial_scale,
        )
        self.muT = self._discrete_normal(
            center=cfg.terminal_center,
            scale=cfg.terminal_scale,
        )
        self.Aeq, self.beq = self._equality_system()
        self.Acone = self._cone_system()
        self.reference_flow = self._reference_bridge(cfg.chi)

    def _discrete_normal(self, center: float, scale: float) -> Array:
        w = np.exp(-0.5 * ((self.grid - center) / scale) ** 2) + 2.0e-3
        return w / w.sum()

    def cursed_anchor(self, chi: float) -> float:
        c = self.cfg.correct_association
        q = self.cfg.independent_association
        return (1.0 - chi) * c + chi * q

    def reference_characteristics(self, chi: float) -> Tuple[Array, Array]:
        target = self.cursed_anchor(chi)
        abar = self.cfg.kappa * self.grid * (1.0 - self.grid)
        bbar = 0.5 * self.cfg.kappa * self.cfg.theta * (target - self.grid)
        return bbar, abar

    def _slice(self, k: int) -> slice:
        lo = k * self.edges_per_step
        return slice(lo, lo + self.edges_per_step)

    def _full_index(self, k: int, edge: int) -> int:
        return k * self.edges_per_step + edge

    def _equality_system(self) -> Tuple[Array, Array]:
        rows: List[Array] = []
        rhs: List[float] = []
        n = self.cfg.n_grid
        K = self.cfg.n_steps

        for i in range(n):
            row = np.zeros(self.n_var)
            for e in self.rows[i]:
                row[self._full_index(0, int(e))] = 1.0
            rows.append(row)
            rhs.append(float(self.mu0[i]))

        for k in range(1, K):
            for i in range(n):
                row = np.zeros(self.n_var)
                for e in self.rows[i]:
                    row[self._full_index(k, int(e))] += 1.0
                for e in self.cols[i]:
                    row[self._full_index(k - 1, int(e))] -= 1.0
                rows.append(row)
                rhs.append(0.0)

        for j in range(n):
            row = np.zeros(self.n_var)
            for e in self.cols[j]:
                row[self._full_index(K - 1, int(e))] = 1.0
            rows.append(row)
            rhs.append(float(self.muT[j]))

        full_A = np.vstack(rows)
        full_b = np.asarray(rhs)
        _, R, piv = qr(full_A.T, mode="economic", pivoting=True)
        diag = np.abs(np.diag(R))
        tol = max(full_A.shape) * np.finfo(float).eps * (diag.max() if diag.size else 1.0)
        rank = int(np.sum(diag > tol))
        keep = np.sort(piv[:rank])
        return full_A[keep], full_b[keep]

    def _cone_system(self) -> Array:
        rows = []
        _, abar = self.reference_characteristics(self.cfg.chi)
        for k in range(self.cfg.n_steps):
            for i in range(self.cfg.n_grid):
                row = np.zeros(self.n_var)
                for e in self.rows[i]:
                    _, j = self.transitions[int(e)]
                    dx = self.grid[j] - self.grid[i]
                    coefficient = dx * dx / self.dt - self.cfg.cone_lambda * abar[i]
                    row[self._full_index(k, int(e))] = coefficient
                rows.append(row)
        return np.vstack(rows)

    def _reference_kernel(self, chi: float) -> Array:
        bbar, abar = self.reference_characteristics(chi)
        n = self.cfg.n_grid
        P = np.zeros((n, n))
        for i in range(n):
            r_plus = max(0.0, (abar[i] + bbar[i] * self.h) / (2.0 * self.h**2))
            r_minus = max(0.0, (abar[i] - bbar[i] * self.h) / (2.0 * self.h**2))
            if i == 0:
                r_plus += r_minus
                r_minus = 0.0
            if i == n - 1:
                r_minus += r_plus
                r_plus = 0.0
            p_plus = self.dt * r_plus
            p_minus = self.dt * r_minus
            if p_plus + p_minus >= 0.92:
                scale = 0.92 / (p_plus + p_minus)
                p_plus *= scale
                p_minus *= scale
            if i > 0:
                P[i, i - 1] = p_minus
            if i < n - 1:
                P[i, i + 1] = p_plus
            P[i, i] = 1.0 - p_plus - p_minus
        P = np.maximum(P, 0.0)
        P /= P.sum(axis=1, keepdims=True)
        return P

    def _reference_bridge(self, chi: float) -> Array:
        P = self._reference_kernel(chi)
        K = self.cfg.n_steps
        endpoint_kernel = np.linalg.matrix_power(P, K)
        if np.any(endpoint_kernel <= 0.0):
            raise RuntimeError("The reference kernel does not connect the endpoint supports")
        v = np.ones_like(self.muT)
        for _ in range(20000):
            u = self.mu0 / (endpoint_kernel @ v)
            v_new = self.muT / (endpoint_kernel.T @ u)
            v_new /= np.exp(np.mean(np.log(v_new)))
            if np.max(np.abs(np.log(v_new / v))) < 2.0e-14:
                v = v_new
                break
            v = v_new
        hfun = [np.empty_like(self.mu0) for _ in range(K + 1)]
        hfun[K] = v
        for k in range(K - 1, -1, -1):
            hfun[k] = P @ hfun[k + 1]

        mu = self.mu0.copy()
        flow = np.zeros(self.n_var)
        for k in range(K):
            controlled = P * hfun[k + 1][None, :] / hfun[k][:, None]
            controlled /= controlled.sum(axis=1, keepdims=True)
            for e, (i, j) in enumerate(self.transitions):
                flow[self._full_index(k, e)] = mu[i] * controlled[i, j]
            mu = mu @ controlled
        if np.max(np.abs(mu - self.muT)) > 5.0e-10:
            raise RuntimeError("Doob initialization failed to match the terminal marginal")
        return np.maximum(flow, self.cfg.lower_bound * 10.0)

    def row_moments(self, z: Array, k: int, i: int) -> Tuple[float, float, float]:
        values = z[self._slice(k)][self.rows[i]]
        r = float(values.sum())
        q = 0.0
        A = 0.0
        for value, e in zip(values, self.rows[i]):
            _, j = self.transitions[int(e)]
            dx = self.grid[j] - self.grid[i]
            q += float(value) * dx / self.dt
            A += float(value) * dx * dx / self.dt
        return r, q, A

    def objective_gradient(self, z: Array, chi: float) -> Tuple[float, Array]:
        bbar, abar = self.reference_characteristics(chi)
        eps_ent = self.cfg.entropy_epsilon
        ref = self.reference_flow
        value = 0.0
        grad = np.zeros_like(z)

        for k in range(self.cfg.n_steps):
            local = z[self._slice(k)]
            for i in range(self.cfg.n_grid):
                edge_ids = self.rows[i]
                gamma = local[edge_ids]
                r = float(gamma.sum())
                dx = np.asarray(
                    [self.grid[self.transitions[int(e)][1]] - self.grid[i] for e in edge_ids]
                )
                q = float(gamma @ (dx / self.dt))
                A = float(gamma @ (dx * dx / self.dt))
                B = r * abar[i]
                A_safe = max(A, 1.0e-18)
                B_safe = max(B, 1.0e-18)
                D = q - r * bbar[i]

                drift = 0.5 * D * D / B_safe
                bures = 0.5 * self.cfg.rho * (
                    A_safe + B_safe - 2.0 * np.sqrt(A_safe * B_safe)
                )
                value += self.dt * (drift + bures)

                f_q = D / B_safe
                f_r = -D * bbar[i] / B_safe - 0.5 * D * D / (r * B_safe)
                f_A = 0.5 * self.cfg.rho * (1.0 - np.sqrt(B_safe / A_safe))
                f_r += 0.5 * self.cfg.rho * abar[i] * (
                    1.0 - np.sqrt(A_safe / B_safe)
                )
                row_grad = f_r + f_q * dx / self.dt + f_A * dx * dx / self.dt
                grad[self._slice(k)][edge_ids] += self.dt * row_grad

        ratio = np.maximum(z, 1.0e-300) / np.maximum(ref, 1.0e-300)
        value += eps_ent * float(np.sum(z * np.log(ratio) - z + ref))
        grad += eps_ent * np.log(ratio)
        return float(value), grad

    def hessian(self, z: Array, chi: float) -> Array:
        bbar, abar = self.reference_characteristics(chi)
        H = np.zeros((self.n_var, self.n_var))
        for k in range(self.cfg.n_steps):
            local = z[self._slice(k)]
            for i in range(self.cfg.n_grid):
                edge_ids = self.rows[i]
                gamma = local[edge_ids]
                r = float(gamma.sum())
                dx = np.asarray(
                    [self.grid[self.transitions[int(e)][1]] - self.grid[i] for e in edge_ids]
                )
                q = float(gamma @ (dx / self.dt))
                A = float(gamma @ (dx * dx / self.dt))
                B = r * abar[i]
                A = max(A, 1.0e-18)
                B = max(B, 1.0e-18)

                base = np.zeros((3, 3))
                base[0, 0] = q * q / (r**3 * abar[i])
                base[0, 1] = base[1, 0] = -q / (r**2 * abar[i])
                base[1, 1] = 1.0 / (r * abar[i])
                base[0, 0] += (
                    0.25
                    * self.cfg.rho
                    * abar[i] ** 2
                    * np.sqrt(A)
                    / B ** 1.5
                )
                base[2, 2] = 0.25 * self.cfg.rho * np.sqrt(B) / A ** 1.5
                base[0, 2] = base[2, 0] = (
                    -0.25 * self.cfg.rho * abar[i] / np.sqrt(A * B)
                )
                jac = np.vstack(
                    [np.ones_like(dx), dx / self.dt, dx * dx / self.dt]
                )
                block = self.dt * (jac.T @ base @ jac)
                ids = [self._full_index(k, int(e)) for e in edge_ids]
                H[np.ix_(ids, ids)] += block
        H[np.diag_indices_from(H)] += self.cfg.entropy_epsilon / np.maximum(z, 1.0e-300)
        return H

    def _active_constraint_matrix(
        self,
        active_bounds: List[int],
        active_cones: List[int],
    ) -> Array:
        blocks = [self.Aeq]
        if active_bounds:
            blocks.append(-np.eye(self.n_var)[active_bounds])
        if active_cones:
            blocks.append(self.Acone[active_cones])
        return np.vstack(blocks)

    def derivative_checks(
        self,
        z: Array,
        chi: float,
        active_bounds: List[int] | None = None,
        active_cones: List[int] | None = None,
    ) -> Dict[str, float]:
        active_bounds = active_bounds or []
        active_cones = active_cones or []
        constraints = self._active_constraint_matrix(active_bounds, active_cones)
        tangent = null_space(constraints)
        rng = np.random.default_rng(self.cfg.seed)
        coordinates = rng.normal(size=tangent.shape[1])
        direction = tangent @ coordinates
        direction /= np.linalg.norm(direction)
        negative = direction < -1.0e-15
        step = 1.0e-6
        if np.any(negative):
            step = min(
                step,
                0.1
                * float(
                    np.min(
                        (z[negative] - self.cfg.lower_bound)
                        / (-direction[negative])
                    )
                ),
            )
        step = max(step, 1.0e-9)
        value, gradient = self.objective_gradient(z, chi)
        plus_value, plus_gradient = self.objective_gradient(z + step * direction, chi)
        minus_value, minus_gradient = self.objective_gradient(z - step * direction, chi)
        finite_gradient = (plus_value - minus_value) / (2.0 * step)
        analytic_gradient = float(gradient @ direction)
        finite_hessian = (plus_gradient - minus_gradient) / (2.0 * step)
        analytic_hessian = self.hessian(z, chi) @ direction
        return {
            "step": float(step),
            "gradient_directional_relative_error": float(
                abs(finite_gradient - analytic_gradient)
                / max(1.0, abs(finite_gradient), abs(analytic_gradient))
            ),
            "hessian_directional_relative_error": float(
                np.linalg.norm(finite_hessian - analytic_hessian)
                / max(1.0, np.linalg.norm(finite_hessian), np.linalg.norm(analytic_hessian))
            ),
            "objective_at_solution": float(value),
        }

    def solve(self, chi: float, start: Array | None = None) -> Dict[str, object]:
        z0 = self._reference_bridge(chi) if start is None else np.asarray(start, dtype=float).copy()
        constraints = [
            LinearConstraint(self.Aeq, self.beq, self.beq),
            LinearConstraint(self.Acone, -np.inf, 0.0),
        ]
        result = minimize(
            lambda z: self.objective_gradient(z, chi),
            z0,
            jac=True,
            method="SLSQP",
            bounds=Bounds(self.cfg.lower_bound, np.inf),
            constraints=constraints,
            options={"ftol": 1.0e-13, "maxiter": 5000, "disp": False},
        )
        z = np.asarray(result.x, dtype=float)

        bound_tol = max(2.0e-11, 20.0 * self.cfg.lower_bound)
        cone_tol = 1.0e-9
        active_bounds = np.where(z <= self.cfg.lower_bound + bound_tol)[0].tolist()
        active_cones = np.where(self.Acone @ z >= -cone_tol)[0].tolist()

        active_matrix = self._active_constraint_matrix(active_bounds, active_cones)
        tangent = null_space(active_matrix)
        active_bound_mask = np.zeros(self.n_var, dtype=bool)
        active_bound_mask[active_bounds] = True
        for _ in range(60):
            value, grad = self.objective_gradient(z, chi)
            reduced_grad = tangent.T @ grad
            if np.max(np.abs(reduced_grad)) < 2.0e-9:
                break
            H = self.hessian(z, chi)
            reduced_H = tangent.T @ H @ tangent
            direction = -tangent @ np.linalg.solve(reduced_H, reduced_grad)
            direction[active_bound_mask] = 0.0
            slope = float(grad @ direction)
            alpha = 1.0
            negative = (direction < -1.0e-14) & ~active_bound_mask
            if np.any(negative):
                alpha = min(
                    alpha,
                    0.98
                    * float(
                        np.min(
                            (z[negative] - self.cfg.lower_bound)
                            / (-direction[negative])
                        )
                    ),
                )
            cone_value = self.Acone @ z
            cone_direction = self.Acone @ direction
            tightening = cone_direction > 1.0e-14
            if np.any(tightening):
                alpha = min(
                    alpha,
                    0.98
                    * float(
                        np.min(
                            -cone_value[tightening]
                            / cone_direction[tightening]
                        )
                    ),
                )
            accepted = False
            for _ in range(70):
                candidate = z + alpha * direction
                if (
                    candidate.min() >= self.cfg.lower_bound - 1.0e-14
                    and np.max(self.Acone @ candidate) < 2.0e-10
                    and self.objective_gradient(candidate, chi)[0]
                    <= value + 1.0e-4 * alpha * slope
                ):
                    z = candidate
                    accepted = True
                    break
                alpha *= 0.5
            if not accepted:
                break

        active_bounds = np.where(z <= self.cfg.lower_bound + bound_tol)[0].tolist()
        active_cones = np.where(self.Acone @ z >= -cone_tol)[0].tolist()
        active_matrix = self._active_constraint_matrix(active_bounds, active_cones)
        tangent = null_space(active_matrix)
        grad = self.objective_gradient(z, chi)[1]
        multipliers = np.linalg.lstsq(active_matrix.T, -grad, rcond=None)[0]
        stationarity = grad + active_matrix.T @ multipliers
        neq = self.Aeq.shape[0]
        nbounds = len(active_bounds)
        bound_multipliers = multipliers[neq : neq + nbounds]
        cone_multipliers = multipliers[neq + nbounds :]
        reduced_H = tangent.T @ self.hessian(z, chi) @ tangent
        min_eigenvalue = float(np.linalg.eigvalsh(reduced_H).min())

        eq_resid = float(np.max(np.abs(self.Aeq @ z - self.beq)))
        cone_resid = float(np.max(self.Acone @ z))
        reduced_residual = float(np.max(np.abs(tangent.T @ grad)))
        multiplier_floor = min(
            float(bound_multipliers.min()) if nbounds else np.inf,
            float(cone_multipliers.min()) if active_cones else np.inf,
        )
        success = bool(
            result.success
            and eq_resid < 2.0e-8
            and cone_resid < 2.0e-8
            and reduced_residual < 1.0e-4
            and multiplier_floor > -2.0e-4
            and min_eigenvalue > 0.0
        )
        return {
            "flow": z,
            "success": success,
            "message": str(result.message),
            "solver": "SLSQP with projected-Newton active-face refinement",
            "iterations": int(result.nit),
            "objective": float(self.objective_gradient(z, chi)[0]),
            "equality_residual": eq_resid,
            "cone_residual": cone_resid,
            "reduced_gradient_residual": reduced_residual,
            "stationarity_residual": float(np.max(np.abs(stationarity))),
            "minimum_flow": float(z.min()),
            "active_bounds": active_bounds,
            "active_cones": active_cones,
            "minimum_active_multiplier": (
                multiplier_floor if np.isfinite(multiplier_floor) else None
            ),
            "reduced_hessian_min_eigenvalue": min_eigenvalue,
        }

    def marginals_and_kernels(self, z: Array) -> Tuple[List[Array], List[Array]]:
        marginals = [self.mu0.copy()]
        kernels: List[Array] = []
        for k in range(self.cfg.n_steps):
            flow = z[self._slice(k)]
            mu = np.zeros(self.cfg.n_grid)
            P = np.zeros((self.cfg.n_grid, self.cfg.n_grid))
            for i in range(self.cfg.n_grid):
                mu[i] = flow[self.rows[i]].sum()
                for e in self.rows[i]:
                    _, j = self.transitions[int(e)]
                    P[i, j] = flow[int(e)] / mu[i]
            kernels.append(P)
            marginals.append(mu @ P)
        return marginals, kernels

    def implicit_derivative(
        self,
        z: Array,
        chi: float,
        active_bounds: List[int] | None = None,
        active_cones: List[int] | None = None,
    ) -> Dict[str, object]:
        active_bounds = active_bounds or []
        active_cones = active_cones or []
        H = self.hessian(z, chi)
        E = self._active_constraint_matrix(active_bounds, active_cones)
        grad = self.objective_gradient(z, chi)[1]
        multiplier = np.linalg.lstsq(E.T, -grad, rcond=None)[0]
        stationarity = grad + E.T @ multiplier
        hchi = 2.0e-5
        cross = (
            self.objective_gradient(z, chi + hchi)[1]
            - self.objective_gradient(z, chi - hchi)[1]
        ) / (2.0 * hchi)
        KKT = np.block(
            [
                [H, E.T],
                [E, np.zeros((E.shape[0], E.shape[0]))],
            ]
        )
        rhs = -np.concatenate([cross, np.zeros(E.shape[0])])
        derivative = np.linalg.solve(KKT, rhs)[: self.n_var]
        tangent = null_space(E)
        reduced = tangent.T @ H @ tangent
        min_eigenvalue = float(np.linalg.eigvalsh(reduced).min())
        neq = self.Aeq.shape[0]
        inequality_multipliers = multiplier[neq:]
        return {
            "flow_derivative": derivative,
            "stationarity_residual": float(np.max(np.abs(stationarity))),
            "constraint_derivative_residual": float(np.max(np.abs(E @ derivative))),
            "kkt_condition_number": float(np.linalg.cond(KKT)),
            "reduced_hessian_min_eigenvalue": min_eigenvalue,
            "active_bound_count": len(active_bounds),
            "active_cone_count": len(active_cones),
            "minimum_active_multiplier": (
                float(inequality_multipliers.min())
                if inequality_multipliers.size
                else None
            ),
        }

    def score_information(self, z: Array, dz: Array) -> Dict[str, float]:
        marginals, kernels = self.marginals_and_kernels(z)
        c = np.zeros(self.cfg.n_grid)
        d = np.zeros(self.cfg.n_grid)
        maximum_row_mean = 0.0
        transition_fisher_sum = 0.0
        for k in range(self.cfg.n_steps):
            flow = z[self._slice(k)]
            dflow = dz[self._slice(k)]
            mu = np.asarray([flow[self.rows[i]].sum() for i in range(self.cfg.n_grid)])
            dmu = np.asarray([dflow[self.rows[i]].sum() for i in range(self.cfg.n_grid)])
            score = np.zeros_like(kernels[k])
            for i in range(self.cfg.n_grid):
                for e in self.rows[i]:
                    _, j = self.transitions[int(e)]
                    score[i, j] = dflow[int(e)] / flow[int(e)] - dmu[i] / mu[i]
                    transition_fisher_sum += flow[int(e)] * score[i, j] ** 2
                maximum_row_mean = max(
                    maximum_row_mean,
                    abs(float(kernels[k][i] @ score[i])),
                )
            c_new = np.zeros_like(c)
            d_new = np.zeros_like(d)
            for i in range(self.cfg.n_grid):
                for j in range(self.cfg.n_grid):
                    p = kernels[k][i, j]
                    if p == 0.0:
                        continue
                    s = score[i, j]
                    c_new[j] += p * (c[i] + marginals[k][i] * s)
                    d_new[j] += p * (d[i] + 2.0 * c[i] * s + marginals[k][i] * s * s)
            c, d = c_new, d_new
        return {
            "score_mean": float(c.sum()),
            "process_fisher": float(d.sum()),
            "transition_fisher_sum": float(transition_fisher_sum),
            "martingale_difference_decomposition_error": float(
                abs(d.sum() - transition_fisher_sum)
            ),
            "maximum_conditional_score_mean": float(maximum_row_mean),
            "endpoint_fisher": 0.0,
        }

    def audit(self) -> Dict[str, object]:
        solved = self.solve(self.cfg.chi)
        if not solved["success"]:
            raise RuntimeError(f"Bridge optimization failed: {solved['message']}")
        z = np.asarray(solved["flow"])
        active_bounds = list(solved["active_bounds"])
        active_cones = list(solved["active_cones"])
        derivative = self.implicit_derivative(
            z,
            self.cfg.chi,
            active_bounds=active_bounds,
            active_cones=active_cones,
        )
        dz = np.asarray(derivative["flow_derivative"])

        fd_step = 2.0e-4
        plus = self.solve(self.cfg.chi + fd_step, start=z)
        minus = self.solve(self.cfg.chi - fd_step, start=z)
        if not plus["success"] or not minus["success"]:
            raise RuntimeError("Finite-difference reoptimization failed")
        active_set_stable = bool(
            plus["active_bounds"] == active_bounds == minus["active_bounds"]
            and plus["active_cones"] == active_cones == minus["active_cones"]
        )
        fd = (np.asarray(plus["flow"]) - np.asarray(minus["flow"])) / (2.0 * fd_step)
        derivative_error = float(np.linalg.norm(dz - fd) / max(1.0, np.linalg.norm(fd)))

        marginals, _ = self.marginals_and_kernels(z)
        endpoint_error = float(np.max(np.abs(marginals[-1] - self.muT)))
        ratios = []
        _, abar = self.reference_characteristics(self.cfg.chi)
        for k in range(self.cfg.n_steps):
            for i in range(self.cfg.n_grid):
                r, _, A = self.row_moments(z, k, i)
                ratios.append(A / (r * abar[i]))
        score = self.score_information(z, dz)
        derivative_check_point = 0.5 * (z + self.reference_flow)
        derivatives = self.derivative_checks(derivative_check_point, self.cfg.chi)
        anchor = self.cursed_anchor(self.cfg.chi)
        inward_margin = self.cfg.theta * min(anchor, 1.0 - anchor) - self.cfg.cone_lambda
        cone_slack = self.cfg.cone_lambda - float(max(ratios))

        checks = {
            "optimization": bool(solved["success"]),
            "exact_endpoint": bool(endpoint_error < 2.0e-8),
            "cone": bool(max(ratios) <= self.cfg.cone_lambda + 2.0e-7),
            "active_set_stable": active_set_stable,
            "strict_complementarity": bool(
                derivative["minimum_active_multiplier"] is None
                or derivative["minimum_active_multiplier"] > 2.0e-6
            ),
            "inward_condition": bool(inward_margin > 0.0),
            "positive_reduced_hessian": bool(
                derivative["reduced_hessian_min_eigenvalue"] > 0.0
            ),
            "implicit_derivative": bool(derivative_error < 2.5e-3),
            "analytic_gradient": bool(
                derivatives["gradient_directional_relative_error"] < 2.0e-7
            ),
            "analytic_hessian": bool(
                derivatives["hessian_directional_relative_error"] < 2.0e-6
            ),
            "score_centered": bool(abs(score["score_mean"]) < 2.0e-7),
            "score_information_decomposition": bool(
                score["martingale_difference_decomposition_error"] < 2.0e-10
            ),
            "process_information_positive": bool(score["process_fisher"] > 0.0),
        }
        return {
            "model": f"conditional one-dimensional WF-SBB {self.cfg.model_label}",
            "config": self.cfg.__dict__,
            "grid": self.grid.tolist(),
            "cursed_anchor": anchor,
            "inward_margin": float(inward_margin),
            "optimization": {k: v for k, v in solved.items() if k != "flow"},
            "endpoint_error": endpoint_error,
            "maximum_relative_covariance": float(max(ratios)),
            "minimum_relative_cone_slack": cone_slack,
            "implicit": {k: v for k, v in derivative.items() if k != "flow_derivative"},
            "implicit_finite_difference_relative_error": derivative_error,
            "analytic_derivative_checks": derivatives,
            "information": score,
            "checks": checks,
            "all_pass": bool(all(checks.values())),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("conditional_wf_sbb.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chi", type=float, default=0.45)
    parser.add_argument("--entropy-epsilon", type=float, default=5.0e-3)
    parser.add_argument("--n-grid", type=int, default=7)
    parser.add_argument("--n-steps", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = BridgeConfig(
        seed=args.seed,
        chi=args.chi,
        entropy_epsilon=args.entropy_epsilon,
        n_grid=args.n_grid,
        n_steps=args.n_steps,
    )
    audit = ConditionalWFSBB(cfg).audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if not audit["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
