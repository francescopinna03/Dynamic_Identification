from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson


def discretize(L: int, U: int, delta: int = 5) -> np.ndarray:
    return np.arange(L, U + 1, delta, dtype=float)


def payoff_matrix(A_i: np.ndarray, A_j: np.ndarray, p_i: float) -> np.ndarray:
    g_i_grid = A_i[:, None]
    g_j_grid = A_j[None, :]
    d = np.abs(g_i_grid - p_i * g_j_grid)
    return np.maximum(0.0, 200.0 - d) + np.maximum(0.0, 100.0 - d / 10.0)


def softmax(logits: np.ndarray, lam: float) -> np.ndarray:
    z = lam * logits
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def level_k_strategies(lam: float, A_i: np.ndarray, A_j: np.ndarray,
                        p_i: float, p_j: float,
                        K_max: int = 6) -> tuple[list[np.ndarray], list[np.ndarray]]:
    U_i = payoff_matrix(A_i, A_j, p_i)
    U_j = payoff_matrix(A_j, A_i, p_j)

    q_i = [np.ones(len(A_i)) / len(A_i)]
    q_j = [np.ones(len(A_j)) / len(A_j)]

    for k in range(1, K_max + 1):
        exp_pay_i = U_i @ q_j[k - 1]
        exp_pay_j = U_j @ q_i[k - 1]
        q_i.append(softmax(exp_pay_i, lam))
        q_j.append(softmax(exp_pay_j, lam))

    return q_i, q_j


def composite_QL(lam: float, tau: float, A_i: np.ndarray, A_j: np.ndarray,
                  p_i: float, p_j: float, K_max: int = 6) -> np.ndarray:
    q_i_list, _ = level_k_strategies(lam, A_i, A_j, p_i, p_j, K_max)
    weights = poisson.pmf(np.arange(1, K_max + 1), mu=tau)
    weights = weights / weights.sum()
    return sum(w * q for w, q in zip(weights, q_i_list[1:]))


def fisher_2x2(lam0: float, tau0: float,
                A_i: np.ndarray, A_j: np.ndarray,
                p_i: float, p_j: float,
                K_max: int = 6,
                eps_rel: float = 1e-3) -> tuple[np.ndarray, np.ndarray]:
    def q_at(eta):
        return composite_QL(eta[0], eta[1], A_i, A_j, p_i, p_j, K_max)

    eta_0 = np.array([lam0, tau0])
    q0 = q_at(eta_0)

    dq = np.zeros((2, len(A_i)))
    for i in range(2):
        h = eps_rel * max(abs(eta_0[i]), 1.0)
        plus = eta_0.copy(); plus[i] += h
        minus = eta_0.copy(); minus[i] -= h
        dq[i] = (q_at(plus) - q_at(minus)) / (2.0 * h)

    I = np.zeros((2, 2))
    mask = q0 > 1e-12
    for a in range(2):
        for b in range(2):
            I[a, b] = np.sum(dq[a, mask] * dq[b, mask] / q0[mask])

    return I, q0


def legacy_tau_category(se_tau: float) -> str:
    if not np.isfinite(se_tau) or se_tau >= 50.0:
        return 'degenerate'
    if se_tau < 7.0:
        return 'strong'
    if se_tau < 20.0:
        return 'medium'
    return 'weak'


def diagnose(I: np.ndarray) -> dict:
    eigvals, eigvecs = np.linalg.eigh(I)
    lam_min = float(eigvals[0])
    lam_max = float(eigvals[1])
    thresh = 1e-10 * max(abs(lam_max), 1.0)
    rank = int(np.sum(eigvals > thresh))
    if lam_min > thresh:
        cond = lam_max / lam_min
        I_inv = np.linalg.inv(I)
        trace_inv = float(np.trace(I_inv))
        se_lam = float(np.sqrt(I_inv[0, 0]))
        se_tau = float(np.sqrt(I_inv[1, 1]))
    else:
        cond = np.inf
        trace_inv = np.inf
        se_lam = np.inf
        se_tau = np.inf

    I_ll, I_lt, I_tt = float(I[0, 0]), float(I[0, 1]), float(I[1, 1])
    if I_ll > thresh:
        I_tau_cond = max(I_tt - I_lt * I_lt / I_ll, 0.0)
    else:
        I_tau_cond = 0.0
    log_I_tau_cond = (float(np.log(I_tau_cond))
                      if I_tau_cond > 0.0 else -np.inf)
    if np.isfinite(se_tau) and I_tau_cond > 0.0:
        identity_error = abs(se_tau - 1.0 / np.sqrt(I_tau_cond))
    else:
        identity_error = 0.0

    weak_vec = eigvecs[:, 0]
    weak_lam_share = float(weak_vec[0] ** 2)
    weak_tau_share = float(weak_vec[1] ** 2)

    if weak_lam_share > 0.75:
        weak_label = 'lambda'
    elif weak_tau_share > 0.75:
        weak_label = 'tau'
    else:
        weak_label = 'mixed'

    return dict(
        rank=rank,
        lam_min=lam_min,
        lam_max=lam_max,
        cond=cond,
        trace_inv=trace_inv,
        se_lambda=se_lam,
        se_tau=se_tau,
        I_tau_cond=I_tau_cond,
        log_I_tau_cond=log_I_tau_cond,
        tau_informativity=legacy_tau_category(se_tau),
        tau_profile_identity_error=float(identity_error),
        weak_channel=weak_label,
        weak_share_lambda=weak_lam_share,
        weak_share_tau=weak_tau_share,
    )


def fisher_per_game(games: pd.DataFrame, lam0: float, tau0: float,
                     delta: int = 5, K_max: int = 6) -> tuple[pd.DataFrame, dict]:
    rows = []
    matrices = {}
    for _, g in games.iterrows():
        game_id = int(g['game_master'])
        A_i = discretize(int(g['L_i']), int(g['U_i']), delta)
        A_j = discretize(int(g['L_j']), int(g['U_j']), delta)
        I, q0 = fisher_2x2(lam0, tau0, A_i, A_j,
                            float(g['p_i']), float(g['p_j']),
                            K_max=K_max)
        diag = diagnose(I)
        row = dict(
            game_master=game_id,
            L_i=int(g['L_i']), U_i=int(g['U_i']), p_i=float(g['p_i']),
            L_j=int(g['L_j']), U_j=int(g['U_j']), p_j=float(g['p_j']),
            n_actions_i=len(A_i), n_actions_j=len(A_j),
            I_ll=float(I[0, 0]), I_lt=float(I[0, 1]), I_tt=float(I[1, 1]),
            **diag,
        )
        rows.append(row)
        matrices[game_id] = I
    return pd.DataFrame(rows), matrices


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--games', type=Path,
                        default=Path('data/processed/cgc2006/games.csv'),
                        help='games.csv produced by preprocess.py')
    parser.add_argument('--output', type=Path,
                        default=Path('data/processed/cgc2006/fisher_diagnostics.csv'),
                        help='Output diagnostics table')
    parser.add_argument('--lambda0', type=float, default=0.01,
                        help='Evaluation value for lambda (default 0.01)')
    parser.add_argument('--tau0', type=float, default=1.5,
                        help='Evaluation value for tau (default 1.5)')
    parser.add_argument('--delta', type=int, default=5,
                        help='Choice-set granularity (default 5)')
    parser.add_argument('--K-max', type=int, default=6,
                        help='Maximum cognitive level (default 6)')
    args = parser.parse_args(argv)

    print(f"Input games: {args.games}")
    print(f"Output:      {args.output}")
    print(f"Parameters:  lambda_0 = {args.lambda0}, tau_0 = {args.tau0}, "
          f"delta = {args.delta}, K_max = {args.K_max}\n")

    games = pd.read_csv(args.games)
    print(f"Loaded {len(games)} games.\n")

    df, mats = fisher_per_game(games, args.lambda0, args.tau0,
                                delta=args.delta, K_max=args.K_max)

    print("=" * 100)
    print("FISHER (lambda, tau) BY GAME -- submodel (Q, L), chi = 0")
    print("=" * 100)
    cols_print = ['game_master', 'p_i', 'p_j',
                  'I_ll', 'I_lt', 'I_tt',
                  'lam_min', 'lam_max', 'cond', 'trace_inv',
                  'se_lambda', 'se_tau', 'I_tau_cond', 'log_I_tau_cond',
                  'tau_informativity', 'weak_channel']
    print(df[cols_print].to_string(index=False,
                                    float_format=lambda x: f"{x:.4g}"))

    I_total = sum(mats.values())
    diag_tot = diagnose(I_total)
    print("\n" + "=" * 100)
    print("COMPOSITE FISHER INFORMATION (sum over 16 games)")
    print("=" * 100)
    print("Composite information matrix:")
    print(I_total)
    print(f"\nEigenvalues:    [{diag_tot['lam_min']:.4f}, {diag_tot['lam_max']:.4f}]")
    print(f"Cond.number:   {diag_tot['cond']:.2f}")
    print(f"tr(I^-1):      {diag_tot['trace_inv']:.4f}")
    print(f"SE(lambda):    {diag_tot['se_lambda']:.4f}")
    print(f"SE(tau):       {diag_tot['se_tau']:.4f}")
    print(f"Weak channel:  {diag_tot['weak_channel']} "
          f"(lambda-share: {diag_tot['weak_share_lambda']:.2f}, "
          f"tau-share: {diag_tot['weak_share_tau']:.2f})")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nTable saved: {args.output}")

    return 0


if __name__ == "__main__":
    main()
