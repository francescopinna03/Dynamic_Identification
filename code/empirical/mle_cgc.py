from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fisher_cgc import discretize, composite_QL


def neg_log_likelihood(eta: np.ndarray, choices: pd.DataFrame,
                       games: pd.DataFrame,
                       delta: int = 5, K_max: int = 6) -> float:
    lam, tau = float(eta[0]), float(eta[1])
    if lam <= 0 or tau <= 0:
        return 1e10

    nll = 0.0
    for _, g in games.iterrows():
        game_id = int(g['game_master'])
        A_i = discretize(int(g['L_i']), int(g['U_i']), delta)
        A_j = discretize(int(g['L_j']), int(g['U_j']), delta)
        try:
            q = composite_QL(lam, tau, A_i, A_j,
                              float(g['p_i']), float(g['p_j']),
                              K_max=K_max)
        except Exception:
            return 1e10

        g_guesses = choices.loc[choices['game_master'] == game_id, 'guess'].values
        if len(g_guesses) == 0:
            continue

        idx = np.argmin(np.abs(g_guesses[:, None] - A_i[None, :]), axis=1)
        q_at_choices = np.clip(q[idx], 1e-300, None)
        nll -= float(np.sum(np.log(q_at_choices)))

    return nll


def fit_mle(choices: pd.DataFrame, games: pd.DataFrame,
             init_lambda: float = 0.05, init_tau: float = 1.5,
             delta: int = 5, K_max: int = 6) -> dict:
    bounds = [(1e-4, 5.0), (0.1, 5.0)]
    x0 = np.array([init_lambda, init_tau])

    print(f"  Init: lambda = {init_lambda}, tau = {init_tau}")
    print(f"  Bounds: lambda in [1e-4, 5], tau in [0.1, 5]")
    print(f"  Granularity: delta = {delta}, K_max = {K_max}")
    print()

    result = minimize(
        neg_log_likelihood, x0, args=(choices, games, delta, K_max),
        method='L-BFGS-B', bounds=bounds,
        options={'maxiter': 50, 'ftol': 1e-7},
    )

    lam_hat, tau_hat = result.x
    nll_hat = result.fun
    print(f"\n  Convergence: {result.success}, nit = {result.nit}")
    print(f"  lambda_hat = {lam_hat:.5f}")
    print(f"  tau_hat    = {tau_hat:.5f}")
    print(f"  log-lik    = {-nll_hat:.2f}")

    if result.success and hasattr(result, 'hess_inv'):
        try:
            cov = result.hess_inv.todense()
            se_lam = float(np.sqrt(cov[0, 0]))
            se_tau = float(np.sqrt(cov[1, 1]))
            print(f"  SE(lambda) ~ {se_lam:.5f}  (from the inverse L-BFGS-B Hessian)")
            print(f"  SE(tau)    ~ {se_tau:.5f}")
        except Exception as e:
            cov = None
            se_lam, se_tau = None, None
            print(f"  Warning: standard errors unavailable ({e})")
    else:
        cov = None
        se_lam, se_tau = None, None

    return dict(
        lambda_hat=lam_hat,
        tau_hat=tau_hat,
        nll=nll_hat,
        loglik=-nll_hat,
        cov=cov,
        se_lambda=se_lam,
        se_tau=se_tau,
        n_obs=len(choices),
        success=result.success,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--games', type=Path,
                        default=Path('data/processed/cgc2006/games.csv'))
    parser.add_argument('--choices', type=Path,
                        default=Path('data/processed/cgc2006/choices.csv'))
    parser.add_argument('--treatment', type=str, default='all',
                        choices=['all', 'Baseline', 'OB'],
                        help='Filter by treatment (default: all)')
    parser.add_argument('--delta', type=int, default=5)
    parser.add_argument('--K-max', type=int, default=6)
    parser.add_argument('--output', type=Path,
                        default=Path('data/processed/cgc2006/mle_pooled.json'))
    args = parser.parse_args(argv)

    games = pd.read_csv(args.games)
    choices = pd.read_csv(args.choices)

    if args.treatment != 'all':
        n_before = len(choices)
        choices = choices[choices['treatment'] == args.treatment]
        print(f"Treatment filter={args.treatment}: {n_before} -> {len(choices)} observations")

    print(f"Dataset: {len(choices)} guesses ({choices['subject_id'].nunique()} "
          f"subjects) over {len(games)} games.\n")

    print("=" * 70)
    print(f"MLE pooled (Q, L) -- treatment = {args.treatment}")
    print("=" * 70)

    result = fit_mle(choices, games,
                      init_lambda=0.05, init_tau=1.5,
                      delta=args.delta, K_max=args.K_max)

    print("\n" + "=" * 70)
    print("Final result")
    print("=" * 70)
    print(f"  N observations: {result['n_obs']}")
    print(f"  lambda_hat    = {result['lambda_hat']:.5f}")
    print(f"  tau_hat       = {result['tau_hat']:.5f}")
    print(f"  log-likelihood = {result['loglik']:.2f}")
    if result['se_lambda'] is not None:
        print(f"  SE(lambda)    ~ {result['se_lambda']:.5f}")
        print(f"  SE(tau)       ~ {result['se_tau']:.5f}")

    payload = dict(result)
    if payload['cov'] is not None:
        payload['cov'] = np.asarray(payload['cov']).tolist()
    for key, value in list(payload.items()):
        if isinstance(value, np.generic):
            payload[key] = value.item()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding='utf-8')
    print(f"  JSON saved: {args.output}")

    return 0


if __name__ == "__main__":
    main()
