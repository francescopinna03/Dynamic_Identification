from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import poisson

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fisher_cgc import discretize, level_k_strategies, composite_QL


def ols_with_intercept(y: np.ndarray, x: np.ndarray) -> dict:
    X = np.column_stack([np.ones(len(y)), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(y) - X.shape[1]
    sigma2 = float(resid @ resid / dof)
    cov = sigma2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else np.nan
    return dict(beta=beta, se=se, t=beta / se, r2=r2)


def predict_M_Q(lam: float, A_i: np.ndarray, A_j: np.ndarray,
                p_i: float, p_j: float) -> np.ndarray:
    q_i, _ = level_k_strategies(lam, A_i, A_j, p_i, p_j, K_max=1)
    return q_i[1]


def predict_M_QL(lam: float, tau: float, A_i: np.ndarray, A_j: np.ndarray,
                  p_i: float, p_j: float, K_max: int = 6) -> np.ndarray:
    return composite_QL(lam, tau, A_i, A_j, p_i, p_j, K_max)


def nll_on_dataset(predict_fn, choices_df: pd.DataFrame,
                    games_df: pd.DataFrame, delta: int) -> float:
    nll = 0.0
    for _, g in games_df.iterrows():
        gid = int(g['game_master'])
        A_i = discretize(int(g['L_i']), int(g['U_i']), delta)
        A_j = discretize(int(g['L_j']), int(g['U_j']), delta)
        q = predict_fn(A_i, A_j, float(g['p_i']), float(g['p_j']))
        guesses = choices_df.loc[choices_df['game_master'] == gid, 'guess'].values
        if len(guesses) == 0:
            continue
        idx = np.argmin(np.abs(guesses[:, None] - A_i[None, :]), axis=1)
        q_at = np.clip(q[idx], 1e-300, None)
        nll -= float(np.sum(np.log(q_at)))
    return nll


def fit_M_Q(train_choices: pd.DataFrame, train_games: pd.DataFrame,
             delta: int = 5) -> tuple[float, float]:
    def f(lam):
        return nll_on_dataset(
            lambda A_i, A_j, p_i, p_j: predict_M_Q(lam, A_i, A_j, p_i, p_j),
            train_choices, train_games, delta,
        )
    res = minimize_scalar(f, bounds=(1e-3, 1.0), method='bounded',
                          options={'xatol': 1e-4})
    return float(res.x), float(res.fun)


def fit_M_QL(train_choices: pd.DataFrame, train_games: pd.DataFrame,
              delta: int = 5, K_max: int = 6,
              x0: tuple[float, float] = (0.02, 1.1)) -> tuple[tuple[float, float], float]:
    def f(eta):
        return nll_on_dataset(
            lambda A_i, A_j, p_i, p_j: predict_M_QL(eta[0], eta[1], A_i, A_j,
                                                    p_i, p_j, K_max),
            train_choices, train_games, delta,
        )
    res = minimize(f, np.array(x0), method='L-BFGS-B',
                   bounds=[(1e-3, 1.0), (0.1, 5.0)],
                   options={'ftol': 1e-8, 'maxiter': 50})
    return (float(res.x[0]), float(res.x[1])), float(res.fun)


def logo_cv(games: pd.DataFrame, choices: pd.DataFrame,
             delta: int = 5, K_max: int = 6) -> pd.DataFrame:
    results = []
    for gid in sorted(games['game_master'].unique()):
        train_games = games[games['game_master'] != gid]
        train_choices = choices[choices['game_master'] != gid]
        test_games = games[games['game_master'] == gid]
        test_choices = choices[choices['game_master'] == gid]

        lam_Q, nll_Q_train = fit_M_Q(train_choices, train_games, delta)
        (lam_QL, tau_QL), nll_QL_train = fit_M_QL(train_choices, train_games,
                                                    delta, K_max)

        nll_Q_out = nll_on_dataset(
            lambda A_i, A_j, p_i, p_j: predict_M_Q(lam_Q, A_i, A_j, p_i, p_j),
            test_choices, test_games, delta,
        )
        nll_QL_out = nll_on_dataset(
            lambda A_i, A_j, p_i, p_j: predict_M_QL(lam_QL, tau_QL, A_i, A_j,
                                                     p_i, p_j, K_max),
            test_choices, test_games, delta,
        )

        n_test = len(test_choices)
        delta_j = nll_Q_out - nll_QL_out
        results.append(dict(
            game_master=gid,
            n_test=n_test,
            lam_Q=lam_Q, lam_QL=lam_QL, tau_QL=tau_QL,
            nll_Q_train=nll_Q_train, nll_QL_train=nll_QL_train,
            nll_Q_out=nll_Q_out, nll_QL_out=nll_QL_out,
            nll_Q_per_obs=nll_Q_out / n_test,
            nll_QL_per_obs=nll_QL_out / n_test,
            delta=delta_j,
            delta_per_obs=delta_j / n_test,
        ))
        print(f"  Game {gid:2d}: lam_Q={lam_Q:.4f}  "
              f"(lam,tau)_QL=({lam_QL:.4f}, {tau_QL:.3f})  "
              f"NLL_Q_out={nll_Q_out:7.1f}  NLL_QL_out={nll_QL_out:7.1f}  "
              f"Delta={delta_j:+7.2f}  per_obs={delta_j/n_test:+.4f}")

    return pd.DataFrame(results)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--games', type=Path,
                        default=Path('data/processed/cgc2006/games.csv'))
    parser.add_argument('--choices', type=Path,
                        default=Path('data/processed/cgc2006/choices.csv'))
    parser.add_argument('--fisher', type=Path,
                        default=Path('data/processed/cgc2006/fisher_diagnostics.csv'))
    parser.add_argument('--output', type=Path,
                        default=Path('data/processed/cgc2006/cv_results.csv'))
    parser.add_argument('--delta', type=int, default=5)
    parser.add_argument('--K-max', type=int, default=6)
    args = parser.parse_args(argv)

    games = pd.read_csv(args.games)
    choices = pd.read_csv(args.choices)
    fisher = pd.read_csv(args.fisher)

    print(f"Dataset: {len(choices)} guesses ({choices['subject_id'].nunique()} "
          f"subjects) over {len(games)} games.\n")

    print("=" * 100)
    print("LEAVE-ONE-GAME-OUT CROSS-VALIDATION")
    print("M_Q (pure L_1, one parameter) vs M_QL (Poisson mixture, two parameters)")
    print("=" * 100)
    cv = logo_cv(games, choices, delta=args.delta, K_max=args.K_max)

    required_fisher = ['game_master', 'I_tau_cond', 'log_I_tau_cond',
                       'se_tau', 'tau_informativity']
    missing = [c for c in required_fisher if c not in fisher.columns]
    if missing:
        raise ValueError(f"Fisher diagnostics missing required columns: {missing}")
    cv = cv.merge(fisher[required_fisher], on='game_master', validate='one_to_one')

    log_info = cv['log_I_tau_cond'].to_numpy(dtype=float)
    if not np.isfinite(log_info).all() or np.std(log_info, ddof=0) == 0:
        raise ValueError("log_I_tau_cond must be finite and nonconstant")
    cv['z_log_I_tau_cond'] = ((log_info - log_info.mean()) /
                              log_info.std(ddof=0))

    print("\n" + "=" * 100)
    print("PRIMARY TEST: Delta_j versus log efficient Fisher information")
    print("=" * 100)

    primary = ols_with_intercept(cv['delta_per_obs'].to_numpy(dtype=float),
                                 cv['z_log_I_tau_cond'].to_numpy(dtype=float))
    print(f"  beta(z log I_tau|lambda) = {primary['beta'][1]:+.6f} "
          f"(SE={primary['se'][1]:.6f}, t={primary['t'][1]:+.3f}, "
          f"R2={primary['r2']:.3f})")

    print("\nSECONDARY DESCRIPTION: legacy categories")
    cat_summary = cv.groupby('tau_informativity').agg(
        n_games=('game_master', 'count'),
        mean_delta=('delta_per_obs', 'mean'),
        median_delta=('delta_per_obs', 'median'),
        std_delta=('delta_per_obs', 'std'),
    ).reindex(['strong', 'medium', 'weak', 'degenerate']).reset_index()
    print(cat_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    delta_strong = cv.loc[cv['tau_informativity'] == 'strong',
                          'delta_per_obs'].values
    delta_degen = cv.loc[cv['tau_informativity'] == 'degenerate',
                         'delta_per_obs'].values
    if len(delta_strong) > 0 and len(delta_degen) > 0:
        print(f"\nStrong Delta (games 1, 10):      "
              f"{delta_strong.mean():+.4f} per obs (n={len(delta_strong)})")
        print(f"Degenerate Delta (games 4, 5):   "
              f"{delta_degen.mean():+.4f} per obs (n={len(delta_degen)})")
        print(f"Difference:                      "
              f"{delta_strong.mean() - delta_degen.mean():+.4f}")

    print("\n" + "=" * 100)
    print("Regression: Delta_j = alpha + beta * informativity_rank")
    print("=" * 100)
    rank_map = {'strong': 3, 'medium': 2, 'weak': 1, 'degenerate': 0}
    cv['info_rank'] = cv['tau_informativity'].map(rank_map)
    legacy = ols_with_intercept(cv['delta_per_obs'].to_numpy(dtype=float),
                                cv['info_rank'].to_numpy(dtype=float))
    print(f"  alpha (intercept) = {legacy['beta'][0]:+.4f}  "
          f"(SE = {legacy['se'][0]:.4f})")
    print(f"  beta  (slope)     = {legacy['beta'][1]:+.4f}  "
          f"(SE = {legacy['se'][1]:.4f}, t = {legacy['t'][1]:+.2f})")
    print(f"  R^2 = {legacy['r2']:.3f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv.to_csv(args.output, index=False)
    print(f"\nResults saved: {args.output}")

    return 0


if __name__ == "__main__":
    main()
