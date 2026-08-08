from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stratified_cv import (
    per_subject_delta, annotate_types, hard_aggregate, soft_aggregate,
    SHALLOW_TYPES, DEEP_TYPES,
)


def beta_contrast_hard(d_annotated: pd.DataFrame,
                        fisher: pd.DataFrame) -> tuple[float, float]:
    hard = hard_aggregate(d_annotated)
    soft = soft_aggregate(d_annotated)
    strat = hard.merge(soft, on='game_master')
    strat['delta_contrast_hard'] = (strat['delta_deep_hard']
                                     - strat['delta_shallow_hard'])
    strat['delta_contrast_soft'] = (strat['delta_deep_soft']
                                     - strat['delta_shallow_soft'])
    strat = strat.merge(fisher[['game_master', 'log_I_tau_cond']],
                         on='game_master', validate='one_to_one')
    x_raw = strat['log_I_tau_cond'].to_numpy(dtype=float)
    strat['info_cont'] = (x_raw - x_raw.mean()) / x_raw.std(ddof=0)

    def beta(y_col):
        y = strat[y_col].values
        x = strat['info_cont'].values
        X = np.column_stack([np.ones(len(y)), x])
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        return float(b[1])

    return beta('delta_contrast_hard'), beta('delta_contrast_soft')


def permutation_test(d: pd.DataFrame, types_df: pd.DataFrame,
                      fisher: pd.DataFrame,
                      n_perm: int = 2000,
                      seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)

    d_obs = annotate_types(d.copy(), types_df)
    beta_obs_hard, beta_obs_soft = beta_contrast_hard(d_obs, fisher)
    print(f"  Observed beta (HARD): {beta_obs_hard:+.4f}")
    print(f"  Observed beta (SOFT): {beta_obs_soft:+.4f}")

    key = ['subject_id', 'game_master']
    if types_df.duplicated(key).any():
        raise ValueError("duplicate cross-fitted profiles")
    subjects = np.array(sorted(types_df['subject_id'].unique()))

    null_hard = np.empty(n_perm)
    null_soft = np.empty(n_perm)
    for k in range(n_perm):
        donors = rng.permutation(subjects)
        donor_to_target = dict(zip(donors, subjects))
        types_perm = types_df.copy()
        types_perm['subject_id'] = types_perm['subject_id'].map(donor_to_target)
        d_perm = annotate_types(d.copy(), types_perm)
        bh, bs = beta_contrast_hard(d_perm, fisher)
        null_hard[k] = bh
        null_soft[k] = bs
        if (k + 1) % 500 == 0:
            print(f"    {k + 1}/{n_perm} permutations completed", flush=True)

    p_hard = float((1 + np.sum(np.abs(null_hard) >= abs(beta_obs_hard))) /
                   (n_perm + 1))
    p_soft = float((1 + np.sum(np.abs(null_soft) >= abs(beta_obs_soft))) /
                   (n_perm + 1))
    p_hard_1sided = float((1 + np.sum(null_hard >= beta_obs_hard)) /
                          (n_perm + 1))
    p_soft_1sided = float((1 + np.sum(null_soft >= beta_obs_soft)) /
                          (n_perm + 1))

    return dict(
        beta_obs_hard=beta_obs_hard, beta_obs_soft=beta_obs_soft,
        null_hard_mean=float(null_hard.mean()),
        null_soft_mean=float(null_soft.mean()),
        null_hard_std=float(null_hard.std()),
        null_soft_std=float(null_soft.std()),
        p_hard_2sided=p_hard, p_soft_2sided=p_soft,
        p_hard_1sided=p_hard_1sided, p_soft_1sided=p_soft_1sided,
        n_perm=n_perm,
        null_hard=null_hard, null_soft=null_soft,
    )


def rank_correlations(d: pd.DataFrame, types_df: pd.DataFrame,
                       fisher: pd.DataFrame) -> dict:
    d_obs = annotate_types(d.copy(), types_df)
    hard = hard_aggregate(d_obs)
    soft = soft_aggregate(d_obs)
    strat = hard.merge(soft, on='game_master')
    strat['delta_contrast_hard'] = (strat['delta_deep_hard']
                                     - strat['delta_shallow_hard'])
    strat['delta_contrast_soft'] = (strat['delta_deep_soft']
                                     - strat['delta_shallow_soft'])
    strat = strat.merge(fisher[['game_master', 'tau_informativity', 'se_tau',
                                'log_I_tau_cond']], on='game_master')
    rank_map = {'strong': 3, 'medium': 2, 'weak': 1, 'degenerate': 0}
    strat['info_rank'] = strat['tau_informativity'].map(rank_map)

    out = {}
    rho_h0, p_h0 = spearmanr(strat['log_I_tau_cond'].values,
                              strat['delta_contrast_hard'].values)
    rho_s0, p_s0 = spearmanr(strat['log_I_tau_cond'].values,
                              strat['delta_contrast_soft'].values)
    out['spearman_log_info_hard'] = (float(rho_h0), float(p_h0))
    out['spearman_log_info_soft'] = (float(rho_s0), float(p_s0))
    rho_h, p_h = spearmanr(strat['info_rank'].values,
                            strat['delta_contrast_hard'].values)
    rho_s, p_s = spearmanr(strat['info_rank'].values,
                            strat['delta_contrast_soft'].values)
    out['spearman_hard'] = (float(rho_h), float(p_h))
    out['spearman_soft'] = (float(rho_s), float(p_s))

    tau_h, pk_h = kendalltau(strat['info_rank'].values,
                              strat['delta_contrast_hard'].values)
    tau_s, pk_s = kendalltau(strat['info_rank'].values,
                              strat['delta_contrast_soft'].values)
    out['kendall_hard'] = (float(tau_h), float(pk_h))
    out['kendall_soft'] = (float(tau_s), float(pk_s))

    rho_h2, p_h2 = spearmanr(-strat['se_tau'].values,
                              strat['delta_contrast_hard'].values)
    rho_s2, p_s2 = spearmanr(-strat['se_tau'].values,
                              strat['delta_contrast_soft'].values)
    out['spearman_se_tau_hard'] = (float(rho_h2), float(p_h2))
    out['spearman_se_tau_soft'] = (float(rho_s2), float(p_s2))

    return out


def leave_out_game_test(d: pd.DataFrame, types_df: pd.DataFrame,
                         fisher: pd.DataFrame, game_to_drop: int) -> dict:
    d_drop = d[d['game_master'] != game_to_drop].copy()
    fisher_drop = fisher[fisher['game_master'] != game_to_drop].copy()
    d_obs = annotate_types(d_drop, types_df)

    hard = hard_aggregate(d_obs)
    soft = soft_aggregate(d_obs)
    strat = hard.merge(soft, on='game_master')
    strat['delta_contrast_hard'] = (strat['delta_deep_hard']
                                     - strat['delta_shallow_hard'])
    strat['delta_contrast_soft'] = (strat['delta_deep_soft']
                                     - strat['delta_shallow_soft'])
    strat = strat.merge(fisher_drop[['game_master', 'log_I_tau_cond']],
                         on='game_master')
    x_raw = strat['log_I_tau_cond'].to_numpy(dtype=float)
    strat['info_cont'] = (x_raw - x_raw.mean()) / x_raw.std(ddof=0)

    def ols(y_col):
        y = strat[y_col].values
        x = strat['info_cont'].values
        X = np.column_stack([np.ones(len(y)), x])
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        r = y - X @ b
        n, p = X.shape
        sig2 = (r ** 2).sum() / (n - p)
        se = np.sqrt(np.diag(sig2 * np.linalg.inv(X.T @ X)))
        ss_res = (r ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
        return float(b[1]), float(se[1]), float(b[1] / se[1]), float(r2)

    bh, seh, th, r2h = ols('delta_contrast_hard')
    bs, ses, ts, r2s = ols('delta_contrast_soft')
    return dict(
        beta_hard=bh, se_hard=seh, t_hard=th, r2_hard=r2h,
        beta_soft=bs, se_soft=ses, t_soft=ts, r2_soft=r2s,
        n_games=len(strat),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--games', type=Path,
                        default=Path('data/processed/cgc2006/games.csv'))
    parser.add_argument('--choices', type=Path,
                        default=Path('data/processed/cgc2006/choices.csv'))
    parser.add_argument('--cv', type=Path,
                        default=Path('data/processed/cgc2006/cv_results.csv'))
    parser.add_argument('--types', type=Path,
                        default=Path('data/processed/cgc2006/subject_types_crossfit.csv'))
    parser.add_argument('--fisher', type=Path,
                        default=Path('data/processed/cgc2006/fisher_diagnostics.csv'))
    parser.add_argument('--n-perm', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--delta', type=int, default=5)
    parser.add_argument('--K-max', type=int, default=6)
    args = parser.parse_args(argv)

    games = pd.read_csv(args.games)
    choices = pd.read_csv(args.choices)
    cv = pd.read_csv(args.cv)
    types_df = pd.read_csv(args.types)
    fisher = pd.read_csv(args.fisher)

    print("Compute delta_{j,s} for each (game, subject)... ", end='', flush=True)
    d = per_subject_delta(games, choices, cv, args.delta, args.K_max)
    print(f"OK ({len(d)} rows)\n")

    print("=" * 80)
    print(f"1. PERMUTATION TEST (n_perm = {args.n_perm})")
    print("=" * 80)
    perm = permutation_test(d, types_df, fisher, n_perm=args.n_perm,
                             seed=args.seed)
    print("\n  Results:")
    print(f"    Observed beta HARD: {perm['beta_obs_hard']:+.4f}")
    print(f"    Null mean HARD:      {perm['null_hard_mean']:+.4f} "
          f"+/- {perm['null_hard_std']:.4f}")
    print(f"    p-value (2-sided):   {perm['p_hard_2sided']:.4f}")
    print(f"    p-value (1-sided):   {perm['p_hard_1sided']:.4f}")
    print(f"\n    Observed beta SOFT: {perm['beta_obs_soft']:+.4f}")
    print(f"    Null mean SOFT:      {perm['null_soft_mean']:+.4f} "
          f"+/- {perm['null_soft_std']:.4f}")
    print(f"    p-value (2-sided):   {perm['p_soft_2sided']:.4f}")
    print(f"    p-value (1-sided):   {perm['p_soft_1sided']:.4f}")

    print("\n" + "=" * 80)
    print("2. RANK CORRELATIONS (Spearman, Kendall)")
    print("=" * 80)
    rc = rank_correlations(d, types_df, fisher)
    print("\n  Continuous log(I_tau|lambda), primary analysis:")
    print(f"    Spearman rho HARD: {rc['spearman_log_info_hard'][0]:+.3f}  "
          f"(p = {rc['spearman_log_info_hard'][1]:.4f})")
    print(f"    Spearman rho SOFT: {rc['spearman_log_info_soft'][0]:+.3f}  "
          f"(p = {rc['spearman_log_info_soft'][1]:.4f})")
    print(f"\n  Categorical info_rank (strong=3, medium=2, weak=1, degenerate=0):")
    print(f"    Spearman rho HARD: {rc['spearman_hard'][0]:+.3f}  "
          f"(p = {rc['spearman_hard'][1]:.4f})")
    print(f"    Spearman rho SOFT: {rc['spearman_soft'][0]:+.3f}  "
          f"(p = {rc['spearman_soft'][1]:.4f})")
    print(f"    Kendall tau HARD:  {rc['kendall_hard'][0]:+.3f}  "
          f"(p = {rc['kendall_hard'][1]:.4f})")
    print(f"    Kendall tau SOFT:  {rc['kendall_soft'][0]:+.3f}  "
          f"(p = {rc['kendall_soft'][1]:.4f})")
    print(f"\n  Continuous -se_tau (high info = low se_tau):")
    print(f"    Spearman rho HARD: {rc['spearman_se_tau_hard'][0]:+.3f}  "
          f"(p = {rc['spearman_se_tau_hard'][1]:.4f})")
    print(f"    Spearman rho SOFT: {rc['spearman_se_tau_soft'][0]:+.3f}  "
          f"(p = {rc['spearman_se_tau_soft'][1]:.4f})")

    print("\n" + "=" * 80)
    print("3. LEAVE-OUT GAME 4 (degenerate with a large positive Delta)")
    print("=" * 80)
    lo4 = leave_out_game_test(d, types_df, fisher, game_to_drop=4)
    print(f"\n  Without game 4, n games = {lo4['n_games']}:")
    print(f"    Beta HARD = {lo4['beta_hard']:+.4f} (SE = {lo4['se_hard']:.4f}, "
          f"t = {lo4['t_hard']:+.2f}, R^2 = {lo4['r2_hard']:.3f})")
    print(f"    Beta SOFT = {lo4['beta_soft']:+.4f} (SE = {lo4['se_soft']:.4f}, "
          f"t = {lo4['t_soft']:+.2f}, R^2 = {lo4['r2_soft']:.3f})")

    print("\n  Symmetric check: leave out game 10 (strong outlier)")
    lo10 = leave_out_game_test(d, types_df, fisher, game_to_drop=10)
    print(f"    Beta HARD = {lo10['beta_hard']:+.4f} (SE = {lo10['se_hard']:.4f}, "
          f"t = {lo10['t_hard']:+.2f}, R^2 = {lo10['r2_hard']:.3f})")
    print(f"    Beta SOFT = {lo10['beta_soft']:+.4f} (SE = {lo10['se_soft']:.4f}, "
          f"t = {lo10['t_soft']:+.2f}, R^2 = {lo10['r2_soft']:.3f})")

    print("\n" + "=" * 80)
    print("ROBUSTNESS SUMMARY")
    print("=" * 80)
    print(f"\n  Baseline (OLS, n=16):  beta_HARD = {perm['beta_obs_hard']:+.4f}")
    print(f"  Permutation p-value (1-sided):     {perm['p_hard_1sided']:.4f}")
    print(f"  Spearman rho (categorical):        {rc['spearman_hard'][0]:+.3f}  "
          f"(p = {rc['spearman_hard'][1]:.4f})")
    print(f"  Spearman rho (continuous -se_tau): {rc['spearman_se_tau_hard'][0]:+.3f}  "
          f"(p = {rc['spearman_se_tau_hard'][1]:.4f})")
    print(f"  Leave-out game 4:  beta = {lo4['beta_hard']:+.4f} "
          f"(t = {lo4['t_hard']:+.2f})")
    print(f"  Leave-out game 10: beta = {lo10['beta_hard']:+.4f} "
          f"(t = {lo10['t_hard']:+.2f})")

    return 0


if __name__ == "__main__":
    main()
