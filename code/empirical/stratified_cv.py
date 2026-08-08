from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fisher_cgc import discretize
from cross_validation import predict_M_Q, predict_M_QL


SHALLOW_TYPES = ['L1', 'D1']
DEEP_TYPES = ['L2', 'L3', 'E', 'D2', 'S']


def per_subject_delta(games: pd.DataFrame, choices: pd.DataFrame,
                       cv: pd.DataFrame, delta: int = 5,
                       K_max: int = 6) -> pd.DataFrame:
    rows = []
    for _, c in cv.iterrows():
        gid = int(c['game_master'])
        g = games[games['game_master'] == gid].iloc[0]
        A_i = discretize(int(g['L_i']), int(g['U_i']), delta)
        A_j = discretize(int(g['L_j']), int(g['U_j']), delta)
        p_i, p_j = float(g['p_i']), float(g['p_j'])

        q_Q  = predict_M_Q(float(c['lam_Q']),  A_i, A_j, p_i, p_j)
        q_QL = predict_M_QL(float(c['lam_QL']), float(c['tau_QL']),
                             A_i, A_j, p_i, p_j, K_max)

        sub_choices = choices[choices['game_master'] == gid]
        for _, sc in sub_choices.iterrows():
            sid = sc['subject_id']
            guess = float(sc['guess'])
            idx = int(np.argmin(np.abs(A_i - guess)))
            log_q_Q  = float(np.log(max(q_Q[idx],  1e-300)))
            log_q_QL = float(np.log(max(q_QL[idx], 1e-300)))
            rows.append(dict(
                game_master=gid,
                subject_id=sid,
                guess=guess,
                log_q_Q=log_q_Q,
                log_q_QL=log_q_QL,
                delta_js=log_q_QL - log_q_Q,
            ))
    return pd.DataFrame(rows)


def annotate_types(d: pd.DataFrame, types_df: pd.DataFrame) -> pd.DataFrame:
    types_df = types_df.copy()
    if 'game_master' not in types_df.columns:
        raise ValueError("fold-specific subject types must include game_master")
    if 'cat' not in types_df.columns:
        types_df['cat'] = types_df['t_hat'].apply(
            lambda t: 'shallow' if t in SHALLOW_TYPES else 'deep'
        )
    if 'w_deep' not in types_df.columns:
        deep_cols = [f'post_{t}' for t in DEEP_TYPES
                     if f'post_{t}' in types_df.columns]
        types_df['w_deep'] = types_df[deep_cols].sum(axis=1)
    key = ['subject_id', 'game_master']
    if types_df.duplicated(key).any():
        raise ValueError("cross-fitted classification has duplicate keys")
    cols_keep = [*key, 't_hat', 'cat', 'w_deep']
    merged = d.merge(types_df[cols_keep], on=key, validate='one_to_one')
    if len(merged) != len(d):
        raise ValueError("cross-fitted classification does not cover every test cell")
    return merged


def hard_aggregate(d: pd.DataFrame) -> pd.DataFrame:
    grp = d.groupby(['game_master', 'cat'])['delta_js'].agg(['sum', 'mean',
                                                              'count']).reset_index()
    return grp.pivot(index='game_master', columns='cat',
                      values='mean').rename(columns={
        'deep': 'delta_deep_hard', 'shallow': 'delta_shallow_hard'
    }).reset_index()


def soft_aggregate(d: pd.DataFrame) -> pd.DataFrame:
    results = []
    for gid, sub in d.groupby('game_master'):
        w_d = sub['w_deep'].values
        w_s = 1.0 - w_d
        delta = sub['delta_js'].values
        denom_d = w_d.sum()
        denom_s = w_s.sum()
        results.append(dict(
            game_master=gid,
            delta_deep_soft=(w_d * delta).sum() / denom_d if denom_d > 0 else np.nan,
            delta_shallow_soft=(w_s * delta).sum() / denom_s if denom_s > 0 else np.nan,
        ))
    return pd.DataFrame(results)


def ols_regression(y: np.ndarray, X: np.ndarray, names: list[str]) -> dict:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    resid = y - y_hat
    n, p = X.shape
    if n <= p:
        return dict(beta=beta, se=None, t=None, r2=None, names=names)
    sigma2 = (resid ** 2).sum() / (n - p)
    cov = sigma2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    t = beta / se
    ss_res = (resid ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    return dict(beta=beta, se=se, t=t, r2=r2, names=names)


def print_ols(out: dict, title: str) -> None:
    print(f"\n  {title}")
    print(f"  {'-' * len(title)}")
    if out['se'] is None:
        print("  (insufficient degrees of freedom)")
        return
    for nm, b, s, t in zip(out['names'], out['beta'], out['se'], out['t']):
        signif = '***' if abs(t) > 2.58 else ('**' if abs(t) > 1.96 else
                                                ('*' if abs(t) > 1.65 else ''))
        print(f"    {nm:<16} = {b:+.4f}  (SE = {s:.4f}, t = {t:+.2f}) {signif}")
    print(f"    R^2              = {out['r2']:.3f}")


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
    parser.add_argument('--output', type=Path,
                        default=Path('data/processed/cgc2006/stratified_cv_crossfit.csv'))
    parser.add_argument('--delta', type=int, default=5)
    parser.add_argument('--K-max', type=int, default=6)
    args = parser.parse_args(argv)

    games = pd.read_csv(args.games)
    choices = pd.read_csv(args.choices)
    cv = pd.read_csv(args.cv)
    types_df = pd.read_csv(args.types)
    fisher = pd.read_csv(args.fisher)

    print(f"Loaded: {len(games)} games, {len(choices)} guesses, "
          f"{len(types_df)} classified subjects.\n")

    print("Categories:")
    print(f"  shallow = {SHALLOW_TYPES}")
    print(f"  deep    = {DEEP_TYPES}")

    n_shallow = (types_df['t_hat'].isin(SHALLOW_TYPES)).sum()
    n_deep = (types_df['t_hat'].isin(DEEP_TYPES)).sum()
    print(f"\n  Fold-specific shallow cells (hard): {n_shallow}")
    print(f"  Fold-specific deep cells    (hard): {n_deep}")

    print("\n[1/3] Compute delta_{j,s} for each (game, subject)...")
    d = per_subject_delta(games, choices, cv, args.delta, args.K_max)
    print(f"  -> {len(d)} rows (1408 expected: 16 games x 88 subjects)")

    d = annotate_types(d, types_df)
    print(f"  Cross-fitted w_deep statistics: mean={d['w_deep'].mean():.3f}, "
          f"median={d['w_deep'].median():.3f}")

    print("\n[2/3] Hard and soft aggregation by game...")
    hard = hard_aggregate(d)
    soft = soft_aggregate(d)
    strat = hard.merge(soft, on='game_master')
    strat['delta_contrast_hard'] = (strat['delta_deep_hard']
                                     - strat['delta_shallow_hard'])
    strat['delta_contrast_soft'] = (strat['delta_deep_soft']
                                     - strat['delta_shallow_soft'])
    fisher_cols = ['game_master', 'tau_informativity', 'se_tau',
                   'I_tau_cond', 'log_I_tau_cond']
    missing = [c for c in fisher_cols if c not in fisher.columns]
    if missing:
        raise ValueError(f"Fisher diagnostics missing columns: {missing}")
    strat = strat.merge(fisher[fisher_cols], on='game_master',
                        validate='one_to_one')
    log_info = strat['log_I_tau_cond'].to_numpy(dtype=float)
    strat['z_log_I_tau_cond'] = ((log_info - log_info.mean()) /
                                  log_info.std(ddof=0))

    rank_map = {'strong': 3, 'medium': 2, 'weak': 1, 'degenerate': 0}
    strat['info_rank'] = strat['tau_informativity'].map(rank_map)

    print("\n  By game (deep versus shallow):")
    print(strat[['game_master', 'tau_informativity',
                  'delta_deep_hard', 'delta_shallow_hard', 'delta_contrast_hard',
                  'delta_deep_soft', 'delta_shallow_soft', 'delta_contrast_soft']]
          .sort_values('game_master').to_string(index=False,
                                                  float_format=lambda x: f"{x:+.4f}"))

    print("\n  Mean by information category:")
    cat_summary = strat.groupby('tau_informativity').agg(
        n=('game_master', 'count'),
        deep_hard_mean=('delta_deep_hard', 'mean'),
        shallow_hard_mean=('delta_shallow_hard', 'mean'),
        contrast_hard_mean=('delta_contrast_hard', 'mean'),
        contrast_soft_mean=('delta_contrast_soft', 'mean'),
    ).reindex(['strong', 'medium', 'weak', 'degenerate']).reset_index()
    print(cat_summary.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

    print("\n[3/3] Directional tests")
    print("=" * 100)
    print("Prediction: for deep subjects, Delta_j increases with information rank")
    print("            for shallow subjects, Delta_j is approximately zero")
    print("            the deep-minus-shallow contrast increases with information rank")
    print("=" * 100)

    X_cont = np.column_stack([np.ones(len(strat)),
                              strat['z_log_I_tau_cond'].values])

    print_ols(ols_regression(strat['delta_deep_hard'].values, X_cont,
                              names=['alpha', 'beta(z log I)']),
              "Delta_deep_HARD ~ z log(I_tau|lambda)")
    print_ols(ols_regression(strat['delta_shallow_hard'].values, X_cont,
                              names=['alpha', 'beta(z log I)']),
              "Delta_shallow_HARD ~ z log(I_tau|lambda)")
    print_ols(ols_regression(strat['delta_contrast_hard'].values, X_cont,
                              names=['alpha', 'beta(z log I)']),
              "HARD contrast ~ z log(I_tau|lambda)")
    print_ols(ols_regression(strat['delta_contrast_soft'].values, X_cont,
                              names=['alpha', 'beta(z log I)']),
              "SOFT contrast ~ z log(I_tau|lambda)")

    print("\n  Legacy categorical analysis")
    X = np.column_stack([np.ones(len(strat)), strat['info_rank'].values])

    print_ols(ols_regression(strat['delta_deep_hard'].values, X,
                              names=['alpha', 'beta(info)']),
              "T1: Delta_deep_HARD ~ info_rank")
    print_ols(ols_regression(strat['delta_shallow_hard'].values, X,
                              names=['alpha', 'beta(info)']),
              "T1b: Delta_shallow_HARD ~ info_rank (control, expected beta ~0)")
    print_ols(ols_regression(strat['delta_contrast_hard'].values, X,
                              names=['alpha', 'beta(info)']),
              "T3: Delta_contrast_HARD (deep - shallow) ~ info_rank")
    print_ols(ols_regression(strat['delta_deep_soft'].values, X,
                              names=['alpha', 'beta(info)']),
              "T2: Delta_deep_SOFT ~ info_rank")
    print_ols(ols_regression(strat['delta_contrast_soft'].values, X,
                              names=['alpha', 'beta(info)']),
              "T4: Delta_contrast_SOFT ~ info_rank")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    strat.to_csv(args.output, index=False)
    print(f"\nResults saved: {args.output}")
    return 0


if __name__ == "__main__":
    main()
