from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fisher_cgc import discretize, payoff_matrix


BASIC_TYPES = ['E', 'L1', 'L2', 'L3', 'D1', 'D2']
ALL_TYPES = BASIC_TYPES + ['S']
SHALLOW_TYPES = {'L1', 'D1'}
DEEP_TYPES = {'L2', 'L3', 'E', 'D2', 'S'}


def log_softmax_utility(util: np.ndarray, lam: float) -> np.ndarray:
    z = lam * np.asarray(util, dtype=float)
    z -= z.max()
    return z - np.log(np.exp(z).sum())


def contribution_table(choices: pd.DataFrame, games: pd.DataFrame,
                       opponent_guesses: pd.DataFrame,
                       types: list[str], lam: float,
                       delta: int) -> pd.DataFrame:
    required = {'game_master', *[f'opp_{t}' for t in types]}
    missing = sorted(required - set(opponent_guesses.columns))
    if missing:
        raise ValueError(f"opponent_guesses is missing columns: {missing}")

    base_cols = ['subject_id', 'game_master', 'treatment', 'session']
    out = choices[base_cols].copy()
    for t in types:
        out[f'll_{t}'] = np.nan

    opp = opponent_guesses.set_index('game_master')
    for _, g in games.iterrows():
        gid = int(g['game_master'])
        mask = out['game_master'].eq(gid).to_numpy()
        observed = choices.loc[mask, 'guess'].to_numpy(dtype=float)
        A_i = discretize(int(g['L_i']), int(g['U_i']), delta)
        A_j = discretize(int(g['L_j']), int(g['U_j']), delta)
        U = payoff_matrix(A_i, A_j, float(g['p_i']))
        obs_idx = np.argmin(np.abs(observed[:, None] - A_i[None, :]), axis=1)

        for t in types:
            opp_guess = float(opp.loc[gid, f'opp_{t}'])
            opp_idx = int(np.argmin(np.abs(A_j - opp_guess)))
            log_q = log_softmax_utility(U[:, opp_idx], lam)
            out.loc[mask, f'll_{t}'] = log_q[obs_idx]

    ll_cols = [f'll_{t}' for t in types]
    if out[ll_cols].isna().any().any():
        raise RuntimeError("incomplete contribution cache")
    return out


def posterior_rows(ll: pd.DataFrame, heldout_game: int) -> pd.DataFrame:
    ll_cols = [f'll_{t}' for t in ALL_TYPES]
    values = ll[ll_cols].to_numpy(dtype=float)
    shifted = values - values.max(axis=1, keepdims=True)
    post = np.exp(shifted)
    post /= post.sum(axis=1, keepdims=True)
    argmax = np.argmax(values, axis=1)

    out = ll[['subject_id', 'treatment', 'session']].copy()
    out.insert(1, 'game_master', int(heldout_game))
    for i, t in enumerate(ALL_TYPES):
        out[f'll_{t}'] = values[:, i]
        out[f'post_{t}'] = post[:, i]
    out['t_hat'] = [ALL_TYPES[i] for i in argmax]
    out['posterior_entropy'] = -np.sum(
        post * np.log(np.clip(post, 1e-300, None)), axis=1
    )
    out['max_posterior'] = post.max(axis=1)
    out['w_deep'] = sum(out[f'post_{t}'] for t in DEEP_TYPES)
    out['cat'] = np.where(out['t_hat'].isin(SHALLOW_TYPES), 'shallow', 'deep')
    out['margin_L1_L2'] = out['post_L1'] - out['post_L2']
    out['margin_L2_L3'] = out['post_L2'] - out['post_L3']
    return out


def crossfit_classification(choices: pd.DataFrame, games: pd.DataFrame,
                            lam: float = 0.0194,
                            delta: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    game_ids = sorted(int(x) for x in games['game_master'].unique())
    subject_ids = sorted(choices['subject_id'].astype(str).unique())
    if len(game_ids) != 16:
        raise ValueError(f"expected 16 games, found {len(game_ids)}")
    if choices.duplicated(['subject_id', 'game_master']).any():
        raise ValueError("choices contains duplicate subject-game keys")

    basic_opp = games[['game_master', *[f'opp_{t}' for t in BASIC_TYPES]]]
    basic_contrib = contribution_table(
        choices, games, basic_opp, BASIC_TYPES, lam, delta
    )
    metadata = (choices[['subject_id', 'treatment', 'session']]
                .drop_duplicates('subject_id')
                .sort_values('subject_id'))

    fold_rows: list[pd.DataFrame] = []
    mixture_rows: list[dict] = []
    for heldout in game_ids:
        train = basic_contrib[basic_contrib['game_master'] != heldout]
        ll_basic = (train.groupby('subject_id', as_index=False)
                    [[f'll_{t}' for t in BASIC_TYPES]].sum())
        ll_basic = metadata.merge(ll_basic, on='subject_id', validate='one_to_one')

        basic_values = ll_basic[[f'll_{t}' for t in BASIC_TYPES]].to_numpy()
        pass1_types = np.asarray(BASIC_TYPES, dtype=object)[np.argmax(basic_values, axis=1)]
        counts = pd.Series(pass1_types).value_counts(normalize=True)
        pi = {t: float(counts.get(t, 0.0)) for t in BASIC_TYPES}
        mixture_rows.append({'game_master': heldout, **{f'pi_{t}': pi[t]
                                                        for t in BASIC_TYPES}})

        opp_fold = games[['game_master']].copy()
        opp_fold['opp_S'] = sum(pi[t] * games[f'opp_{t}'] for t in BASIC_TYPES)
        s_contrib = contribution_table(
            choices, games, opp_fold, ['S'], lam, delta
        )
        ll_s = (s_contrib[s_contrib['game_master'] != heldout]
                .groupby('subject_id', as_index=False)['ll_S'].sum())

        ll_all = ll_basic.merge(ll_s, on='subject_id', validate='one_to_one')
        fold_rows.append(posterior_rows(ll_all, heldout))
        print(f"Fold G{heldout:02d}: training=15 games, "
              f"pi=" + ", ".join(f"{t}:{pi[t]:.3f}" for t in BASIC_TYPES))

    result = pd.concat(fold_rows, ignore_index=True)
    result = result.sort_values(['subject_id', 'game_master']).reset_index(drop=True)
    mixtures = pd.DataFrame(mixture_rows).sort_values('game_master')

    expected = len(subject_ids) * len(game_ids)
    if len(result) != expected or result.duplicated(
            ['subject_id', 'game_master']).any():
        raise RuntimeError("incomplete or duplicate cross-fitted output")
    posterior_cols = [f'post_{t}' for t in ALL_TYPES]
    if not np.allclose(result[posterior_cols].sum(axis=1), 1.0, atol=1e-12):
        raise RuntimeError("cross-fitted posteriors are not normalized")
    return result, mixtures


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--games', type=Path,
                        default=Path('data/processed/cgc2006/games.csv'))
    parser.add_argument('--choices', type=Path,
                        default=Path('data/processed/cgc2006/choices.csv'))
    parser.add_argument('--output', type=Path,
                        default=Path('data/processed/cgc2006/subject_types_crossfit.csv'))
    parser.add_argument('--mixtures-output', type=Path,
                        default=Path('data/processed/cgc2006/crossfit_mixtures.csv'))
    parser.add_argument('--lambda', type=float, default=0.0194, dest='lam')
    parser.add_argument('--delta', type=int, default=5)
    args = parser.parse_args(argv)

    games = pd.read_csv(args.games)
    choices = pd.read_csv(args.choices, dtype={'subject_id': str})
    result, mixtures = crossfit_classification(
        choices, games, lam=args.lam, delta=args.delta
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.mixtures_output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    mixtures.to_csv(args.mixtures_output, index=False)
    print(f"\nSaved {len(result)} cross-fitted classifications: {args.output}")
    print(f"Saved {len(mixtures)} fold-specific mixtures: {args.mixtures_output}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
