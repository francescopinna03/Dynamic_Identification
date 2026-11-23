from __future__ import annotations

import os

for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(variable, '1')

import argparse
import contextlib
import io
import json
import multiprocessing
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from classify_subjects_crossfit import crossfit_classification
from cross_validation import logo_cv
from fisher_cgc import fisher_per_game
from mle_cgc import fit_mle
from stratified_cv import (
    annotate_types, hard_aggregate, per_subject_delta, soft_aggregate,
)


POOLED = ['beta_soft', 'beta_hard',
          'beta_soft_leave_out', 'beta_hard_leave_out']
OUT_OF_FOLD = [f'{name}_out_of_fold' for name in POOLED]
STATISTICS = POOLED + OUT_OF_FOLD
STATE: dict = {}


def slope(contrast: np.ndarray, log_info: np.ndarray) -> float:
    if not (np.isfinite(contrast).all() and np.isfinite(log_info).all()):
        return float('nan')
    spread = log_info.std(ddof=0)
    if spread == 0:
        return float('nan')
    z = (log_info - log_info.mean()) / spread
    X = np.column_stack([np.ones(len(z)), z])
    beta, *_ = np.linalg.lstsq(X, contrast, rcond=None)
    return float(beta[1])


def contrast_table(cells: pd.DataFrame, fisher: pd.DataFrame) -> pd.DataFrame:
    hard = hard_aggregate(cells)
    for column in ('delta_deep_hard', 'delta_shallow_hard'):
        if column not in hard.columns:
            hard[column] = np.nan
    soft = soft_aggregate(cells)
    table = hard[['game_master', 'delta_deep_hard', 'delta_shallow_hard']].merge(
        soft, on='game_master', how='outer', validate='one_to_one')
    table = table.merge(fisher[['game_master', 'log_I_tau_cond']],
                        on='game_master', validate='one_to_one')
    table['contrast_hard'] = (table['delta_deep_hard']
                              - table['delta_shallow_hard'])
    table['contrast_soft'] = (table['delta_deep_soft']
                              - table['delta_shallow_soft'])
    return table.sort_values('game_master').reset_index(drop=True)


def slopes(table: pd.DataFrame, drop_game: int) -> dict:
    kept = table[table['game_master'] != drop_game]
    info = table['log_I_tau_cond'].to_numpy(dtype=float)
    info_kept = kept['log_I_tau_cond'].to_numpy(dtype=float)
    return dict(
        beta_soft=slope(table['contrast_soft'].to_numpy(dtype=float), info),
        beta_hard=slope(table['contrast_hard'].to_numpy(dtype=float), info),
        beta_soft_leave_out=slope(
            kept['contrast_soft'].to_numpy(dtype=float), info_kept),
        beta_hard_leave_out=slope(
            kept['contrast_hard'].to_numpy(dtype=float), info_kept),
    )


def fold_inputs(games: pd.DataFrame, choices: pd.DataFrame,
                cv: pd.DataFrame, delta: int,
                k_max: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    fisher_parts = []
    type_parts = []
    for _, fold in cv.iterrows():
        game = int(fold['game_master'])
        fold_lam = float(fold['lam_QL'])
        fold_tau = float(fold['tau_QL'])
        table, _ = fisher_per_game(games[games['game_master'] == game],
                                   fold_lam, fold_tau, delta=delta,
                                   K_max=k_max)
        fisher_parts.append(table)
        labels, _ = crossfit_classification(choices, games, lam=fold_lam,
                                            delta=delta)
        type_parts.append(labels[labels['game_master'] == game])
    fisher = pd.concat(fisher_parts, ignore_index=True)
    types = (pd.concat(type_parts, ignore_index=True)
             .sort_values(['subject_id', 'game_master'])
             .reset_index(drop=True))
    return fisher, types


def chain(games: pd.DataFrame, choices: pd.DataFrame, delta: int,
          k_max: int, drop_game: int) -> dict:
    with contextlib.redirect_stdout(io.StringIO()):
        mle = fit_mle(choices, games, init_lambda=0.05, init_tau=1.5,
                      delta=delta, K_max=k_max)
        lam = float(mle['lambda_hat'])
        tau = float(mle['tau_hat'])
        fisher, _ = fisher_per_game(games, lam, tau, delta=delta, K_max=k_max)
        cv = logo_cv(games, choices, delta=delta, K_max=k_max)
        types, _ = crossfit_classification(choices, games, lam=lam,
                                           delta=delta)
        gains = per_subject_delta(games, choices, cv, delta, k_max)
        cells = annotate_types(gains.copy(), types)
        fisher_fold, types_fold = fold_inputs(games, choices, cv, delta,
                                              k_max)
        cells_fold = annotate_types(gains.copy(), types_fold)
    result = dict(lambda_hat=lam, tau_hat=tau)
    result.update(slopes(contrast_table(cells, fisher), drop_game))
    fold_slopes = slopes(contrast_table(cells_fold, fisher_fold), drop_game)
    result.update({f'{name}_out_of_fold': value
                   for name, value in fold_slopes.items()})
    return result


def initialize(games: pd.DataFrame, choices: pd.DataFrame, delta: int,
               k_max: int, drop_game: int) -> None:
    subjects = sorted(choices['subject_id'].unique())
    STATE['games'] = games
    STATE['choices'] = choices
    STATE['subjects'] = subjects
    STATE['blocks'] = {s: choices[choices['subject_id'] == s]
                       for s in subjects}
    STATE['delta'] = delta
    STATE['k_max'] = k_max
    STATE['drop_game'] = drop_game


def resample(draw: np.ndarray) -> pd.DataFrame:
    parts = []
    for position, index in enumerate(draw):
        subject = STATE['subjects'][int(index)]
        block = STATE['blocks'][subject].copy()
        block['subject_id'] = f'{position:03d}_{subject}'
        parts.append(block)
    return pd.concat(parts, ignore_index=True)


def original() -> dict:
    return chain(STATE['games'], STATE['choices'], STATE['delta'],
                 STATE['k_max'], STATE['drop_game'])


def replicate(task: tuple[int, np.random.SeedSequence]) -> dict:
    index, child = task
    rng = np.random.default_rng(child)
    n_subjects = len(STATE['subjects'])
    draw = rng.integers(0, n_subjects, size=n_subjects)
    row = dict(replicate=index, status='ok',
               distinct_subjects=int(len(np.unique(draw))))
    try:
        row.update(chain(STATE['games'], resample(draw), STATE['delta'],
                         STATE['k_max'], STATE['drop_game']))
    except Exception as error:
        row['status'] = f'{type(error).__name__}: {error}'
    return row


def reference_slopes(path: Path, drop_game: int) -> dict:
    stored = pd.read_csv(path).rename(columns={
        'delta_contrast_hard': 'contrast_hard',
        'delta_contrast_soft': 'contrast_soft',
    })
    return slopes(stored.sort_values('game_master').reset_index(drop=True),
                  drop_game)


def summarize(values: np.ndarray, estimate: float) -> dict:
    valid = np.sort(values[np.isfinite(values)])
    count = int(len(valid))
    if count == 0:
        return dict(estimate=estimate, valid=0)
    lower_rank = max((count + 1) * 25 // 1000, 1)
    upper_rank = min(count + 1 - lower_rank, count)
    return dict(
        estimate=estimate,
        valid=count,
        mean=float(valid.mean()),
        sd=float(valid.std(ddof=1)) if count > 1 else float('nan'),
        lower=float(valid[lower_rank - 1]),
        upper=float(valid[upper_rank - 1]),
        share_nonpositive=float(np.mean(valid <= 0.0)),
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--n-boot', type=int, default=1999)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--drop-game', type=int, default=1)
    parser.add_argument('--workers', type=int,
                        default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument('--delta', type=int, default=5)
    parser.add_argument('--K-max', type=int, default=6)
    parser.add_argument('--tolerance', type=float, default=1e-5)
    parser.add_argument('--skip-manifest', action='store_true',
                        help='Development only: do not use for the official replication')
    args = parser.parse_args(argv)

    root = args.root.resolve()
    code_dir = Path(__file__).resolve().parent.parent
    data = root / 'data' / 'processed' / 'cgc2006'
    if not ((data / 'games.csv').is_file() and (data / 'choices.csv').is_file()):
        raise SystemExit('games.csv and choices.csv not found: run run_all.py first')
    output = (args.output_dir.resolve() if args.output_dir else
              root / 'output' / 'empirical')
    output.mkdir(parents=True, exist_ok=True)

    if not args.skip_manifest:
        subprocess.run([sys.executable, str(code_dir / 'verify_data_manifest.py'),
                        '--root', str(root)], check=True)
    else:
        print('Warning: manifest verification skipped by request')

    games = pd.read_csv(data / 'games.csv')
    choices = pd.read_csv(data / 'choices.csv', dtype={'subject_id': str})
    n_subjects = int(choices['subject_id'].nunique())
    n_games = int(games['game_master'].nunique())
    if args.drop_game not in set(games['game_master']):
        raise SystemExit(f'game {args.drop_game} is not in games.csv')
    print(f'Dataset: {len(choices)} guesses, {n_subjects} subjects, '
          f'{n_games} games')

    children = np.random.SeedSequence(args.seed).spawn(args.n_boot)
    tasks = list(enumerate(children))
    rows: list[dict] = []
    reference = root / 'reference_outputs' / 'empirical' / 'stratified_cv_crossfit.csv'
    os.environ['PYTHONHASHSEED'] = '0'
    context = multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(
            max_workers=args.workers, mp_context=context,
            initializer=initialize,
            initargs=(games, choices, args.delta, args.K_max,
                      args.drop_game)) as pool:
        print('\nBASELINE: full chain on the original sample', flush=True)
        baseline = pool.submit(original).result()
        for name in ('lambda_hat', 'tau_hat', *STATISTICS):
            print(f'  {name:<34} = {baseline[name]:+.6f}')
        if reference.is_file():
            stored = reference_slopes(reference, args.drop_game)
            gap = max(abs(baseline[name] - stored[name])
                      for name in POOLED)
            print(f'  largest gap from the stored reference table: {gap:.3e}')
            if not gap <= args.tolerance:
                raise SystemExit('BASELINE CHECK: FAIL')
            print('BASELINE CHECK: PASS')
        else:
            print('Warning: stored reference table not found; baseline check skipped')

        started = time.time()
        print(f'\nSUBJECT BOOTSTRAP: {args.n_boot} replicates, '
              f'seed {args.seed}, {args.workers} workers', flush=True)
        for row in pool.map(replicate, tasks):
            rows.append(row)
            done = len(rows)
            if done % 50 == 0 or done == args.n_boot:
                minutes = (time.time() - started) / 60.0
                print(f'  {done}/{args.n_boot} replicates, '
                      f'{minutes:.1f} minutes elapsed', flush=True)

    columns = ['replicate', 'status', 'distinct_subjects',
               'lambda_hat', 'tau_hat', *STATISTICS]
    frame = pd.DataFrame(rows).sort_values('replicate').reset_index(drop=True)
    frame = frame.reindex(columns=columns)
    failed = int((frame['status'] != 'ok').sum())

    summary = {name: summarize(frame[name].to_numpy(dtype=float),
                               baseline[name])
               for name in STATISTICS}
    payload = dict(
        settings=dict(n_boot=args.n_boot, seed=args.seed,
                      drop_game=args.drop_game, delta=args.delta,
                      K_max=args.K_max, n_subjects=n_subjects,
                      n_games=n_games),
        baseline={name: baseline[name]
                  for name in ('lambda_hat', 'tau_hat', *STATISTICS)},
        failed_replicates=failed,
        summary=summary,
    )

    print(f'\n  failed replicates: {failed}')
    print(f"  {'statistic':<34} {'estimate':>9} {'valid':>6} {'mean':>9} "
          f"{'sd':>8} {'2.5%':>9} {'97.5%':>9} {'share<=0':>9}")
    for name in STATISTICS:
        s = summary[name]
        if s['valid'] == 0:
            print(f"  {name:<34} {s['estimate']:>+9.5f} {0:>6}")
            continue
        print(f"  {name:<34} {s['estimate']:>+9.5f} {s['valid']:>6} "
              f"{s['mean']:>+9.5f} {s['sd']:>8.5f} {s['lower']:>+9.5f} "
              f"{s['upper']:>+9.5f} {s['share_nonpositive']:>9.4f}")

    replicates_path = output / 'bootstrap_subjects_replicates.csv'
    summary_path = output / 'bootstrap_subjects.json'
    frame.to_csv(replicates_path, index=False)
    summary_path.write_text(json.dumps(payload, indent=2) + '\n',
                            encoding='utf-8')
    print(f'\nReplicates saved: {replicates_path}')
    print(f'Summary saved:    {summary_path}')
    print('SUBJECT BOOTSTRAP: PASS' if failed == 0 else
          'SUBJECT BOOTSTRAP: COMPLETED WITH FAILED REPLICATES')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
