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
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bootstrap_subjects import (
    contrast_table, fold_inputs, reference_slopes, slopes,
)
from classify_subjects_crossfit import crossfit_classification
from cross_validation import logo_cv, ols_with_intercept
from fisher_cgc import fisher_per_game
from mle_cgc import fit_mle
from stratified_cv import annotate_types, per_subject_delta


VARIANTS = [
    ('pooled', 'pooled'),
    ('out_of_fold', 'pooled'),
    ('pooled', 'out_of_fold'),
    ('out_of_fold', 'out_of_fold'),
]


def regression(table: pd.DataFrame, column: str) -> dict:
    info = table['log_I_tau_cond'].to_numpy(dtype=float)
    contrast = table[column].to_numpy(dtype=float)
    z = (info - info.mean()) / info.std(ddof=0)
    fit = ols_with_intercept(contrast, z)
    rank = spearmanr(info, contrast)
    return dict(beta=float(fit['beta'][1]), se=float(fit['se'][1]),
                t=float(fit['t'][1]), r2=float(fit['r2']),
                spearman=float(rank[0]), spearman_p=float(rank[1]))


def compute(root: str, delta: int, k_max: int, drop_game: int) -> dict:
    data = Path(root) / 'data' / 'processed' / 'cgc2006'
    games = pd.read_csv(data / 'games.csv')
    choices = pd.read_csv(data / 'choices.csv', dtype={'subject_id': str})
    with contextlib.redirect_stdout(io.StringIO()):
        mle = fit_mle(choices, games, init_lambda=0.05, init_tau=1.5,
                      delta=delta, K_max=k_max)
        lam = float(mle['lambda_hat'])
        tau = float(mle['tau_hat'])
        fisher_pooled, _ = fisher_per_game(games, lam, tau, delta=delta,
                                           K_max=k_max)
        cv = logo_cv(games, choices, delta=delta, K_max=k_max)
        types_pooled, _ = crossfit_classification(choices, games, lam=lam,
                                                  delta=delta)
        fisher_fold, types_fold = fold_inputs(games, choices, cv, delta,
                                              k_max)
        gains = per_subject_delta(games, choices, cv, delta, k_max)
    fisher = dict(pooled=fisher_pooled, out_of_fold=fisher_fold)
    types = dict(pooled=types_pooled, out_of_fold=types_fold)
    results = []
    tables = {}
    for information, precision in VARIANTS:
        cells = annotate_types(gains.copy(), types[precision])
        table = contrast_table(cells, fisher[information])
        tables[(information, precision)] = table
        kept = table[table['game_master'] != drop_game]
        for sample, frame in (('all_games', table), ('leave_out', kept)):
            for rule in ('soft', 'hard'):
                row = dict(information=information, precision=precision,
                           sample=sample, classification=rule)
                row.update(regression(frame, f'contrast_{rule}'))
                results.append(row)
    merged = types_pooled.merge(types_fold, on=['subject_id', 'game_master'],
                                suffixes=('_pooled', '_fold'))
    games_table = cv[['game_master', 'lam_QL', 'tau_QL']].merge(
        fisher_pooled[['game_master', 'log_I_tau_cond']],
        on='game_master', validate='one_to_one').merge(
        fisher_fold[['game_master', 'log_I_tau_cond']],
        on='game_master', validate='one_to_one',
        suffixes=('_pooled', '_out_of_fold'))
    for key, label in ((('pooled', 'pooled'), 'pooled'),
                       (('out_of_fold', 'out_of_fold'), 'out_of_fold')):
        part = tables[key][['game_master', 'contrast_soft', 'contrast_hard']]
        games_table = games_table.merge(
            part.rename(columns={'contrast_soft': f'contrast_soft_{label}',
                                 'contrast_hard': f'contrast_hard_{label}'}),
            on='game_master', validate='one_to_one')
    return dict(
        settings=dict(delta=delta, K_max=k_max, drop_game=drop_game,
                      n_subjects=int(choices['subject_id'].nunique()),
                      n_games=int(games['game_master'].nunique())),
        pooled_estimate=dict(lambda_hat=lam, tau_hat=tau),
        baseline=slopes(tables[('pooled', 'pooled')], drop_game),
        deep_assignments_changed=int((merged['cat_pooled']
                                      != merged['cat_fold']).sum()),
        type_labels_changed=int((merged['t_hat_pooled']
                                 != merged['t_hat_fold']).sum()),
        cells=int(len(merged)),
        largest_weight_change=float(np.abs(merged['w_deep_pooled']
                                           - merged['w_deep_fold']).max()),
        regressions=results,
        games=games_table.sort_values('game_master').to_dict('records'),
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--drop-game', type=int, default=1)
    parser.add_argument('--delta', type=int, default=5)
    parser.add_argument('--K-max', type=int, default=6)
    parser.add_argument('--tolerance', type=float, default=1e-5)
    parser.add_argument('--skip-manifest', action='store_true',
                        help='Development only: do not use for the official replication')
    args = parser.parse_args(argv)

    root = args.root.resolve()
    code_dir = Path(__file__).resolve().parent.parent
    output = (args.output_dir.resolve() if args.output_dir else
              root / 'output' / 'empirical')
    output.mkdir(parents=True, exist_ok=True)
    if not args.skip_manifest:
        subprocess.run([sys.executable, str(code_dir / 'verify_data_manifest.py'),
                        '--root', str(root)], check=True)
    else:
        print('Warning: manifest verification skipped by request')

    os.environ['PYTHONHASHSEED'] = '0'
    context = multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(max_workers=1, mp_context=context) as pool:
        payload = pool.submit(compute, str(root), args.delta, args.K_max,
                              args.drop_game).result()

    print(f"Pooled estimate: lambda={payload['pooled_estimate']['lambda_hat']:.6f}  "
          f"tau={payload['pooled_estimate']['tau_hat']:.4f}")
    reference = (root / 'reference_outputs' / 'empirical'
                 / 'stratified_cv_crossfit.csv')
    if reference.is_file():
        stored = reference_slopes(reference, args.drop_game)
        gap = max(abs(payload['baseline'][name] - stored[name])
                  for name in stored)
        print(f'Largest gap of the pooled slopes from the stored reference '
              f'table: {gap:.3e}')
        if not gap <= args.tolerance:
            raise SystemExit('BASELINE CHECK: FAIL')
        print('BASELINE CHECK: PASS')
    else:
        print('Warning: stored reference table not found; baseline check skipped')
    print(f"Fold-specific precision: shallow or deep assignments changed "
          f"{payload['deep_assignments_changed']}, type labels changed "
          f"{payload['type_labels_changed']}, cells {payload['cells']}, "
          f"largest change in a deep weight "
          f"{payload['largest_weight_change']:.4f}")
    print(f"\n  {'information':<12} {'precision':<12} {'sample':<10} "
          f"{'rule':<5} {'beta':>8} {'t':>6} {'spearman':>9} {'p':>6}")
    for row in payload['regressions']:
        print(f"  {row['information']:<12} {row['precision']:<12} "
              f"{row['sample']:<10} {row['classification']:<5} "
              f"{row['beta']:>+8.4f} {row['t']:>6.2f} "
              f"{row['spearman']:>9.3f} {row['spearman_p']:>6.3f}")
    frame = pd.DataFrame(payload['games'])
    print('\n' + frame.to_string(index=False,
                                 float_format=lambda value: f'{value:.4f}'))

    table_path = output / 'out_of_fold_information_games.csv'
    summary_path = output / 'out_of_fold_information.json'
    frame.to_csv(table_path, index=False)
    summary_path.write_text(json.dumps(payload, indent=2) + '\n',
                            encoding='utf-8')
    print(f'\nGame table saved: {table_path}')
    print(f'Summary saved:    {summary_path}')
    print('OUT-OF-FOLD INFORMATION: PASS')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
