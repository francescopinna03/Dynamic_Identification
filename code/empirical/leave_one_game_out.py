from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cross_validation import ols_with_intercept


def fit(table: pd.DataFrame, column: str) -> tuple[float, float]:
    info = table['log_I_tau_cond'].to_numpy(dtype=float)
    z = (info - info.mean()) / info.std(ddof=0)
    result = ols_with_intercept(table[column].to_numpy(dtype=float), z)
    return float(result['beta'][1]), float(result['t'][1])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--input', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    source = args.input or (root / 'output' / 'empirical'
                            / 'stratified_cv_crossfit.csv')
    target = args.output or (root / 'output' / 'empirical'
                             / 'leave_one_game_out.csv')
    table = (pd.read_csv(source).sort_values('game_master')
             .reset_index(drop=True))
    rows = []
    for game in table['game_master']:
        own = table[table['game_master'] == game].iloc[0]
        rest = table[table['game_master'] != game]
        beta_soft, t_soft = fit(rest, 'delta_contrast_soft')
        beta_hard, t_hard = fit(rest, 'delta_contrast_hard')
        rows.append(dict(
            omitted_game=int(game),
            log_I_tau_cond=float(own['log_I_tau_cond']),
            contrast_soft=float(own['delta_contrast_soft']),
            beta_soft=beta_soft, t_soft=t_soft,
            beta_hard=beta_hard, t_hard=t_hard,
        ))
    result = pd.DataFrame(rows)
    target.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(target, index=False, float_format='%.10f')
    print(result.to_string(index=False,
                           float_format=lambda value: f'{value:.4f}'))
    print(f'Table saved: {target}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
