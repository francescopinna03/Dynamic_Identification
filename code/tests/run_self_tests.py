from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
EMPIRICAL = HERE.parents[1] / 'empirical'
THEORETICAL = HERE.parents[1] / 'theoretical'
sys.path[:0] = [str(EMPIRICAL), str(THEORETICAL)]

from auction import perfect_signal_collapse
from classify_subjects_crossfit import crossfit_classification
from fisher_cgc import fisher_per_game
from sample_size import battery_information, required_subjects


def test_fisher_profile(games: pd.DataFrame) -> None:
    df, _ = fisher_per_game(games, 0.0194085151, 1.138462755,
                            delta=5, K_max=6)
    expected = 1.0 / np.sqrt(df['I_tau_cond'].to_numpy())
    np.testing.assert_allclose(df['se_tau'], expected, rtol=2e-10, atol=2e-10)
    assert df['log_I_tau_cond'].map(np.isfinite).all()
    assert df['tau_informativity'].notna().all()
    print("PASS: Schur complement and SE(tau) coincide for all 16 games")


def test_crossfit_no_leakage(games: pd.DataFrame,
                             choices: pd.DataFrame) -> None:
    base, _ = crossfit_classification(choices, games, lam=0.0194, delta=5)
    changed = choices.copy()
    g1 = games.loc[games['game_master'] == 1].iloc[0]
    changed.loc[changed['game_master'] == 1, 'guess'] = float(g1['L_i'])
    perturbed, _ = crossfit_classification(changed, games, lam=0.0194, delta=5)

    cols = ['subject_id', 'game_master', 't_hat', 'w_deep',
            *[f'll_{t}' for t in ['E', 'L1', 'L2', 'L3', 'D1', 'D2', 'S']],
            *[f'post_{t}' for t in ['E', 'L1', 'L2', 'L3', 'D1', 'D2', 'S']]]
    left = base.loc[base['game_master'] == 1, cols].reset_index(drop=True)
    right = perturbed.loc[perturbed['game_master'] == 1, cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right, check_exact=True)
    print("PASS: changing every G1 test choice does not change the G1 fold types")


def test_perfect_signal() -> None:
    result = perfect_signal_collapse()
    assert result['passed'], result
    print("PASS: conditional G4 collapses relative to chi under a perfect signal")


def test_sample_size_accounting() -> None:
    I1 = np.diag([2.0, 1.0])
    I2 = np.diag([1.0, 3.0])
    total = battery_information([I1, I2])
    np.testing.assert_allclose(total, np.diag([3.0, 4.0]))
    target = 0.1
    expected = int(np.ceil(np.trace(np.linalg.inv(total)) / target**2))
    assert required_subjects(total, target, criterion='total_mse') == expected
    expected_avg = int(np.ceil(np.trace(np.linalg.inv(total)) /
                               (total.shape[0] * target**2)))
    assert required_subjects(total, target, criterion='avg_mse') == expected_avg
    print("PASS: sample size uses per-subject Fisher information without post-hoc division")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--games', type=Path, required=True)
    parser.add_argument('--choices', type=Path, required=True)
    args = parser.parse_args(argv)
    games = pd.read_csv(args.games)
    choices = pd.read_csv(args.choices, dtype={'subject_id': str})

    test_fisher_profile(games)
    test_crossfit_no_leakage(games, choices)
    test_perfect_signal()
    test_sample_size_accounting()
    print("SELF-TESTS: ALL PASS")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
