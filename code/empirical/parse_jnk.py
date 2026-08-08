from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


GAUSS_TYPE_ORDER = ['L1', 'L2', 'L3', 'S', 'D1', 'D2', 'E']


def parse_anadhb(path: Path, n_games: int = 16, n_types: int = 7,
                  n_subjects: int = 71) -> np.ndarray:
    with open(path) as f:
        text = f.read()
    nums = [float(x) for x in text.split()]
    expected = n_games * n_types * n_subjects
    if len(nums) != expected:
        raise ValueError(f"Parsed {len(nums)} numbers, expected {expected}")
    arr = np.array(nums).reshape(n_games * n_types, n_subjects)
    return arr


def parse_datis(path: Path, n_types: int = 7, n_subjects: int = 71,
                 n_cols: int = 9) -> np.ndarray:
    with open(path) as f:
        text = f.read()
    nums = [int(x) for x in text.split()]
    expected = n_types * n_subjects * n_cols
    if len(nums) != expected:
        raise ValueError(f"Parsed {len(nums)} numbers, expected {expected}")
    arr = np.array(nums).reshape(n_types * n_subjects, n_cols)
    return arr


def anadhb_to_long(arr: np.ndarray, look_up: str,
                    baseline_subject_ids: list[str]) -> pd.DataFrame:
    rows = []
    n_games = 16
    n_types = len(GAUSS_TYPE_ORDER)
    for ty in range(1, n_types + 1):
        type_name = GAUSS_TYPE_ORDER[ty - 1]
        for g in range(1, n_games + 1):
            row_idx = (ty - 1) * n_games + (g - 1)
            for s_idx, sid in enumerate(baseline_subject_ids):
                rows.append(dict(
                    subject_id=sid,
                    game_master=g,
                    type=type_name,
                    look_up=look_up,
                    density=float(arr[row_idx, s_idx]),
                ))
    return pd.DataFrame(rows)


def datis_to_long(arr: np.ndarray, look_up: str,
                   baseline_subject_ids: list[str]) -> pd.DataFrame:
    rows = []
    n_subjects = len(baseline_subject_ids)
    n_types = len(GAUSS_TYPE_ORDER)
    for ty in range(1, n_types + 1):
        type_name = GAUSS_TYPE_ORDER[ty - 1]
        for s_idx, sid in enumerate(baseline_subject_ids):
            row_idx = (ty - 1) * n_subjects + s_idx
            row = dict(
                subject_id=sid,
                type=type_name,
                look_up=look_up,
            )
            for c in range(9):
                row[f'cat_{c + 1}'] = int(arr[row_idx, c])
            rows.append(row)
    return pd.DataFrame(rows)


def check_type_convention(density_long: pd.DataFrame,
                            types_df: pd.DataFrame) -> None:
    print("\n  Type-convention check:")
    print("  For each type T, compute the mean compliance of subjects")
    print("  classified as T under each of the seven Gauss types.")
    print("  Under the correct convention, the maximum should lie on the")
    print("  diagonal (classified type equals Gauss type).")
    print()

    classified = types_df[types_df['treatment'] == 'Baseline'].copy()
    mean_by_class = (density_long
                     .merge(classified[['subject_id', 't_hat']],
                             on='subject_id')
                     .groupby(['t_hat', 'type'])['density']
                     .mean().unstack())
    mean_by_class = mean_by_class.reindex(index=GAUSS_TYPE_ORDER,
                                             columns=GAUSS_TYPE_ORDER)
    print("  Mean compliance density (row = CGC classification, "
          "column = Gauss type):")
    print(mean_by_class.round(3).to_string())
    print()
    print("  Argmax by row (expected to match the row name):")
    for t in GAUSS_TYPE_ORDER:
        if t in mean_by_class.index:
            row = mean_by_class.loc[t].dropna()
            if len(row) > 0:
                argmax_t = row.idxmax()
                check = "✓" if argmax_t == t else "✗"
                print(f"    Subjects classified as {t}: "
                      f"argmax = {argmax_t}  {check}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path,
                        default=Path('data/raw/cgc2006/data'))
    parser.add_argument('--choices', type=Path,
                        default=Path('data/processed/cgc2006/choices.csv'))
    parser.add_argument('--types', type=Path,
                        default=Path('data/processed/cgc2006/subject_types.csv'))
    parser.add_argument('--output-dir', type=Path,
                        default=Path('data/processed/cgc2006'))
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    choices = pd.read_csv(args.choices)
    baseline_subjects = (choices[choices['treatment'] == 'Baseline']
                         .drop_duplicates('subject_id')
                         .sort_values(['session', 'subject_in_session'])
                         ['subject_id'].tolist())
    print(f"Baseline subjects: {len(baseline_subjects)} (expected 71)")
    assert len(baseline_subjects) == 71, \
        f"Expected 71 Baseline subjects, found {len(baseline_subjects)}"

    print("\n[1/2] Parsing ANADHB (compliance density per game-type-subject)...")
    arr_early = parse_anadhb(args.data_dir / 'ANADHB0EQ.jnk')
    arr_late  = parse_anadhb(args.data_dir / 'ANADHB0TEQ.jnk')
    print(f"  ANADHB0EQ:  shape {arr_early.shape}, "
          f"range [{arr_early.min():.3f}, {arr_early.max():.3f}]")
    print(f"  ANADHB0TEQ: shape {arr_late.shape}, "
          f"range [{arr_late.min():.3f}, {arr_late.max():.3f}]")

    density_early = anadhb_to_long(arr_early, 'early', baseline_subjects)
    density_late  = anadhb_to_long(arr_late, 'late', baseline_subjects)
    density_long  = pd.concat([density_early, density_late], ignore_index=True)

    print(f"  Long format: {len(density_long)} rows "
          f"(expected 16*7*71*2 = {16*7*71*2})")

    print("\n[2/2] Parsing DATIS (compliance counts in 9 categories)...")
    datis_early = parse_datis(args.data_dir / 'DATISEEQ.jnk')
    datis_late  = parse_datis(args.data_dir / 'DATISLEQ.jnk')
    print(f"  DATISEEQ: shape {datis_early.shape}")
    print(f"  DATISLEQ: shape {datis_late.shape}")

    datis_early_long = datis_to_long(datis_early, 'early', baseline_subjects)
    datis_late_long  = datis_to_long(datis_late,  'late',  baseline_subjects)
    datis_long = pd.concat([datis_early_long, datis_late_long],
                            ignore_index=True)

    print(f"  Long format: {len(datis_long)} rows "
          f"(expected 71*7*2 = {71*7*2})")

    if args.types.exists():
        types_df = pd.read_csv(args.types)
        check_type_convention(density_long, types_df)

    density_long.to_csv(args.output_dir / 'compliance_density_long.csv',
                         index=False)
    datis_long.to_csv(args.output_dir / 'compliance_counts_long.csv',
                       index=False)
    print("\nSaved:")
    print(f"  {args.output_dir / 'compliance_density_long.csv'}")
    print(f"  {args.output_dir / 'compliance_counts_long.csv'}")
    return 0


if __name__ == "__main__":
    main()
