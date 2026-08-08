from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


MASTER_ORDER = [1, 3, 5, 7, 9, 11, 13, 15, 2, 4, 6, 8, 10, 12, 14, 16]

TABLE3_PLAYED = pd.DataFrame([
    (1,  100, 900, 1.5, 300, 500, 0.7, 600, 525, 630.0, 600.0, 611.25, 750, 630),
    (2,  300, 900, 1.3, 300, 500, 1.5, 520, 650, 650.0, 617.5, 650.00, 650, 650),
    (3,  300, 900, 1.3, 300, 900, 1.3, 780, 900, 900.0, 838.5, 900.00, 900, 900),
    (4,  300, 900, 0.7, 100, 900, 1.3, 350, 546, 318.5, 451.5, 423.15, 300, 420),
    (5,  100, 500, 1.5, 100, 500, 0.7, 450, 315, 472.5, 337.5, 341.25, 500, 375),
    (6,  100, 500, 0.7, 100, 900, 0.5, 350, 105, 122.5, 122.5, 122.50, 100, 122),
    (7,  100, 500, 0.7, 100, 500, 1.5, 210, 315, 220.5, 227.5, 227.50, 350, 262),
    (8,  300, 500, 0.7, 100, 900, 1.5, 350, 420, 367.5, 420.0, 420.00, 500, 420),
    (9,  300, 500, 1.5, 300, 900, 1.3, 500, 500, 500.0, 500.0, 500.00, 500, 500),
    (10, 300, 500, 0.7, 100, 900, 0.5, 350, 300, 300.0, 300.0, 300.00, 300, 300),
    (11, 100, 500, 1.5, 100, 900, 0.5, 500, 225, 375.0, 262.5, 262.50, 150, 300),
    (12, 300, 900, 1.3, 300, 900, 1.3, 780, 900, 900.0, 838.5, 900.00, 900, 900),
    (13, 100, 900, 1.3, 300, 900, 0.7, 780, 455, 709.8, 604.5, 604.50, 390, 695),
    (14, 100, 900, 0.5, 300, 500, 0.7, 200, 175, 150.0, 200.0, 150.00, 150, 162),
    (15, 100, 900, 0.5, 100, 500, 0.7, 150, 175, 100.0, 150.0, 100.00, 100, 132),
    (16, 100, 900, 0.5, 100, 500, 1.5, 150, 250, 112.5, 162.5, 131.25, 100, 187),
], columns=['game_played', 'L_i', 'U_i', 'p_i', 'L_j', 'U_j', 'p_j',
            'L1', 'L2', 'L3', 'D1', 'D2', 'E', 'S'])


def treatment_from_session(s: int) -> str:
    if 1 <= s <= 4:
        return 'Baseline'
    if s == 5:
        return 'OB'
    return 'R/TS'


def parse_subject_id(raw: str) -> tuple[int, int]:
    s = raw.replace('0.', '', 1)
    if len(s) == 4:
        return int(s[:2]), int(s[2:])
    if len(s) == 3:
        return int(s[:2]), int(s[2:]) * 10
    raise ValueError(f"Unrecognized ID format: {raw!r}")


def build_games(input_dir: Path) -> pd.DataFrame:
    df_tg = pd.read_csv(input_dir / 'TypesGuesses.xls', sep='\t', header=None,
                        names=['game_master', 'E', 'L1', 'L2', 'L3', 'D1', 'D2'])

    df_og = pd.read_csv(input_dir / 'OpponentsGuesses.xls', sep='\t',
                        header=None,
                        names=['game_master', 'opp_E', 'opp_L1', 'opp_L2',
                               'opp_L3', 'opp_D1', 'opp_D2'])

    dom_cols = (['game_master'] +
                [f'r{r}_{b}' for r in range(1, 6) for b in ['lo', 'hi']])
    df_dom = pd.read_csv(input_dir / 'DominanceRounds.xls', sep='\t',
                         header=None, names=dom_cols)

    mapping_rows = []
    used_played = set()
    for _, row in df_tg.iterrows():
        candidates = TABLE3_PLAYED[
            (abs(TABLE3_PLAYED['E']  - row['E'])  < 0.5) &
            (abs(TABLE3_PLAYED['L1'] - row['L1']) < 0.5) &
            (abs(TABLE3_PLAYED['L2'] - row['L2']) < 0.5) &
            (abs(TABLE3_PLAYED['L3'] - row['L3']) < 0.5) &
            (abs(TABLE3_PLAYED['D1'] - row['D1']) < 0.5) &
            (abs(TABLE3_PLAYED['D2'] - row['D2']) < 0.5)
        ]
        candidates = candidates[~candidates['game_played'].isin(used_played)]
        if len(candidates) >= 1:
            gp = int(candidates['game_played'].iloc[0])
        else:
            gp = -1
            print(f"  Warning: no match for game_master={row['game_master']}",
                  file=sys.stderr)
        mapping_rows.append((int(row['game_master']), gp))
        if gp > 0:
            used_played.add(gp)

    df_map = pd.DataFrame(mapping_rows, columns=['game_master', 'game_played'])

    games = (df_map
             .merge(TABLE3_PLAYED[['game_played', 'L_i', 'U_i', 'p_i',
                                   'L_j', 'U_j', 'p_j', 'S']],
                    on='game_played')
             .merge(df_tg, on='game_master')
             .merge(df_og, on='game_master')
             .merge(df_dom, on='game_master'))

    games = games.sort_values('game_master').reset_index(drop=True)

    ordered_cols = (
        ['game_master', 'game_played',
         'L_i', 'U_i', 'p_i', 'L_j', 'U_j', 'p_j']
        + ['E', 'L1', 'L2', 'L3', 'D1', 'D2', 'S']
        + ['opp_E', 'opp_L1', 'opp_L2', 'opp_L3', 'opp_D1', 'opp_D2']
        + [f'r{r}_{b}' for r in range(1, 6) for b in ['lo', 'hi']]
    )
    return games[ordered_cols]


def build_choices(input_dir: Path) -> pd.DataFrame:
    df_sg = pd.read_csv(input_dir / 'SubjectsGuesses.xls', sep='\t',
                        header=None, dtype=str)
    df_sg.columns = ['subject_id_raw'] + [f'g{g}' for g in MASTER_ORDER]

    parsed = df_sg['subject_id_raw'].apply(parse_subject_id)
    df_sg['session'] = parsed.apply(lambda t: t[0])
    df_sg['subject_in_session'] = parsed.apply(lambda t: t[1])
    df_sg['treatment'] = df_sg['session'].apply(treatment_from_session)

    df_sg['subject_id'] = (df_sg['session'].astype(str).str.zfill(2) + '_'
                           + df_sg['subject_in_session'].astype(str).str.zfill(2))

    id_vars = ['subject_id', 'subject_id_raw', 'session', 'subject_in_session',
               'treatment']
    value_vars = [f'g{g}' for g in MASTER_ORDER]
    long = df_sg.melt(id_vars=id_vars, value_vars=value_vars,
                      var_name='game_col', value_name='guess')
    long['game_master'] = long['game_col'].str.replace('g', '').astype(int)
    long['guess'] = long['guess'].astype(float)
    long = long.drop(columns=['game_col'])

    long = long.sort_values(['subject_id', 'game_master']).reset_index(drop=True)

    return long[['subject_id', 'subject_id_raw', 'session',
                 'subject_in_session', 'treatment', 'game_master', 'guess']]


def sanity_report(games: pd.DataFrame, choices: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("Sanity checks")
    print("=" * 70)

    assert len(games) == 16, f"Expected 16 games, found {len(games)}"
    assert games['game_master'].is_unique
    print(f"  games:    {len(games)} rows (expected 16) OK")

    same = games.set_index('game_master').loc[[7, 8],
                                              ['L_i', 'U_i', 'p_i',
                                               'L_j', 'U_j', 'p_j']]
    if (same.loc[7] == same.loc[8]).all():
        print("  games 7 and 8 have identical parameters (delta3 delta3) OK")
    else:
        print("  Warning: games 7 and 8 should have identical parameters")

    n_sub = choices['subject_id'].nunique()
    n_games_per_sub = choices.groupby('subject_id').size().unique()
    assert len(n_games_per_sub) == 1 and n_games_per_sub[0] == 16, \
        f"Expected 16 games per subject, found {n_games_per_sub}"
    print(f"  choices:  {len(choices)} rows, "
          f"{n_sub} subjects x 16 games OK")

    treat_dist = choices.groupby(['treatment', 'session'])['subject_id']\
        .nunique().sort_index()
    print("\n  Subject distribution by (treatment, session):")
    for (t, s), n in treat_dist.items():
        print(f"    {t:<10} session {s}: {n} subjects")

    merged = choices.merge(games[['game_master', 'L_i', 'U_i']],
                           on='game_master')
    n_below = (merged['guess'] < merged['L_i']).sum()
    n_above = (merged['guess'] > merged['U_i']).sum()
    print(f"\n  Guess range: [{choices['guess'].min():.0f}, "
          f"{choices['guess'].max():.0f}]")
    print("  Pre-adjustment guesses outside the bounds:")
    print(f"    below L_i: {n_below:4d} "
          f"({100*n_below/len(merged):.2f}%)")
    print(f"    above U_i: {n_above:4d} "
          f"({100*n_above/len(merged):.2f}%)")
    print("  (Pre-adjustment by design: MouseLab adjusted automatically.)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path,
                        default=Path('data/raw/cgc2006/data'),
                        help="Directory containing the four raw TSV files")
    parser.add_argument('--output', type=Path,
                        default=Path('data/processed/cgc2006'),
                        help="Output directory for the processed CSV files")
    args = parser.parse_args(argv)

    input_dir = args.input.resolve()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}\n")

    required = ['SubjectsGuesses.xls', 'TypesGuesses.xls',
                'OpponentsGuesses.xls', 'DominanceRounds.xls']
    missing = [f for f in required if not (input_dir / f).exists()]
    if missing:
        print(f"Error: missing files in {input_dir}: {missing}",
              file=sys.stderr)
        return 1

    print("[1/2] Build games.csv")
    games = build_games(input_dir)
    games.to_csv(output_dir / 'games.csv', index=False)
    print(f"  {len(games)} rows -> {output_dir / 'games.csv'}")

    print("\n[2/2] Build choices.csv")
    choices = build_choices(input_dir)
    choices.to_csv(output_dir / 'choices.csv', index=False)
    print(f"  {len(choices)} rows -> {output_dir / 'choices.csv'}")

    sanity_report(games, choices)

    print("\n" + "=" * 70)
    print("Preprocessing completed.")
    print("=" * 70)
    print(f"  games.csv:   {output_dir / 'games.csv'}")
    print(f"  choices.csv: {output_dir / 'choices.csv'}")
    print("\nNext step: parse the Gauss .jnk files for compliance search data "
          "and compute the Fisher information for each game.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
