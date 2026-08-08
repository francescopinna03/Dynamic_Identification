from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(command: list[str], stdout_path: Path | None = None) -> None:
    print("\n$ " + " ".join(command), flush=True)
    if stdout_path is None:
        subprocess.run(command, check=True)
    else:
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        with stdout_path.open('w', encoding='utf-8') as stream:
            subprocess.run(command, check=True, stdout=stream,
                           stderr=subprocess.STDOUT)
        print(f"Output saved: {stdout_path}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--delta', type=int, default=5)
    parser.add_argument('--K-max', type=int, default=6)
    parser.add_argument('--n-perm', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--skip-manifest', action='store_true',
                        help='Development only: do not use for the official replication')
    args = parser.parse_args(argv)

    root = args.root.resolve()
    scripts = Path(__file__).resolve().parent
    code_dir = scripts.parent
    python = sys.executable

    candidates = [root / 'data' / 'processed' / 'cgc2006',
                  root / 'data' / 'processed']
    data = next((p for p in candidates
                 if (p / 'games.csv').is_file() and
                 (p / 'choices.csv').is_file()), None)
    if data is None:
        raise SystemExit("games.csv and choices.csv not found: run preprocess.py")
    output = (args.output_dir.resolve() if args.output_dir else
              root / 'output' / 'empirical')
    output.mkdir(parents=True, exist_ok=True)

    games = data / 'games.csv'
    choices = data / 'choices.csv'
    mle = output / 'mle_pooled.json'
    fisher = output / 'fisher_diagnostics.csv'
    cv = output / 'cv_results.csv'
    types = output / 'subject_types_crossfit.csv'
    mixtures = output / 'crossfit_mixtures.csv'
    strat = output / 'stratified_cv_crossfit.csv'
    process_density = data / 'compliance_density_long.csv'

    if not args.skip_manifest:
        run([python, str(code_dir / 'verify_data_manifest.py'),
             '--root', str(root)])
    else:
        print("Warning: manifest verification skipped by request")
    run([python, str(scripts / 'mle_cgc.py'),
         '--games', str(games), '--choices', str(choices),
         '--delta', str(args.delta), '--K-max', str(args.K_max),
         '--output', str(mle)])
    mle_values = json.loads(mle.read_text(encoding='utf-8'))

    run([python, str(scripts / 'fisher_cgc.py'),
         '--games', str(games), '--output', str(fisher),
         '--lambda0', str(mle_values['lambda_hat']),
         '--tau0', str(mle_values['tau_hat']),
         '--delta', str(args.delta), '--K-max', str(args.K_max)])
    run([python, str(scripts / 'cross_validation.py'),
         '--games', str(games), '--choices', str(choices),
         '--fisher', str(fisher), '--output', str(cv),
         '--delta', str(args.delta), '--K-max', str(args.K_max)])
    run([python, str(scripts / 'classify_subjects_crossfit.py'),
         '--games', str(games), '--choices', str(choices),
         '--output', str(types), '--mixtures-output', str(mixtures),
         '--lambda', str(mle_values['lambda_hat']),
         '--delta', str(args.delta)])
    run([python, str(scripts / 'stratified_cv.py'),
         '--games', str(games), '--choices', str(choices),
         '--cv', str(cv), '--types', str(types), '--fisher', str(fisher),
         '--output', str(strat), '--delta', str(args.delta),
         '--K-max', str(args.K_max)])
    run([python, str(scripts / 'robustness.py'),
         '--games', str(games), '--choices', str(choices),
         '--cv', str(cv), '--types', str(types), '--fisher', str(fisher),
         '--n-perm', str(args.n_perm), '--seed', str(args.seed),
         '--delta', str(args.delta), '--K-max', str(args.K_max)],
        stdout_path=output / 'robustness.txt')
    run([python, str(code_dir / 'tests' / 'run_self_tests.py'),
         '--games', str(games), '--choices', str(choices)],
        stdout_path=output / 'self_tests.txt')
    if process_density.is_file():
        run([python, str(scripts / 'process_tracing_crossfit.py'),
             '--types', str(types), '--fisher', str(fisher),
             '--density', str(process_density),
             '--output', str(output / 'process_tracing_crossfit.json'),
             '--csv-output',
             str(output / 'process_tracing_crossfit_games.csv')])
    else:
        print("Warning: compliance_density_long.csv is absent; MouseLab audit skipped")

    print("\nEMPIRICAL PIPELINE: PASS")
    print(f"Output: {output}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
