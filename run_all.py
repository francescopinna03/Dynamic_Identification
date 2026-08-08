from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(command: list[str], root: Path) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=root, check=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-empirical", action="store_true")
    parser.add_argument("--skip-theoretical", action="store_true")
    parser.add_argument("--n-perm", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parent
    python = sys.executable
    empirical = root / "code" / "empirical"
    theoretical = root / "code" / "theoretical"
    output = root / "output"
    output.mkdir(parents=True, exist_ok=True)

    if not args.skip_empirical:
        if not args.skip_download:
            run([python, str(empirical / "acquire_cgc2006.py"),
                 "--root", str(root)], root)
        run([python, str(empirical / "preprocess.py"),
             "--input", str(root / "data" / "raw" / "cgc2006" / "data"),
             "--output", str(root / "data" / "processed" / "cgc2006")], root)
        run([python, str(empirical / "parse_jnk.py"),
             "--data-dir", str(root / "data" / "raw" / "cgc2006" / "data"),
             "--choices", str(root / "data" / "processed" / "cgc2006" / "choices.csv"),
             "--output-dir", str(root / "data" / "processed" / "cgc2006")], root)
        run([python, str(empirical / "run_empirical.py"),
             "--root", str(root),
             "--output-dir", str(output / "empirical"),
             "--n-perm", str(args.n_perm),
             "--seed", str(args.seed)], root)

    if not args.skip_theoretical:
        run([python, str(theoretical / "design_games.py"),
             "--output", str(output / "design_games.json"),
             "--seed", str(args.seed)], root)
        run([python, str(theoretical / "verify_math.py"),
             "--output", str(output / "math_checks.json"),
             "--seed", str(args.seed)], root)
        run([python, str(theoretical / "conditional_wf_sbb.py"),
             "--output", str(output / "conditional_wf_sbb.json"),
             "--seed", str(args.seed)], root)
        run([python, str(theoretical / "robustness_wf_sbb.py"),
             "--json-output", str(output / "robustness_wf_sbb.json"),
             "--csv-output", str(output / "robustness_wf_sbb.csv"),
             "--seed", str(args.seed)], root)
        run([python, str(theoretical / "central_path_transition.py"),
             "--output", str(output / "central_path_transition.json"),
             "--seed", str(args.seed)], root)
        run([python, str(theoretical / "log_domain_boundary_layer.py"),
             "--output", str(output / "log_domain_boundary_layer.json"),
             "--csv-output", str(output / "log_domain_information_path.csv"),
             "--seed", str(args.seed)], root)

    run([python, str(root / "code" / "verify_artifacts.py"),
         "--root", str(root)], root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
