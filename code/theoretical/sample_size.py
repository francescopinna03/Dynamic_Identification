from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def battery_information(per_game: list[np.ndarray]) -> np.ndarray:
    if not per_game:
        raise ValueError("the battery must contain at least one game")
    shapes = {np.asarray(x).shape for x in per_game}
    if len(shapes) != 1:
        raise ValueError(f"Fisher matrices have incompatible shapes: {shapes}")
    total = np.sum(np.stack(per_game), axis=0)
    if not np.allclose(total, total.T, atol=1e-10):
        raise ValueError("the composite Fisher matrix is not symmetric")
    return total


def required_subjects(I_subject: np.ndarray, target: float,
                      criterion: str = 'avg_mse') -> int:
    if target <= 0:
        raise ValueError("target must be positive")
    eig = np.linalg.eigvalsh(I_subject)
    if eig.min() <= 0:
        raise ValueError(f"per-subject Fisher matrix is not positive definite: eig={eig}")
    cov_one = np.linalg.inv(I_subject)
    if criterion == 'total_mse':
        risk_one = float(np.trace(cov_one))
    elif criterion == 'avg_mse':
        risk_one = float(np.trace(cov_one) / I_subject.shape[0])
    elif criterion == 'max_se':
        risk_one = float(np.max(np.diag(cov_one)))
    else:
        raise ValueError("criterion must be avg_mse, total_mse, or max_se")
    return int(np.ceil(risk_one / (target * target)))


def load_matrices(path: Path) -> list[np.ndarray]:
    payload = json.loads(path.read_text(encoding='utf-8'))
    games = payload.get('games', [])
    return [np.asarray(g['fisher'], dtype=float) for g in games]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('fisher_json', type=Path)
    parser.add_argument('--target', type=float, required=True)
    parser.add_argument('--criterion',
                        choices=['avg_mse', 'total_mse', 'max_se'],
                        default='avg_mse')
    args = parser.parse_args(argv)

    matrices = load_matrices(args.fisher_json)
    I_subject = battery_information(matrices)
    n = required_subjects(I_subject, args.target, args.criterion)
    cov_one = np.linalg.inv(I_subject)
    print("Per-subject Fisher matrix (battery sum):")
    print(I_subject)
    print(f"Eigenvalues: {np.linalg.eigvalsh(I_subject)}")
    print(f"tr(I_subject^-1): {np.trace(cov_one):.10g}")
    print(f"N_subjects: {n}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
