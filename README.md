# Dynamic Identification of Behavioral Reasoning in Finite Games

Replication package for the working paper *Dynamic Identification of Behavioral
Reasoning in Finite Games* (Francesco Pinna, LUISS Guido Carli).

Paper: forthcoming on SSRN; this line will carry the link once the preprint is
live.
Contact: [francesco.pinna@studenti.luiss.it](mailto:francesco.pinna@studenti.luiss.it)

The results reported in the paper were produced at tag
[`v1.0`](https://github.com/francescopinna03/Dynamic_Identification/tree/v1.0).
Later commits on `main` may refine documentation without changing those
results.

The package covers four things that are kept deliberately separate: data
provenance, numerical checks of the mathematical statements, optimization
certificates for the finite-grid bridge, and leakage tests for the empirical
reanalysis. Each can be run and inspected on its own.

## Data

The Costa-Gomes and Crawford (2006) archive is **not redistributed here**. The
acquisition script downloads the public archives, records their SHA-256 digests
in `CHECKSUMS.txt`, and extracts the source files. The preprocessing and
parsing scripts reconstruct the four processed tables.
`code/verify_data_manifest.py` then checks the raw archives and reconstructed
tables against `code/data_manifest.json`, including file hashes, dimensions,
and required source files.

This step is mandatory for rerunning the empirical analysis. The theoretical
audits and the stored reference outputs can be inspected independently.

## Quick start

```bash
git clone https://github.com/francescopinna03/Dynamic_Identification.git
cd Dynamic_Identification
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_all.py
```

`run_all.py` performs the acquisition step itself. To run it separately, for
instance behind a proxy or to inspect the downloaded archives before anything
else touches them:

```bash
python code/empirical/acquire_cgc2006.py
python run_all.py --skip-download
```

Python version: see `.python-version`. Pinned library versions: see
`requirements.txt` and `environment.json`, which records the interpreter and
library versions under which the reported numbers were produced. The reported
residuals reach the order of 1e-15, so results at the last digits are sensitive
to the linear-algebra backend, and `environment.json` is what makes a
discrepancy diagnosable.

Approximate runtime: Observed runtime on a MacBook Air with an Apple M2 processor and 8 GB of memory, using macOS 26.5.2 and Python 3.14.0: 1 minute 40.45 seconds for a fresh run including data acquisition, and 1 minute 4.77 seconds with the archives already downloaded (`--skip-download`). Environment creation and dependency installation are excluded.

## Layout

```
code/theoretical/        finite-grid Wright-Fisher / Schrodinger-Bass bridge,
                         entropic central path, auction fixed point, design
                         audit, sample-size accounting, numerical checks
code/empirical/          acquisition, preprocessing, pooled estimation,
                         cross-fitted classification, leave-one-game-out
                         scoring, permutations, MouseLab process contrasts
code/tests/              self-tests (leakage, perfect-signal auction limit,
                         per-subject Fisher accounting)
code/data_manifest.json  hashes and dimensions of the source archives and of
                         the reconstructed data
code/verify_data_manifest.py
                         checks the data against that manifest
artifact_manifest.json   hashes of the 17 stored reference outputs
environment.json         interpreter and library versions used for the paper
output/                  results produced by a full run
reference_outputs/       the same results as reported in the paper
```

## From the paper to the code

| Paper result | Produced by | Reported artifact |
| --- | --- | --- |
| Table 1, regular / active-face / game-anchored bridges | `code/theoretical/conditional_wf_sbb.py`, `code/theoretical/robustness_wf_sbb.py` | `output/robustness_wf_sbb.json`, `.csv` |
| Table 2, mesh and time robustness | `code/theoretical/robustness_wf_sbb.py` | `output/robustness_wf_sbb.csv` |
| Table 3, log-domain path and information decomposition | `code/theoretical/log_domain_boundary_layer.py`, `code/theoretical/central_path_transition.py` | `output/log_domain_boundary_layer.json`, `output/log_domain_information_path.csv`, `output/central_path_transition.json` |
| Tables 4, 5, 6 and 12, design audit, battery progression, primitive slices, parameter landscape | `code/theoretical/design_games.py`, `code/theoretical/auction.py` | `reference_outputs/theoretical/design_games.json` |
| Table 7, sample sizes | `code/theoretical/sample_size.py` | `reference_outputs/theoretical/design_games.json` |
| Tables 8 and 13, efficient endpoint information for depth | `code/empirical/fisher_cgc.py` | `reference_outputs/empirical/fisher_diagnostics.csv` |
| Table 9, cross-fitted sorting | `code/empirical/classify_subjects_crossfit.py`, `code/empirical/stratified_cv.py`, `code/empirical/robustness.py` | `reference_outputs/empirical/stratified_cv_crossfit.csv`, `subject_types_crossfit.csv`, `robustness.txt` |
| Table 10, MouseLab process contrasts | `code/empirical/process_tracing_crossfit.py` | `reference_outputs/empirical/process_tracing_crossfit_games.csv`, `process_tracing_crossfit.json` |
| Section 12.6, pooled maximum likelihood | `code/empirical/mle_cgc.py` | `reference_outputs/empirical/mle_pooled.json` |
| Numerical checks of the theorem identities | `code/theoretical/verify_math.py` | `reference_outputs/theoretical/math_checks.json` |

Table 11 is a conceptual mapping of channels to measurements and has no
computational counterpart.

## What the audits certify

The numerical checks validate representative finite-dimensional identities used
by the analytical arguments, including the auction contraction certificate, the
Wright-Fisher-Bures feedback solution, the singular-jet KL coefficient, and
terminal copula assembly.

The optimization certificates check exact flow conservation, the relative
covariance cone, inward drift, analytic first and second derivatives,
reduced-Hessian curvature, conditional score centering, and the
martingale-difference information decomposition. Analytic derivatives in the
cursing parameter are compared against two full reoptimizations at each of the
sixteen entropy weights, and the reconstructed minimum flow is the exponential
of a finite log-domain multiplier rather than a raw primal coordinate.

The empirical self-tests check three things: mutating every held-out choice in
a fold leaves that fold's training classifications unchanged, a perfectly
revealing signal makes conditional auction cursing collapse to the correct
conditional law, and the sample-size routine sums per-subject Fisher matrices
without any later division by the number of games.

These are implementation checks. The analytical arguments are in the paper, and
none of these tests validates the behavioral specification itself, which is the
object of the empirical analysis rather than a premise of it.

## Citation

<!-- Coming soon! -->

```
Pinna, F. (2026). Dynamic Identification of Behavioral Reasoning in Finite
Games. Working paper.
```

## License

The MIT License in `LICENSE` applies to the software in `code/` and
`run_all.py`. The paper itself is not covered by it. The Costa-Gomes and
Crawford data remain under the terms of their original distribution and are not
redistributed here.
