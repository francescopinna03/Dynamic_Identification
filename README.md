# Dynamic Identification of Behavioral Reasoning in Finite Games

Replication package for the working paper *Dynamic Identification of Behavioral
Reasoning in Finite Games* (Francesco Pinna, LUISS Guido Carli).

Preprint available at https://ssrn.com/abstract=7251718 or http://dx.doi.org/10.2139/ssrn.7251718
Contact: [francesco.pinna@studenti.luiss.it](mailto:francesco.pinna@studenti.luiss.it)

The results of the first version of the paper (August 2026) were produced at
tag
[`v1.0`](https://github.com/francescopinna03/Dynamic_Identification/tree/v1.0).
The revised version (November 2026) leaves those results unchanged and adds
the verification scripts for its new sections, the subject bootstrap, the
out-of-fold evaluation, and the leave-one-game-out table, at tag
[`v2.1`](https://github.com/francescopinna03/Dynamic_Identification/tree/v2.1).
Tag
[`v2.0`](https://github.com/francescopinna03/Dynamic_Identification/tree/v2.0)
is an earlier snapshot of the same revision and is superseded by `v2.1`.
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

The last stage of `run_all.py` runs four verification scripts and writes their
transcripts to `output/verification/`. It requires CVXPY with the Clarabel
solver, both pinned in `requirements.txt`, and can be skipped with
`--skip-verification`. During the coverage experiment of
`verify_identified_sets.py` CVXPY may warn that a solution may be inaccurate.
The script counts those solves and prints the count in its transcript.

The subject bootstrap is not part of `run_all.py`, because it recomputes the
whole empirical chain on each of 1,999 resamples of the subjects. After a full
run it is launched separately:

```bash
python code/empirical/bootstrap_subjects.py --root .
```

It runs under seed 42 on all cores but one unless `--workers` is given, and
writes `bootstrap_subjects.json` and `bootstrap_subjects_replicates.csv` to
`output/empirical/`.

Python version: see `.python-version`. Pinned library versions: see
`requirements.txt` and `environment.json`, which records the interpreter and
library versions under which the reported numbers were produced. The reported
residuals reach the order of 1e-15, so results at the last digits are sensitive
to the linear-algebra backend, and `environment.json` is what makes a
discrepancy diagnosable. The verification transcripts were produced under
Python 3.14.0 on Linux x86-64 with the pinned versions, recorded in the
`verification` entry of `environment.json`. The stored bootstrap and
out-of-fold outputs were produced on macOS arm64. A run of the out-of-fold
evaluation and of the first 399 bootstrap replicates on Linux x86-64 agrees
with them within 3e-5.

Approximate runtime: Observed runtime on a MacBook Air with an Apple M2 processor and 8 GB of memory, using macOS 26.5.2 and Python 3.14.0: 1 minute 40.45 seconds for a fresh run including data acquisition, and 1 minute 4.77 seconds with the archives already downloaded (`--skip-download`). Environment creation and dependency installation are excluded. These timings predate the verification stage and the two stages that follow the cross-fitted analysis, the leave-one-game-out table and the out-of-fold evaluation, and correspond to `--skip-verification`. On a Linux x86-64 machine the out-of-fold evaluation took about twenty seconds, the verification stage about five minutes, and one bootstrap replicate under twenty seconds on one core.

## Layout

```
code/theoretical/        finite-grid Wright-Fisher / Schrodinger-Bass bridge,
                         entropic central path, auction fixed point, design
                         audit, sample-size accounting, numerical checks,
                         verification scripts of the revised version
code/empirical/          acquisition, preprocessing, pooled estimation,
                         cross-fitted classification, leave-one-game-out
                         scoring, permutations, MouseLab process contrasts,
                         subject bootstrap, out-of-fold evaluation
code/tests/              self-tests (leakage, perfect-signal auction limit,
                         per-subject Fisher accounting)
code/data_manifest.json  hashes and dimensions of the source archives and of
                         the reconstructed data
code/verify_data_manifest.py
                         checks the data against that manifest
artifact_manifest.json   hashes of the 26 stored reference outputs
environment.json         interpreter and library versions used for the paper
output/                  results produced by a full run and by the bootstrap
reference_outputs/       the same results as reported in the paper
```

## From the paper to the code

Section, table, and appendix numbers refer to the revised version of the
paper.

| Paper result | Produced by | Reported artifact |
| --- | --- | --- |
| Tables 1, 11, 12 and 13, design audit, battery progression, primitive slices, parameter landscape | `code/theoretical/design_games.py`, `code/theoretical/auction.py` | `reference_outputs/theoretical/design_games.json` |
| Table 2, sample sizes | `code/theoretical/sample_size.py` | `reference_outputs/theoretical/design_games.json` |
| Table 3, regular / active-face / game-anchored bridges | `code/theoretical/conditional_wf_sbb.py`, `code/theoretical/robustness_wf_sbb.py` | `output/robustness_wf_sbb.json`, `.csv` |
| Tables 4 and 14, efficient endpoint information for depth | `code/empirical/fisher_cgc.py` | `reference_outputs/empirical/fisher_diagnostics.csv` |
| Table 5, cross-fitted sorting | `code/empirical/classify_subjects_crossfit.py`, `code/empirical/stratified_cv.py`, `code/empirical/robustness.py` | `reference_outputs/empirical/stratified_cv_crossfit.csv`, `subject_types_crossfit.csv`, `robustness.txt` |
| Table 5, subject bootstrap intervals and shares of nonpositive slopes | `code/empirical/bootstrap_subjects.py` | `reference_outputs/empirical/bootstrap_subjects.json`, `bootstrap_subjects_replicates.csv` |
| Table 5, out-of-fold slopes | `code/empirical/out_of_fold_information.py` | `reference_outputs/empirical/out_of_fold_information.json`, `out_of_fold_information_games.csv` |
| Table 6, leave-one-game-out slopes, with the slopes of Table 5 that omit game 1 | `code/empirical/leave_one_game_out.py` | `reference_outputs/empirical/leave_one_game_out.csv` |
| Table 7, MouseLab process contrasts | `code/empirical/process_tracing_crossfit.py` | `reference_outputs/empirical/process_tracing_crossfit_games.csv`, `process_tracing_crossfit.json` |
| Table 9, mesh and time robustness | `code/theoretical/robustness_wf_sbb.py` | `output/robustness_wf_sbb.csv` |
| Table 10, log-domain path and information decomposition | `code/theoretical/log_domain_boundary_layer.py`, `code/theoretical/central_path_transition.py` | `output/log_domain_boundary_layer.json`, `output/log_domain_information_path.csv`, `output/central_path_transition.json` |
| Section 7.2, pooled maximum likelihood | `code/empirical/mle_cgc.py` | `reference_outputs/empirical/mle_pooled.json` |
| Numerical checks of the theorem identities | `code/theoretical/verify_math.py` | `reference_outputs/theoretical/math_checks.json` |
| Appendix D, two-scale expansion, local limit, Bures and Fisher-Rao forms, shadow price | `code/theoretical/verify_two_scale.py` | `reference_outputs/verification/verify_two_scale.txt` |
| Sections 2.2, 3.1, 3.2 and 5.2 and Appendices B and D, causal counterexample, budget interval, quadrature term, innovation energy, depth variance, sender-receiver determinant, reservation-price identities | `code/theoretical/verify_revision_checks.py` | `reference_outputs/verification/verify_revision_checks.txt` |
| Sections 3.3 and 4 and Appendix G, efficient information under nested correction classes, intervention information, budgeted instance, coverage, boundary information, elbow exponents | `code/theoretical/verify_identified_sets.py` | `reference_outputs/verification/verify_identified_sets.txt` |
| Appendix G, occupation moments and split-sample estimator | `code/theoretical/verify_repeated_paths.py` | `reference_outputs/verification/verify_repeated_paths.txt` |

Table 8 is a conceptual mapping of channels to measurements and has no
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

The mutation test holds fixed the precision at which the type likelihoods are
evaluated, which in the reported analysis is the estimate obtained on all
sixteen games, so that it certifies the absence of own-outcome leakage at a
given precision. `out_of_fold_information.py` replaces that precision, and the
point at which the information of each game is evaluated, by the estimates
obtained without the game, and reports the resulting slopes together with the
number of classifications that change.

The verification scripts print, under fixed seeds, the figures quoted in
Sections 3 and 4 and in Appendices D and G: the expansion of the
sampled Gaussian entropy and its local limit, the budget interval, the
identities of the reservation-price module, the profiled cost, the identified
set, and the boundary information of the budgeted instance with its Monte
Carlo coverage, and the occupation moments and split-sample rates of the
repeated-path experiment.

These are implementation checks. The analytical arguments are in the paper, and
none of these tests validates the behavioral specification itself, which is the
object of the empirical analysis rather than a premise of it.

## Citation

```
Pinna, F. (2026). Dynamic Identification of Behavioral Reasoning in Finite
Games. Available at SSRN: https://ssrn.com/abstract=7251718 or http://dx.doi.org/10.2139/ssrn.7251718
```

## License

The MIT License in `LICENSE` applies to the software in `code/` and
`run_all.py`. The paper itself is not covered by it. The Costa-Gomes and
Crawford data remain under the terms of their original distribution and are not
redistributed here.
