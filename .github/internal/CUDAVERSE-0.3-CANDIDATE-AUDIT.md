# Internal cudaverse 0.3 candidate completion audit

This is the requirement-by-requirement completion record for the 0.3
development line. It is intentionally stricter than a roadmap: a requirement
is `proven` only when current authoritative evidence covers its full scope.
Passing a narrow test or merging a checkpoint does not prove the final
candidate.

Status vocabulary:

- `proven`: exact evidence covers the stated requirement at its named source;
- `in progress`: implementation or evidence is actively being completed;
- `pending`: required evidence has not yet been produced for a candidate; and
- `evidence-deferred`: a bounded non-critical item has a written rationale and
  cannot be presented as completed.

## Current branch and release boundaries

| Requirement | Status | Authoritative evidence | Remaining work |
|---|---|---|---|
| Freeze `main` and `release/cran-0.1.0` | proven | Remote refs both equal `59e15c8c5a56d26e09a594886c875b1b8249f6f9` | Recheck before the final decision. |
| Develop only through reviewed checkpoints | proven at current checkpoint | `develop/native-cuda` remote is `acd105fe8ef0ed8703e9e16f9ca4ce68a0d3db7d`; CP-01 through CP-05 are recorded in `.github/internal/CUDAVERSE-0.3-CHECKPOINTS.md` | Merge CP-06 and CP-07 only after their exact gates pass. |
| No release-side action without approval | proven so far | No 0.3 tag, release, main Pages deployment, or CRAN/Bioconductor submission exists in this work | Stop again for approval after the final decision report. |
| One optional, lazy, lightweight package backend | proven at current checkpoint | Registry contract, redistribution gate, SBOM, artifact checks, and installed-footprint evidence | Re-run all gates on the exact final candidate. |

## Milestone coverage

| Goal requirement | Status | Current evidence | What still prevents final completion |
|---|---|---|---|
| Ordered roadmap, export/API audit, CI and footprint baseline | proven | `CUDAVERSE-0.3-ROADMAP.md`, capability matrix, baseline section of the checkpoint log | Refresh counts and remote state at the final candidate. |
| Native dense object semantics, ownership, errors, interruption, and lifecycle | proven at CP-01 | PR 16 evidence, exact RTX test source, reproducible PTX, 1,000 gather/scatter cycles, and package check recorded in the checkpoint log | Repeat the complete RTX gate on the final candidate. |
| Resident PCA/SVD prediction, distance, stable top-k/kNN, and k-means | proven at CP-02/current dense surface | Native Phase 2 evidence, CP-02 resident k-means evidence, CP-04 public conformance | Final full workload benchmark and final-candidate RTX repetition. |
| COO/CSR sparse operations, normalization, multiplication, reductions, transpose, PCA/kNN preparation | proven for the current float64 sparse surface; float32 is evidence-deferred | Sparse Phase 3 evidence, CP-03 transpose evidence, CP-04 conformance and 1,000-cycle sparse lifecycle; `.github/internal/CUDAVERSE-0.3-SPARSE-FLOAT32-DECISION.md` records the bounded dtype decision and re-entry gates | Complete the full float64 sparse benchmark and repeat the sparse gates on the final candidate. |
| One base/torch/native public conformance suite for every export and failure boundary | proven at CP-04 | PR 19, 38-export capability matrix, public workflow suite, no-skip RTX package gate | Repeat on the final candidate and confirm no exports changed without matrix coverage. |
| Matmul and dense/sparse PCA-kNN benchmark contract | proven at CP-06 evidence head, pending final-candidate repeat | Complete 12-case/36-result report from clean source `42d6ab4` at `inst/reports/benchmarks/CP06-FULL.json` (SHA-256 `9363aba0c4f5af0ee7ef84648bc2d6bec2177c7deccfd445454f9f7bdda43e93`); every numerical gate passes; generated summary and both checker logs are retained beside it | Rerun the same full contract only for the exact final candidate, then bind that report into the final evidence manifest. |
| Cross-platform packaging, ABI/PTX, SBOM, license, and small artifacts | proven at checkpoint, pending for final | CP-05 CI and all 11 required checks on CP-07 head `c6132b565e344235ff0110ccdc29668bfb46b13d` cover Windows, macOS, Ubuntu, R-devel, artifacts, ABI/PTX, supply chain, CPU contract, and pkgdown | Repeat on the final merged candidate and retain exact run/artifact identifiers. |
| README, NEWS, reference, examples, diagnostics, troubleshooting, and pkgdown alignment | proven at CP-07 head, pending final-candidate repeat | CP-07 draft PR 21 at `c6132b565e344235ff0110ccdc29668bfb46b13d` adds native-first troubleshooting; an isolated local pkgdown build and browser review passed after moving maintainer records out of the public site and repairing rendered links; its exact-head pkgdown CI also passed the public-boundary checker | Run the post-benchmark local source check, merge CP-07 after CP-06, then re-audit rendered final pages and benchmark claims. |
| Exact review-ready 0.3 candidate and release decision | pending | None yet; checkpoint evidence cannot substitute for final-candidate evidence | Merge remaining checkpoints, choose candidate version, build one exact tarball, run all final gates, and write the decision report. |

## Final candidate evidence manifest

The final decision cannot be `release` unless one exact clean candidate commit
has all of the following retained together:

1. source tarball, SHA-256, source commit, package version, and clean-tree
   proof;
2. local Windows build/check log plus matching Windows, macOS, Ubuntu, and
   R-devel GitHub checks;
3. Windows and Linux no-CUDA installable artifacts and their checksums;
4. CUDA 12.8.1 ABI build and byte-reproducible PTX evidence;
5. CycloneDX SBOM, third-party license inventory, and zero bundled
   NVIDIA/LibTorch runtime result;
6. exact RTX 2000 base/torch/native conformance, structured recovery,
   interruption, backend reuse, and no-skip hardware report;
7. 1,000-cycle dense and sparse lifecycle evidence with no double-free and no
   more than 1 MiB post-cleanup whole-device difference;
8. complete `cudaverse-benchmark/1` JSON, its SHA-256, machine checker output,
   generated human summary, numerical validation, memory provenance, and
   workload-specific interpretation;
9. rendered reference, vignettes, troubleshooting, examples, and pkgdown
   evidence that agrees with actual backend behavior; and
10. a release/defer/reduce-scope report that names every remaining limitation
    and stops before any tag, release, deployment, or repository submission.

The final JSON manifest uses schema `cudaverse-candidate-evidence/1` and must
pass `tools/check-candidate-evidence.R`. The checker requires every source,
check, artifact, RTX report, benchmark, documentation render, and decision to
identify the same 40-character candidate commit. Its positive and rejection
cases run in CI through `tools/test-candidate-evidence.R`. Every retained file
must also exist below the manifest directory and match a freshly computed
SHA-256, including the benchmark checker logs, pkgdown render log, and final
decision report. An outside path, tampered artifact, or plausible-looking
manifest assembled from different checkpoints is therefore not final evidence.
The checker also parses the CycloneDX component, complete full benchmark JSON,
RTX report, and candidate decision report to require their internal schema or
fixed metadata, clean source commit, package version, outcome, limitations, and
pass/release-boundary state to match the candidate rather than trusting
manifest metadata alone. The required decision structure is retained in
`.github/internal/CUDAVERSE-0.3-CANDIDATE-DECISION-TEMPLATE.md`.
`tools/build-candidate-evidence.R` assembles the manifest from a declarative
`cudaverse-candidate-evidence-input/1` file located in the evidence bundle. It
reads the clean `develop/native-cuda` commit and 0.3 version directly from the
worktree, rechecks both frozen remote refs, computes every retained file hash
and artifact size, and derives RTX/benchmark pass fields from their JSON rather
than copying operator-entered booleans. It validates a staging manifest with
the normal checker before atomically installing the final file and refuses to
overwrite an existing manifest.
The retained local check, benchmark checker, summary checker, and pkgdown
render logs must also contain their stable success markers, and the retained
license inventory must name CUDA Driver, cuBLAS, cuSOLVER, and PTX. A matching
SHA therefore cannot bless a retained failure or an incomplete inventory.

### Final RTX report chain

The final hardware report is produced inside this package repository and has
three retained layers. `tools/run-phase4-report.R` records the RTX 2000
consolidation workload, including parity, residency, structured recovery,
interruption, and dense/sparse lifecycle evidence.
`tools/run-native-package-tests.R` independently runs the complete package
test suite with `CUDAVERSE_NATIVE_TESTS=true`; any failure, error, skip,
unavailable native runtime, non-native automatic selection, or dirty source
fails the gate while still retaining JSON diagnostics. Finally,
`tools/build-native-candidate-report.R` accepts those two reports only when
their clean commit and package version match, records both SHA-256 values, and
emits `cudaverse-native-candidate/1` only when every candidate gate passes.

The hardware sequence for an exact candidate is therefore:

1. install that exact clean candidate on the RTX 2000 machine;
2. run `run-phase4-report.R` and retain its JSON;
3. run `run-native-package-tests.R` with its report path outside the source
   tree, so the evidence file itself cannot dirty the candidate;
4. build and check the final report with
   `build-native-candidate-report.R` and
   `check-native-candidate-report.R`; and
5. copy all three reports into the final evidence bundle before creating and
   validating the candidate manifest.

This package-owned chain is the final authority. The current organization
reusable hardware workflow is not yet aligned with this exact no-skip report
contract, so it remains a final-candidate audit item and is not treated as
proof until that alignment is reviewed and rerun on the candidate commit.

## Final decision rule

`release` is available only when every release-critical row above is proven on
the same candidate. `defer` is required when a release-critical proof is
missing or contradicted. `reduce scope` is available only when the removed
claim or feature is explicitly documented, tested as unsupported or hybrid,
and does not weaken the public correctness contract.
