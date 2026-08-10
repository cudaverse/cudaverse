# cudaverse 0.3 candidate completion audit

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
| Develop only through reviewed checkpoints | proven at current checkpoint | `develop/native-cuda` remote is `acd105fe8ef0ed8703e9e16f9ca4ce68a0d3db7d`; CP-01 through CP-05 are recorded in `CUDAVERSE-0.3-CHECKPOINTS.md` | Merge CP-06 and CP-07 only after their exact gates pass. |
| No release-side action without approval | proven so far | No 0.3 tag, release, main Pages deployment, or CRAN/Bioconductor submission exists in this work | Stop again for approval after the final decision report. |
| One optional, lazy, lightweight package backend | proven at current checkpoint | Registry contract, redistribution gate, SBOM, artifact checks, and installed-footprint evidence | Re-run all gates on the exact final candidate. |

## Milestone coverage

| Goal requirement | Status | Current evidence | What still prevents final completion |
|---|---|---|---|
| Ordered roadmap, export/API audit, CI and footprint baseline | proven | `CUDAVERSE-0.3-ROADMAP.md`, capability matrix, baseline section of the checkpoint log | Refresh counts and remote state at the final candidate. |
| Native dense object semantics, ownership, errors, interruption, and lifecycle | proven at CP-01 | PR 16 evidence, exact RTX test source, reproducible PTX, 1,000 gather/scatter cycles, and package check recorded in the checkpoint log | Repeat the complete RTX gate on the final candidate. |
| Resident PCA/SVD prediction, distance, stable top-k/kNN, and k-means | proven at CP-02/current dense surface | Native Phase 2 evidence, CP-02 resident k-means evidence, CP-04 public conformance | Final full workload benchmark and final-candidate RTX repetition. |
| COO/CSR sparse operations, normalization, multiplication, reductions, transpose, PCA/kNN preparation | proven for the current float64 sparse surface | Sparse Phase 3 evidence, CP-03 transpose evidence, CP-04 conformance and 1,000-cycle sparse lifecycle | Complete full sparse benchmark; make and document the evidence-based float32 sparse decision. |
| One base/torch/native public conformance suite for every export and failure boundary | proven at CP-04 | PR 19, 38-export capability matrix, public workflow suite, no-skip RTX package gate | Repeat on the final candidate and confirm no exports changed without matrix coverage. |
| Matmul and dense/sparse PCA-kNN benchmark contract | in progress | CP-05 contract/smoke; full matmul and dense 1k/10k results complete from clean source `acd105f`; reproducible summary/checker commit `2d8a010` | Finish dense 50k and every sparse case; validate exact JSON; investigate regressions; retain JSON/SHA/summary. |
| Cross-platform packaging, ABI/PTX, SBOM, license, and small artifacts | proven at checkpoint, pending for final | CP-05 CI and green CP-07 draft checks cover Windows, macOS, Ubuntu, R-devel, artifacts, ABI/PTX, supply chain, and pkgdown | Repeat at the final merged candidate and retain exact run/artifact identifiers. |
| README, NEWS, reference, examples, diagnostics, troubleshooting, and pkgdown alignment | in progress | CP-07 draft PR 21 at `d1570134e84a9d80634380f0c0d73c41918c6fa5` is green and adds native-first package troubleshooting | Run the post-benchmark local source check, merge CP-07, then re-audit rendered final pages and benchmark claims. |
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

## Final decision rule

`release` is available only when every release-critical row above is proven on
the same candidate. `defer` is required when a release-critical proof is
missing or contradicted. `reduce scope` is available only when the removed
claim or feature is explicitly documented, tested as unsupported or hybrid,
and does not weaken the public correctness contract.
