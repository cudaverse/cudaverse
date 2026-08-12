# Internal cudaverse 0.4 candidate completion audit

This is the requirement-by-requirement completion record for the 0.4 line.
A merged checkpoint proves only the named source and gate. It does not prove
the final candidate until all release-critical evidence identifies one exact
clean commit.

Status vocabulary:

- `proven`: exact evidence covers the full stated requirement;
- `proven at current checkpoint`: checkpoint evidence is complete but must be
  repeated on the final candidate;
- `in progress`: implementation or evidence is still being completed;
- `pending`: candidate evidence does not yet exist; and
- `evidence-deferred`: a bounded non-critical item has a written rationale and
  cannot be presented as completed.

## Current source and release boundaries

| Requirement | Status | Authoritative evidence | Remaining work |
|---|---|---|---|
| Preserve the exact 0.3 candidate | proven | External manifest `cudaverse/evidence/0.3.0.9000-b0ab90f/candidate-manifest.json` revalidates source `b0ab90f8547d42a5244963bc6b15d0ba223382ff`; its retained manifest SHA-256 is recorded in `CUDAVERSE-0.4-ROADMAP.md` | Do not rewrite or replace the 0.3 bundle. |
| Freeze `main` and `release/cran-0.1.0` | proven at current checkpoint | Local, remote-tracking, and directly resolved remote refs equal `59e15c8c5a56d26e09a594886c875b1b8249f6f9` | Resolve the remote refs again immediately before candidate assembly. |
| Develop through short reviewed branches | proven at current checkpoint | `develop/0.4` is `deb77127eebf68e0c6f788e9e10c40c5cb8dacfc`; CP-01 through CP-07A and their reviewed evidence are recorded in `CUDAVERSE-0.4-CHECKPOINTS.md` | Merge CP-07B and candidate-readiness work before selecting one clean candidate. |
| Keep one optional, lazy, lightweight user package | proven at current checkpoint | Registry, no-CUDA install, artifact-size, redistribution, SBOM, license, and public capability gates | Repeat every packaging and supply-chain gate on the exact candidate. |
| Take no external release action without approval | proven so far | Development work has not created a 0.4 tag, release, Pages deployment, CRAN submission, or Bioconductor submission | Stop for explicit approval after the candidate decision report. |

## Milestone coverage

| Goal requirement | Status | Current evidence | What still prevents final completion |
|---|---|---|---|
| Baseline audit, ordered roadmap, setup, diagnostics, capabilities, fallback, and structured errors | proven at current checkpoint | `CUDAVERSE-0.4-ROADMAP.md`, CP-01, capability matrix, diagnostics tests, strict explicit-CUDA and no-CUDA gates | Repeat export counts, diagnostics, source checks, and public documentation on the exact candidate. |
| Remove avoidable dense tensor host round trips while preserving ownership, dtype, dimnames, interruption, and provenance | proven at current checkpoint for the implemented surface | CP-02A/B/C/D cover allocation-free reshape and contiguous subset views, replacement conversion, reconstruction casts, shared ownership, and error/lifecycle gates | Re-run the complete conformance and RTX suites and confirm every final export remains covered. |
| Strengthen resident dense reductions, SVD/PCA, distance, stable top-k/kNN, and k-means | proven at current checkpoint for the current dense surface | CP-03A/B/C/D cover explicit batching, bounded-memory k-means, resident PCA prediction, and native SVD/PCA input; prior parity gates cover reductions, matmul, stable ties, and resident PCA/kNN | Complete the full benchmark and repeat parity, residency, memory, and deterministic-tie gates on the exact candidate. |
| Optimize sparse COO/CSR, normalization, ownership, rematerialization, Matrix interop, and PCA/kNN preparation | proven at current checkpoint for the float64 sparse surface; sparse float32 remains evidence-deferred | CP-04A/B/C/D plus the capability matrix and lifecycle tests cover resident normalization, shared pattern ownership, direct COO rematerialization, and Matrix-free preprocessing | Complete full sparse benchmark evidence and repeat sparse parity and 1,000-cycle lifecycle gates on the exact candidate. Do not add float32 without its written re-entry evidence. |
| Audit graph, clustering, embeddings, diffusion, and SingleCellExperiment workflow boundaries | proven at current checkpoint | CP-05A proves the resident PCA-to-diffusion-distance boundary; CP-09A and `CUDAVERSE-0.4-WORKFLOW-BOUNDARIES.md` bind every higher-level export and optional object input to its intentional CPU/hybrid, host-output, provenance, and re-entry contract | Repeat public conformance, optional-input, capability-matrix, and rendered-documentation gates on the exact candidate; accelerate no stage without parity, residency, and memory evidence. |
| Expand error recovery, OOM, interruption, shared views, repeated release, backend reuse, sessions, lifecycle, and VRAM observability | proven at current checkpoint | CP-06A/B provide public memory telemetry and two independent native-session lifecycle reports; earlier RTX gates cover injected errors, interruption, reuse, and dense/sparse 1,000-cycle cleanup | Repeat the full no-skip RTX chain on the exact candidate. |
| Reproducible smoke/full benchmark with timing, transfer, allocator, VRAM, footprint, and provenance | in progress | CP-07A retains the complete public-memory smoke contract; CP-07B is producing the 12-case/36-backend full report from exact source `deb7712` | Require one strictly complete report, both machine checkers, generated summary, exact hashes, and workload-specific interpretation before candidate selection. |
| Cross-platform checks, reproducible PTX/ABI, SBOM, licenses, compact no-CUDA artifacts, and aligned reference/pkgdown | proven at current checkpoint, pending for final | Every merged checkpoint has kept Windows, macOS, Ubuntu, R-devel, artifact, CUDA 12.8.1, supply-chain, source, and public-documentation gates green | Repeat and retain the complete check/run/artifact chain for the exact final commit. |
| Exact review-ready 0.4 candidate and decision | pending | Candidate-policy tooling accepts only supported branch/version pairs while retaining validation of the frozen 0.3 manifest | Merge all required evidence, freeze one clean `develop/0.4` commit, assemble its evidence bundle, validate the manifest, and write the decision report. |

## Exact candidate evidence contract

One candidate commit must retain all of the following together:

1. source tarball, SHA-256, source commit, package version, branch, and
   clean-tree proof;
2. local source/check log and matching Windows, macOS, Ubuntu, and R-devel
   GitHub checks;
3. Windows and Linux no-CUDA artifacts, checksums, install checks, and compact
   installed-size evidence;
4. CUDA 12.8.1 ABI build and byte-reproducible PTX evidence;
5. CycloneDX SBOM, third-party license inventory, and zero bundled NVIDIA
   runtime and LibTorch bytes;
6. exact RTX 2000 base/torch/native conformance, structured recovery,
   interruption, backend reuse, independent-session, and no-skip reports;
7. dense and sparse 1,000-cycle lifecycle evidence with no double-free,
   invalid view, retained tracked allocation, or more than 1 MiB post-cleanup
   whole-device difference;
8. complete 12-case/36-result `cudaverse-benchmark/1` JSON, generated summary,
   both checker logs, numerical validation, memory provenance, and exact hashes;
9. rendered reference, examples, vignettes, troubleshooting, capability
   matrix, and pkgdown-boundary evidence matching actual behavior; and
10. a release, defer, or reduce-scope decision naming every remaining
    limitation and recording that no external release action was taken.

The bundle must pass `tools/check-candidate-evidence.R`. The checker validates
retained files and their internal metadata rather than trusting manifest
booleans. It accepts the historical `0.3.0[.9000]` plus
`develop/native-cuda` pair and the current `0.4.0[.9000]` plus `develop/0.4`
pair; a cross-line branch/version combination fails closed. Frozen release
refs remain exact hard gates for assembly.

## Final decision rule

`release` is available only when every release-critical row above is proven
on the same candidate. `defer` is required when a release-critical proof is
missing or contradicted. `reduce scope` is available only when the removed
claim or feature is documented and tested as unsupported or hybrid without
weakening public correctness. In every case, stop before tag, release, Pages,
CRAN, or Bioconductor action until the maintainer gives explicit approval.
