# cudaverse 0.3 checkpoint log

## Baseline audit

- Candidate line: `develop/native-cuda` at
  `1b1707366f06a9667063f1703f6470117d517320`.
- Frozen lines: `main` and `release/cran-0.1.0` at
  `59e15c8c5a56d26e09a594886c875b1b8249f6f9`.
- Surface: 38 exports, 27 S3 registrations, and a row-per-export backend
  capability matrix enforced by tests and `tools/check-capability-matrix.R`.
- Local baseline: complete CPU suite passed; native cases were correctly gated
  behind `CUDAVERSE_NATIVE_TESTS=true`.
- GitHub baseline: no open issues or pull requests; the preceding checkpoint
  had green Windows/macOS/Ubuntu/R-devel, CPU contract, pkgdown, supply-chain,
  artifact, CUDA ABI, and reproducible PTX checks.
- Payload: 123 tracked files and about 1.1 MB of tracked data, including about
  166 KB of PTX and 253 KB of retained reports. No CUDA Toolkit, NVIDIA runtime,
  LibTorch, or other large binary payload was present.

### Audit findings

1. CUDA tensor `[` and `[<-` were the remaining full-value host round trips in
   the dense object API.
2. Native PCA, distance, and stable top-k already retain their important
   intermediate buffers, but k-means and several graph/embedding stages remain
   explicitly hybrid or CPU-only.
3. Native sparse COO/CSR, normalization, reductions, multiplication, PCA, and
   kNN preparation exist; transpose and broader conformance coverage remain 0.3
   work.
4. Historical reports validate earlier hardware checkpoints but do not prove a
   future 0.3 candidate. Final evidence must be regenerated from one exact
   clean commit.

## CP-01: device-native tensor indexing

Status: completed and merged as PR 16 into `develop/native-cuda`.

Scope:

- add registry operations for tensor gather and replacement;
- preserve ordinary R indexing, drop, dimnames, dtype, recycling, and
  last-write semantics for duplicate replacement indices;
- keep native tensor values and compatible tensor replacements on the GPU;
- retain explicit compatibility fallback for missing indices and backends that
  do not implement indexing;
- extend native self-test, parity tests, capabilities, documentation, and PTX.

Required evidence before merge:

- [x] native bridge compiles on local Windows without a CUDA Toolkit;
- [x] local CPU suite passes without weakening tests;
- [x] CUDA 12.8.1 PTX rebuilt by the pinned CI container; committed SHA-256
  `42f0bf069d4c4b5ec507658f43e01088391508f10e63358a2a4b68d0bd331bd4`
  is recorded in `SHA256SUMS` and the CycloneDX SBOM;
- [x] Windows, macOS, Ubuntu, R-devel, pkgdown, supply-chain, CPU contract,
  CUDA ABI/PTX, and Windows/Linux artifact CI passed on the checkpoint source;
- [x] exact clean commit `888aa9bdb0d9e6961c4f0808b24c83142fe86b87`
  passed the complete native test file on the RTX 2000 with no skipped cases,
  including integer/float32/float64 indexing and device-resident tensor-value
  replacement parity;
- [x] the same RTX run passed structured indexing error recovery and 1,000
  gather/scatter cycles with no tracked leak and at most 1 MiB whole-device
  post-cleanup difference;
- [x] the checkpoint source is committed and synchronized on PR 16 into
  `develop/native-cuda`; all non-hardware PR checks are green, and the exact
  source also passed the local RTX gate above.

Additional local packaging evidence: a full source build generated all three
vignettes and `R CMD check --no-manual` completed with `Status: OK` on R 4.6.0
for Windows.

Deferred beyond CP-01:

- missing-index gather remains a provenance-visible compatibility path;
- native sparse transpose and the broader export-parameterized conformance
  suite remain later milestones.

## CP-02: resident native CUDA k-means

Status: completed and merged as PR 17 into `develop/native-cuda` at
`15d2554fc3955944765f946af37a9cd35d80a501`.

Scope:

- add an optional `algorithm_kmeans` backend-registry operation without
  matching on the literal native backend name;
- upload observations and initial centres once, then keep Euclidean distance,
  lowest-index tie assignment, accumulation, and Lloyd centre updates on the
  device;
- retain an empty centre at its previous value and preserve the established
  final-assignment semantics;
- transfer only the per-iteration centre-movement summary and the compact
  final assignments, centres, and within-cluster sums;
- preserve the existing base and torch compatibility paths and record the
  native device-resident stages through `cudaverse-stage/1`.

Required evidence before merge:

- [x] local Windows C++17 bridge compiles, installs, and loads without a CUDA
  Toolkit;
- [x] the complete local CPU suite passes, including a fake-backend contract
  test proving operation-driven dispatch;
- [x] CUDA 12.8.1 PTX is rebuilt reproducibly; committed SHA-256
  `8daaaf372a164bc91fc1e6ee5b634ba266a4dcb9cc631bb122a117f4d76da7cf`
  agrees with `SHA256SUMS`, the CycloneDX SBOM, and pinned CI output;
- [x] exact implementation/PTX/test commit
  `4b51af28f2064713c9002d6c73832749d18ad75f` passes native parity for
  ordinary, tied/empty-centre, named-dimension, and large-offset inputs on the
  RTX 2000 as part of the complete hardware test file, with no skipped case;
- [x] the same RTX gate passes structured failure recovery and 1,000 resident
  k-means cycles with zero tracked leak and at most 1 MiB whole-device
  post-cleanup difference;
- [x] Windows, macOS, Ubuntu, R-devel, pkgdown, supply-chain, CPU contract,
  CUDA ABI/PTX, and Windows/Linux artifact CI are green on PR 17 source, and
  the exact implementation/PTX/test source passes the local RTX gate above.

Deferred beyond CP-02:

- prediction can still return the complete distance matrix by contract; a
  compact device assignment operation may be added separately after the
  fitting path is accepted;
- PCA and kNN benchmarking remains part of the later unified benchmark
  milestone rather than this functional checkpoint.

## CP-03: same-device sparse transpose

Status: completed and merged as PR 18 into `develop/native-cuda` at
`ccd0da77c5624ac87eac75d4605af59e9a08673e`.

Scope:

- add `t.cudasparse()` without adding another public package or backend name
  branch;
- preserve values, COO/CSR logical format, zero structure, rectangular shape,
  dimnames including axis names, device, backend, and provenance;
- add a native `sparse_transpose` registry operation that builds transposed CSR
  row counts, row pointers, and stable COO-aligned values on the device;
- let compatibility backends rebuild same-device storage from the public COO
  metadata when they do not implement the optional operation;
- keep the source allocation independent and safe across transpose,
  double-transpose, explicit release, errors, and repeated lifecycle cycles.

Required evidence before merge:

- [x] local Windows C++17 bridge compiles, installs, and loads without a CUDA
  Toolkit;
- [x] the complete local CPU suite passes for CSR, COO, rectangular, empty,
  named, double-transpose, provenance, and operation-driven dispatch cases;
- [x] pinned CUDA 12.8.1 PTX is rebuilt reproducibly; committed SHA-256
  `d15dddeb84e8c54ccffc051c299940958b310e1adf4a4f8bc8e6527b75cd4800`
  agrees with `SHA256SUMS`, the CycloneDX SBOM, and pinned CI output;
- [x] exact implementation/PTX commit
  `52ab43b0cda5900082d42fd0f421b4c7df6c4d8a` passes the complete RTX
  native test file with no skipped case, including parity, released-pointer
  recovery, source reuse, and 1,000 transpose/double-transpose lifecycle
  cycles with zero tracked leak and at most 1 MiB whole-device difference;
- [x] Windows, macOS, Ubuntu, R-devel, pkgdown, supply-chain, CPU contract,
  CUDA ABI/PTX, and Windows/Linux artifact checks are green on PR 18 source;
  the exact implementation/PTX source passes the local RTX gate above.

Deferred beyond CP-03:

- the first stable scatter kernel is correctness-first and single-threaded to
  guarantee that device storage and public COO order stay aligned; a parallel
  stable scatter is benchmark-driven follow-up work, not a release claim;
- float32 sparse storage remains evidence-gated and is not introduced here.

## CP-04: executable public-backend conformance

Status: completed and merged as PR 19 into `develop/native-cuda` at
`60d04ae0cf98525f8f596f4c406cef5af4a10e27`.

Scope:

- assign every public export to one executable diagnostics, tensor, sparse,
  algorithm, graph, or embedding contract case in the installed capability
  matrix;
- run identical compact public workflows on base, torch, and native whenever
  protected CUDA coverage is required;
- compare values, stable indices, shapes, dimnames, devices, backend identity,
  provenance, and intentional host/hybrid boundaries;
- require visible automatic fallback, strict explicit CUDA errors, consistent
  invalid-input behavior, and successful backend reuse after those errors;
- retain the deeper native ownership, injected OOM, interruption, structured
  error, and 1,000-cycle lifecycle tests as part of the same no-skip hardware
  package gate.

Required evidence before merge:

- [x] the capability-matrix checker covers all 38 exports and validates every
  `contract_case` value;
- [x] the complete local CPU suite passes with the new base contract cases;
- [x] the focused public conformance suite passes locally on the RTX 2000 for
  base, torch, and native together;
- [x] the complete package-owned hardware suite passes on the RTX 2000 with no
  skipped cases, including all pre-existing recovery and lifecycle tests;
- [x] a local source build and `R CMD check --no-manual` complete with
  `Status: OK`;
- [x] Windows, macOS, Ubuntu, R-devel, pkgdown, supply-chain, CPU contract,
  CUDA ABI/PTX, and Windows/Linux artifact CI pass on the exact PR source.

Deferred beyond CP-04:

- this checkpoint increases correctness evidence but makes no performance
  claim; cold/warm timing, transfer, and peak-VRAM reporting remain the later
  benchmark milestone;
- torch remains a compatibility backend through 0.2 and its future status is
  still an evidence-based 0.3 decision.

## CP-05: reproducible benchmark contract

Status: completed and merged as PR 20 into `develop/native-cuda` at
`acd105fe8ef0ed8703e9e16f9ca4ce68a0d3db7d`.

Scope:

- replace ad hoc benchmark entry points with one versioned smoke/full workload
  definition and `cudaverse-benchmark/1` report schema;
- require float32/float64 matmul at 256, 1024, and 4096, and dense/sparse
  PCA-kNN at 1,000 x 50, 10,000 x 100, and 50,000 x 128 with `k = 15`;
- require five warmups and ten timed observations in every full case, retaining
  raw times, median, p95, synchronized stage timing, numerical error, complete
  provenance, installed size, and documented peak-memory sources;
- report both host-boundary and resident timing where the public API separates
  them, while explicitly marking dense-PCA upload as inseparable instead of
  inferring a transfer duration;
- preserve the historical phase reports unchanged and avoid universal GPU
  performance claims.

Required evidence before merge:

- [x] the definition checker enforces every requested workload, dtype,
  `k = 15`, and the 5-warmup/10-run full contract;
- [x] package tests verify that the installed benchmark definition contains
  both complete profiles;
- [x] a development smoke run completes for base, native, and torch on the RTX
  2000 and its report passes the machine-readable checker;
- [x] dirty-source smoke evidence is visibly marked and a full dirty-source
  report is rejected;
- [x] exact clean commit
  `cf40e4793ed33fe98bf7967641194aede0431d40` repeats the RTX smoke run and
  retains a checked
  report summary without committing bulky transient output;
- [x] the local CPU suite and source package check pass, including vignette
  rebuild, with `R CMD check --no-manual` reporting `Status: OK`;
- [x] all GitHub cross-platform, pkgdown,
  supply-chain, ABI/PTX, and artifact gates pass.

Deferred beyond CP-05:

- the expensive full 5/10 workload matrix is the next M5 evidence checkpoint,
  not a prerequisite for accepting the runner/contract itself;
- workload interpretation, regression investigation, and the final
  release/defer/reduce-scope decision require the clean full report.

## CP-06: retained full benchmark evidence

Status: in progress on `agent/full-benchmark-evidence` against exact clean
source commit `acd105fe8ef0ed8703e9e16f9ca4ce68a0d3db7d`.

Scope:

- execute every full-profile matmul and dense/sparse PCA-kNN case for base,
  native, and torch on the RTX 2000 development machine;
- retain the complete machine-readable `cudaverse-benchmark/1` report and a
  reproducibly generated human-readable summary tied to the same source
  commit;
- report cold and warm host-boundary timing, resident timing where separable,
  stage timing, peak-memory provenance, installed footprint, and numerical
  validation without making a universal GPU speed claim;
- investigate workload-specific regressions and record whether the evidence
  supports release, deferral, or a reduced benchmark/release scope;
- maintain a candidate completion audit that maps every long-term requirement
  to exact evidence and does not infer final completion from checkpoint tests.

Required evidence before merge:

- [x] full float32/float64 matmul at 256, 1024, and 4096 completes for all
  three backends with numerical parity;
- [ ] dense PCA-kNN at 1,000 x 50, 10,000 x 100, and 50,000 x 128 completes
  for all three backends with projector/reconstruction and exact-index parity;
- [ ] sparse PCA-kNN at the same three dimensions completes for all three
  backends with normalization, projector/reconstruction, and exact-index
  parity;
- [ ] `tools/check-benchmark-report.R` accepts the final complete report;
- [ ] the retained summary is generated from that exact report and records
  the report SHA-256;
- [ ] observed regressions and workload-specific conclusions are reviewed and
  documented;
- [ ] local package checks and all required GitHub checks pass on the evidence
  branch.
