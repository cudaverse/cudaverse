# cudaverse 0.3 internal development roadmap

## Outcome

The 0.3 development line ends with one review-ready `cudaverse` source
candidate whose optional native CUDA core is lightweight, reproducible, and
useful for device-resident dense and sparse numerical workflows. A merged PR
is a checkpoint, not a release decision.

`main` and `release/cran-0.1.0` stay frozen at
`59e15c8c5a56d26e09a594886c875b1b8249f6f9`. Development targets
`develop/native-cuda`; tagging, releasing, deploying the main pkgdown site, or
submitting to CRAN or Bioconductor requires explicit maintainer approval.

## Audited baseline

The baseline at `1b1707366f06a9667063f1703f6470117d517320` has 38 exports,
27 registered S3 methods, and 160 test blocks. The export-level base, torch,
and native contract is checked against
`inst/reports/backend-capability-matrix.csv`. GitHub had no open issues or pull
requests at the start of this roadmap. Required pull-request checks covered
Windows, macOS, Ubuntu, R-devel, CPU contracts, pkgdown, supply-chain policy,
CUDA 12.8.1 ABI/PTX reproducibility, and Windows/Linux package artifacts.

The tracked source payload was about 1.1 MB: the committed PTX was about
166 KB and retained reports about 253 KB. No NVIDIA runtime, CUDA Toolkit, or
LibTorch binary was tracked. Historical hardware reports remain immutable
evidence; new 0.3 evidence must identify one exact candidate commit and
artifact.

## Ordered milestones and gates

### M1: native dense object completion

- Device-native subsetting and replacement for ordinary R indices, all three
  tensor dtypes, duplicate indices, recycling, drop behavior, and dimnames.
- Shared ownership remains single-free across reshape and replacement inputs.
- Printing, synchronization, injected error, interruption, and 1,000-cycle
  lifecycle tests remain within the 1 MiB post-cleanup VRAM ceiling.
- Unsupported edge semantics use an explicit, provenance-visible compatibility
  path rather than a silent transfer.

Gate: CPU suite and cross-platform checks pass; CUDA 12.8.1 PTX is exactly
reproducible; RTX 2000 parity and lifecycle evidence is attached to the exact
checkpoint commit.

### M2: resident dense algorithms

- PCA/SVD prediction, distance, stable top-k/kNN, and justified k-means stages
  use registered operations rather than backend-name branches.
- `PCA -> distance -> top-k` retains intermediate device storage and reports
  every host/device boundary through `cudaverse-stage/1`.
- PCA validation compares projectors and reconstruction; kNN ties use original
  row order.

Gate: integer/index results are exact; float32 uses `rtol = 1e-5` and
`atol = 1e-6`; heavy float64 operations use `rtol = 1e-8`.

### M3: sparse numerical completion

- COO/CSR conversion and transpose, reductions, normalization, matvec,
  sparse-dense multiplication, PCA preparation, and kNN preparation preserve
  Matrix dimensions and labels.
- Host work and transfers are measured and reduced. Float32 sparse storage is
  added only after a benchmark and accuracy note demonstrates value.

Gate: dense-reference parity, structured error recovery, shared ownership,
interruption, and 1,000 sparse lifecycle cycles pass on the RTX 2000.

### M4: unified conformance and failure recovery

- One parameterized contract covers base, torch, and native behavior for every
  export and intentional CPU/hybrid boundary.
- Invalid shapes/dtypes, non-finite values, absent runtimes, missing symbols,
  OOM, injected CUDA errors, interruption, repeated release, views, diagnostics,
  and backend reuse have explicit assertions.
- Tests are never weakened to admit a backend discrepancy.

Gate: Windows/macOS/Ubuntu R-release checks, Ubuntu R-devel, CPU/torch contract
tests, no-CUDA artifacts, and native hardware tests all pass.

### M5: benchmark and footprint evidence

- Matmul sizes: 256, 1024, and 4096 square matrices.
- Dense and sparse PCA/kNN cases include representative small, medium, and large
  workloads with `k = 15`.
- Report cold time, warm median/p95 after five warm-ups and ten runs, included
  and excluded transfer time, peak VRAM, installed size, numerical error, and
  complete backend provenance for CPU, torch, and native.

Gate: investigated regressions and workload-specific conclusions; no universal
GPU speed claim.

### M6: packaging, documentation, and final candidate

- CUDA 12.8.1 ABI/PTX, SBOM, third-party license, redistribution, and package
  size checks remain reproducible and green without bundling runtime libraries.
- README, NEWS, reference pages, tutorials, diagnostics, troubleshooting, and
  pkgdown agree on installation, runtime discovery, residency, limitations,
  fallback behavior, and measured performance.
- Build one exact source tarball and retain its SHA-256 plus matching R checks,
  artifacts, RTX report, benchmarks, SBOM, licenses, and pkgdown evidence.

Gate: write a release/defer/reduce-scope decision report, then stop before any
external release action for maintainer approval.

## Checkpoint policy

Each short feature branch must update
`.github/internal/CUDAVERSE-0.3-CHECKPOINTS.md`
with its scope, evidence, and unresolved risks. A checkpoint can merge only
after its meaningful required checks pass. Evidence-deferred work must name the
missing proof and cannot be presented as completed.
