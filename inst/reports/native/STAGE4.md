# Native CUDA Phase 4 release-hardening report

## Outcome

Phase 4 passed the local RTX 2000 automatic-selection, float32/float64 tensor-surface parity, device-residency, shared-ownership, structured-error, interruption, dense and sparse 1,000-cycle lifecycle, and Phase 3 benchmark-regression gates. The resident sparse path remains normalization -> PCA -> exact distance -> stable top-k/kNN.

With extension 0.4, `device = "auto"` prefers native only when the versioned backend contract, complete capability set, healthy runtime components (driver, cuBLAS, cuSOLVER, and PTX), and cached self-test all pass. An unhealthy or incomplete native extension remains ineligible; explicit CUDA never silently falls back. This work does not start graph or embedding work.

- Hardware: NVIDIA RTX 2000 Ada Generation, 595.97, 16380, 8.9.
- R: R version 4.6.0 (2026-04-24 ucrt).
- Packages: cudaverse 0.2.0.9000; cudaverseCUDA 0.4.0.9000.
- Source commits: cudaverse `32c47a0d6ddf2ee3dfe6876b259edcf57236f822`; cudaverseCUDA `ea496443f3ecf15a669e0b5e10fb04efa8cdaa2b`.
- Installed size: cudaverse 0.3 MiB; cudaverseCUDA 0.6 MiB.

## Sparse normalization, PCA, and exact kNN benchmark

Inputs are deterministic non-negative float64 `dgCMatrix` objects with at least two non-zero values per row and no empty columns. Rows are sum-normalized to 1,000 and transformed with `log1p()`; PCA returns 20 components and exact Euclidean kNN uses k = 15 with batch size 256. Each median/p95 uses five timed runs after two warm-ups. Full timing starts from the host sparse matrix and includes transfer, normalization, PCA, exact distance/top-k, and the final host n-by-k result.

Torch peak memory is its allocator session high-water mark. Native peak memory is the operation-owned cudaverseCUDA high-water mark; neither metric claims ownership of driver context memory.

| Input (requested density) | Backend | Transfer | Normalize | PCA | kNN | Full median | Full p95 | Speedup vs base | Peak backend allocation | Parity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1000x50@0.1 | base | 0.000 s | 0.000 s | 0.000 s | 0.140 s | 0.140 s | 0.220 s | 1.0x | n/a | pass |
| 1000x50@0.1 | native | 0.000 s | 0.000 s | 0.020 s | 0.000 s | 0.030 s | 0.030 s | 4.7x | 9.4 MiB | pass |
| 1000x50@0.1 | torch | 0.000 s | 0.000 s | 0.040 s | 0.080 s | 0.130 s | 0.130 s | 1.1x | 18.7 MiB | pass |
| 5000x100@0.03 | base | 0.000 s | 0.000 s | 0.090 s | 4.840 s | 4.920 s | 5.070 s | 1.0x | n/a | pass |
| 5000x100@0.03 | native | 0.000 s | 0.000 s | 0.040 s | 0.060 s | 0.110 s | 0.110 s | 44.7x | 37.9 MiB | pass |
| 5000x100@0.03 | torch | 0.020 s | 0.000 s | 0.140 s | 1.800 s | 1.970 s | 2.170 s | 2.5x | 48.5 MiB | pass |
| 10000x128@0.01 | base | 0.000 s | 0.000 s | 0.250 s | 17.690 s | 17.920 s | 18.650 s | 1.0x | n/a | pass |
| 10000x128@0.01 | native | 0.010 s | 0.000 s | 0.050 s | 0.160 s | 0.220 s | 0.230 s | 81.5x | 89.8 MiB | pass |
| 10000x128@0.01 | torch | 0.010 s | 0.010 s | 0.280 s | 6.870 s | 7.170 s | 7.610 s | 2.5x | 90.6 MiB | pass |

## Phase 3 benchmark-regression gate

Native median and p95 timings, operation-owned peak VRAM, and installed size are compared with the checksum-pinned Phase 3 report. Small absolute allowances prevent the Windows timer quantum from turning a 0.01-second measurement into a false regression.

| Input | Phase 3 median | Phase 4 median | Median limit | Phase 3 peak | Phase 4 peak | Result |
|---|---:|---:|---:|---:|---:|---:|
| 1000x50@0.1 | 0.030 s | 0.030 s | 0.050 s | 9.4 MiB | 9.4 MiB | pass |
| 5000x100@0.03 | 0.090 s | 0.110 s | 0.110 s | 37.9 MiB | 37.9 MiB | pass |
| 10000x128@0.01 | 0.210 s | 0.220 s | 0.252 s | 89.8 MiB | 89.8 MiB | pass |

- Baseline: `phase3-rtx2000.json` (`e0d7f1120c21323d6e94ca7930a797f42ab7730fc254897eb2a2c3a4da67f43a`).
- Installed-size gate: cudaverse pass; cudaverseCUDA pass.

## RTX 2000 validation

| Gate | Contract | Post-cleanup difference | Result |
|---|---:|---:|---:|
| automatic native selection | four eligibility gates | n/a | pass |
| float32/float64 tensor surface | parity and native residency | n/a | pass |
| dense allocate-transfer-free | 1000 cycles | 0 B | pass |
| sparse allocate-normalize-free | 1000 cycles | 0 B | pass |
| shared COO/CSR ownership | release first owner, use second | n/a | pass |
| structured sparse error | backend reusable | n/a | pass |
| injected CUDA error | tracked cleanup | 0 B | pass |
| time-limit interruption | backend reusable | 0 B | pass |
| resident sparse pipeline | stage/1 provenance | n/a | pass |
| exact-distance ties | original row order | n/a | pass |

Both lifecycle ceilings are 1 MiB for whole-device used memory and exactly zero bytes for cudaverseCUDA-tracked allocations after synchronization and `gc()`. The interruption and injected-error checks also prove that the same backend instance remains reusable.

## Device residency and provenance

Largest benchmark: `10000x128@0.01`.

- Normalization: `normalization`.
- PCA: `normalization -> sparse_to_dense -> preprocessing -> decomposition -> scores_resident`.
- kNN: `input_materialization -> distance -> neighbor_selection`.
- Every recorded provenance object uses `cudaverse-stage/1`.
- PCA parity compares the numerically identifiable subspace plus full truncated reconstruction, so a zero singular-value tie cannot create a false failure.
- Normalized CSR storage and PCA scores are external pointers with shared ownership; PCA scores feed distance/top-k without a host round trip.
- Stable top-k resolves equal distances by original one-based row index.

## Packaging boundary

- The extension is optional and is not a second user-facing compute API.
- No CUDA Toolkit, LibTorch, Rcpp, or NVIDIA runtime DLL is bundled.
- CUDA 12.8.1 CI builds the package-owned PTX artifact; runtime libraries are discovered dynamically.
- The SBOM and redistributable-license gate remain part of CI.
- This report is release-candidate evidence; it does not publish an artifact, create a tag, or authorize CRAN/Bioconductor submission.

Canonical-LF report JSON SHA-256: `e6ab55930ee941fc39c38c2bc79db161ec22d0bd6a74bbff0e9aae093cae3d42`.

The complete raw timings, numerical errors, memory snapshots, diagnostics, and validation evidence are retained in `phase4-rtx2000.json`.
