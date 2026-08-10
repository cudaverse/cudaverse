# Native CUDA sparse Phase 3 report

## Outcome

Sparse Phase 3 passed the local RTX 2000 numerical-parity, device-residency, shared-ownership, structured-error, interruption, 1,000-cycle lifecycle, and benchmark gates. The completed path is sparse normalization -> PCA -> exact distance -> stable top-k/kNN.

Native selection remains explicit. Global `auto` preference is unchanged until the remaining element-wise and broadcasting contract is native-complete; this report does not widen the public API or start graph or embedding work.

- Hardware: NVIDIA RTX 2000 Ada Generation, 595.97, 16380, 8.9.
- R: R version 4.6.0 (2026-04-24 ucrt).
- Packages: cudaverse 0.2.0.9000; cudaverseCUDA 0.3.0.9000.
- Source commits: cudaverse `ec55bd524db190af74fb5f0c12c75dfc8227a602`; cudaverseCUDA `61314d12fd0c895c3a5c0c0537cf3e858c57d52c`.
- Installed size: cudaverse 0.3 MiB; cudaverseCUDA 0.5 MiB.

## Sparse normalization, PCA, and exact kNN benchmark

Inputs are deterministic non-negative float64 `dgCMatrix` objects with at least two non-zero values per row and no empty columns. Rows are sum-normalized to 1,000 and transformed with `log1p()`; PCA returns 20 components and exact Euclidean kNN uses k = 15 with batch size 256. Each median/p95 uses five timed runs after two warm-ups. Full timing starts from the host sparse matrix and includes transfer, normalization, PCA, exact distance/top-k, and the final host n-by-k result.

Torch peak memory is its allocator session high-water mark. Native peak memory is the operation-owned cudaverseCUDA high-water mark; neither metric claims ownership of driver context memory.

| Input (requested density) | Backend | Transfer | Normalize | PCA | kNN | Full median | Full p95 | Speedup vs base | Peak backend allocation | Parity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1000x50@0.1 | base | 0.000 s | 0.000 s | 0.000 s | 0.080 s | 0.100 s | 0.120 s | 1.0x | n/a | pass |
| 1000x50@0.1 | native | 0.000 s | 0.000 s | 0.010 s | 0.010 s | 0.030 s | 0.030 s | 3.3x | 9.4 MiB | pass |
| 1000x50@0.1 | torch | 0.000 s | 0.000 s | 0.020 s | 0.050 s | 0.070 s | 0.080 s | 1.4x | 18.7 MiB | pass |
| 5000x100@0.03 | base | 0.000 s | 0.000 s | 0.060 s | 3.350 s | 3.410 s | 3.540 s | 1.0x | n/a | pass |
| 5000x100@0.03 | native | 0.000 s | 0.000 s | 0.040 s | 0.050 s | 0.090 s | 0.100 s | 37.9x | 37.9 MiB | pass |
| 5000x100@0.03 | torch | 0.000 s | 0.010 s | 0.190 s | 1.340 s | 1.560 s | 1.580 s | 2.2x | 48.5 MiB | pass |
| 10000x128@0.01 | base | 0.000 s | 0.000 s | 0.170 s | 13.230 s | 13.400 s | 14.210 s | 1.0x | n/a | pass |
| 10000x128@0.01 | native | 0.000 s | 0.000 s | 0.040 s | 0.160 s | 0.210 s | 0.220 s | 63.8x | 89.8 MiB | pass |
| 10000x128@0.01 | torch | 0.000 s | 0.000 s | 0.280 s | 5.240 s | 5.460 s | 7.160 s | 2.5x | 90.6 MiB | pass |

## RTX 2000 validation

| Gate | Contract | Post-cleanup difference | Result |
|---|---:|---:|---:|
| sparse allocate-normalize-free | 1000 cycles | 0 B | pass |
| shared COO/CSR ownership | release first owner, use second | n/a | pass |
| structured sparse error | backend reusable | n/a | pass |
| injected CUDA error | tracked cleanup | 0 B | pass |
| time-limit interruption | backend reusable | 0 B | pass |
| resident sparse pipeline | stage/1 provenance | n/a | pass |
| exact-distance ties | original row order | n/a | pass |

The lifecycle ceiling is 1 MiB for whole-device used memory and exactly zero bytes for cudaverseCUDA-tracked allocations after synchronization and `gc()`. The interruption and injected-error checks also prove that the same backend instance remains reusable.

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

Canonical-LF report JSON SHA-256: `e0d7f1120c21323d6e94ca7930a797f42ab7730fc254897eb2a2c3a4da67f43a`.

The complete raw timings, numerical errors, memory snapshots, diagnostics, and validation evidence are retained in `phase3-rtx2000.json`.
