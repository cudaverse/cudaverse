# Native CUDA dense Phase 2 report

## Outcome

Dense Phase 2 passed the local RTX 2000 parity, device-residency, structured-error, interruption, lifecycle, and benchmark gates. The benchmark candidate used explicit native selection. Global automatic preference remains compatibility-first because element-wise/broadcast and sparse native paths are outside this milestone. Sparse CUDA was intentionally not started.

- Hardware: NVIDIA RTX 2000 Ada Generation, 595.97, 16380, 8.9.
- R: R version 4.6.0 (2026-04-24 ucrt).
- Packages: cudaverse 0.2.0.9000; cudaverseCUDA 0.2.0.9000.
- Installed size: cudaverse 0.3 MiB; cudaverseCUDA 0.4 MiB.

## PCA and exact kNN benchmark

Inputs are deterministic float64 matrices. PCA returns 50 components, and exact Euclidean kNN uses k = 15 and batch size 256. Each median/p95 uses 10 timed runs after 5 warm-ups; cold runs and every raw timing remain in the JSON report. Full-pipeline timing includes host input, PCA result materialization, exact kNN, and the final host n-by-k result.

| Input | Backend | PCA median | kNN continuation median | Full median | Full p95 | Speedup vs base | Peak backend VRAM | Parity |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1000x50 | base | 0.005 s | 0.110 s | 0.120 s | 0.150 s | 1.0x | n/a | pass |
| 1000x50 | native | 0.010 s | 0.020 s | 0.025 s | 0.030 s | 4.8x | 9.3 MiB | pass |
| 1000x50 | torch | 0.020 s | 0.065 s | 0.090 s | 0.130 s | 1.3x | 19.3 MiB | pass |
| 10000x100 | base | 0.190 s | 13.230 s | 13.415 s | 13.710 s | 1.0x | n/a | pass |
| 10000x100 | native | 0.060 s | 0.200 s | 0.255 s | 0.270 s | 52.6x | 73.9 MiB | pass |
| 10000x100 | torch | 0.290 s | 7.320 s | 7.625 s | 9.140 s | 1.8x | 76.9 MiB | pass |
| 50000x128 | base | 1.910 s | 337.200 s | 339.100 s | 348.390 s | 1.0x | n/a | pass |
| 50000x128 | native | 0.260 s | 3.590 s | 3.855 s | 3.910 s | 88.0x | 441.0 MiB | pass |
| 50000x128 | torch | 0.440 s | 180.635 s | 180.960 s | 187.700 s | 1.9x | 400.7 MiB | pass |

Native peak VRAM is the exact high-water mark of allocations owned by cudaverseCUDA. Torch peak VRAM comes from its CUDA allocator statistics. Driver, library-context, and unrelated process allocations are excluded from those backend peaks and remain visible only in whole-device snapshots.

## Numerical and provenance gates

Every native and torch result matched the CPU reference under the float64 1e-8 relative contract; kNN index matrices were exactly identical. PCA was compared by singular values, reconstruction, and subspace projectors rather than sign-sensitive vectors.

- PCA stages: `preprocessing -> decomposition -> scores_resident`.
- kNN stages: `input_materialization -> distance -> neighbor_selection`.
- Provenance schema: `cudaverse-stage/1`.
- Native distance outputs remain on CUDA; stable top-k returns only the compact CPU result.

## Stability and recovery

| Contract | Work | Whole-device difference after sync/gc | Result |
|---|---:|---:|---:|
| allocate-transfer-free | 1000 cycles | 0 B | pass |
| injected PCA errors | 100 errors | 0 B | pass |
| injected CUDA OOM | 100 errors | 0 B | pass |
| repeated PCA -> kNN | 100 cycles | 0 B | pass |
| time-limit interruption | backend reused | 0 B | pass |

The extension-owned allocation counter returned exactly to baseline in every contract; the whole-device acceptance limit was 1 MiB.

## Packaging and scope

- CUDA 12.8.1 CI reproducibly rebuilds the committed compute_75 PTX.
- Windows, macOS, Ubuntu, and Ubuntu R-devel source checks pass without CUDA hardware.
- Windows/Linux installable artifacts pass the redistribution scan.
- The SBOM and license gate verify the PTX SHA-256 and report cuBLAS/cuSOLVER as runtime-discovered, not bundled.
- No CUDA Toolkit, NVIDIA runtime library, LibTorch, Rcpp, or sparse CUDA implementation is bundled.

Machine-readable report SHA-256:
`22d1e137986072173130873f56ecc0b3a26f6f87a3ed24a1e79405812b63320d`.

The complete raw timings, memory snapshots, validation evidence, software diagnostics, and per-fragment source commits are retained in `phase2-rtx2000.json`.
