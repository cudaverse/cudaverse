# Native CUDA stage 1 report

## Outcome

The first native milestone is complete on the local RTX 2000 Ada development
machine: driver detection, shared allocation/finalization, host-device
transfer, synchronization, and float64 cuBLAS matrix multiplication work
through the unchanged `cudaverse` API. PCA, distance, top-k, and sparse native
kernels were intentionally not started.

Hardware and software:

- NVIDIA RTX 2000 Ada Generation, compute capability 8.9, 16,380 MiB;
- NVIDIA driver 595.97, reported Driver API version 13.2;
- R 4.6.0, `cudaverse` 0.2.0.9000, `cudaverseCUDA` 0.1.0.9000;
- no local CUDA Toolkit used for compilation;
- cuBLAS 12 was loaded from an existing local runtime path for this private
  development test and was not copied into the package or an artifact.

## Matmul results

Each reported median uses 10 timed runs after at least 5 warm-up runs. The
4096 CPU measurements were collected as two five-run batches because a single
process exceeded the execution window; each batch had its own five warm-ups,
and the final median/p95 use all 10 timed values.

| Size | Backend | Resident median | Transfer-included median | p95 included | Max relative error |
|---:|---|---:|---:|---:|---:|
| 256 | base | <0.01 s | <0.01 s | 0.02 s | 0 |
| 256 | native | <0.01 s | <0.01 s | 0.02 s | 3.93e-16 |
| 256 | torch | <0.01 s | <0.01 s | 0.02 s | 3.93e-16 |
| 1024 | base | 0.355 s | 0.355 s | 0.420 s | 0 |
| 1024 | native | 0.030 s | 0.040 s | 0.060 s | 5.50e-16 |
| 1024 | torch | 0.030 s | 0.060 s | 0.090 s | 5.50e-16 |
| 4096 | base | 34.800 s | 34.800 s | 40.170 s | 0 |
| 4096 | native | 0.720 s | 0.840 s | 0.850 s | 9.21e-16 |
| 4096 | torch | 0.720 s | 1.160 s | 1.250 s | 9.21e-16 |

The Windows elapsed-time resolution is too coarse for reliable sub-10 ms
claims at size 256. The complete runs, cold-start values, provenance, and
SHA-verifiable data are in `stage1-rtx2000.json`.

## Lifetime and packaging gates

- 1,000 allocate-transfer-free cycles completed.
- Used VRAM before/after synchronization and `gc()` differed by 0 bytes
  (acceptance limit: 1 MiB).
- A shared external pointer remained valid after its first owner was released
  and freed the allocation exactly once after its last owner was released.
- Native errors are translated to structured `cudaverse_native_error` /
  `cudaverse_backend_error` conditions.
- Windows source installation and `R CMD check --no-manual` completed with
  `Status: OK`.
- The installed extension measured 218,449 bytes, excluding external CUDA
  runtime libraries.
- The redistribution gate found no NVIDIA or LibTorch binary in the package.

Report SHA-256:
`54FC5641E4C21EFC18F4FAAC229EA970868021EC302F5A83C20F26F4768B62F6`.

Windows/Linux CI artifacts and the CUDA 12.8.1 ABI compile are the remaining
remote checks for this milestone. No binary is promoted to a public release.
