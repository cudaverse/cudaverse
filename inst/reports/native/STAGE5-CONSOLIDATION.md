# Native CUDA consolidation report

## Outcome

The lightweight native CUDA implementation now builds and runs inside the
`cudaverse` package. The public API, `cudaverse-backend/1` contract, structured
conditions, and `cudaverse-stage/1` provenance remain unchanged. The package
does not require or load `cudaverseCUDA`, and NVIDIA libraries remain
runtime-discovered rather than bundled.

The machine-readable RTX report passed every benchmark, numerical, lifetime,
error-recovery, interruption, provenance, automatic-selection, and installed-
size gate.

## Candidate identity

- consolidation source commit: `61f5d4e7353ccb82c69ffd1a5bc0ff5e4834b061`;
- imported extension source commit:
  `38266cd254947235615baf71d8a1de433ceb02d5`;
- hardware report:
  `inst/reports/native/consolidation-rtx2000.json`;
- report SHA-256:
  `13ed0e28c1337dbfd934c5b6dbb1a81a26f2b58e73b04194c63c2c964f5f2888`;
- NVIDIA RTX 2000 Ada Generation, 16,380 MiB, driver 595.97;
- R 4.6.0 and `cudaverse` 0.2.0.9000.

The report recorded a clean source tree at the candidate commit. For this
private hardware run, cuBLAS and cuSOLVER were supplied through explicit local
paths; neither library was copied into the source package or artifacts.

## Validation

| Gate | Result |
|---|---:|
| Isolated install with `cudaverseCUDA` absent | pass |
| Windows R 4.6 `R CMD check --as-cran --no-manual` | 0 errors, 0 warnings, 1 development metadata note |
| CPU/torch combined contract suite | pass |
| Native float32/float64 tensor and algorithm parity | pass |
| Sparse normalization -> PCA -> distance -> stable top-k | pass, device-resident |
| Dense allocate/transfer/free lifecycle | 1,000 cycles; 0 tracked and whole-device byte difference |
| Sparse allocate/normalize/free lifecycle | 1,000 cycles; 0 tracked and whole-device byte difference |
| Shared ownership, injected CUDA error, R interruption | pass; backend reusable |
| SBOM, third-party inventory, PTX checksum, redistribution scan | pass |

## End-to-end benchmark

Times are median seconds for the complete host-input-to-result boundary after
two warm-ups and five timed runs. They are measurements of this machine and
contract, not universal speed claims.

| Case | Base R | Native CUDA | torch CUDA | Native p95 | Native peak VRAM |
|---|---:|---:|---:|---:|---:|
| `1,000 x 50 @ 0.10` | 0.12 | 0.03 | 0.08 | 0.05 | 9.4 MiB |
| `5,000 x 100 @ 0.03` | 4.23 | 0.11 | 1.78 | 0.11 | 37.9 MiB |
| `10,000 x 128 @ 0.01` | 17.34 | 0.21 | 6.80 | 0.25 | 89.8 MiB |

All native medians, p95 values, and peak allocations passed the checksum-pinned
Phase 3 regression limits. The integrated installed-size measurement was
994,113 bytes. The prior packages measured 897,619 bytes together and the
consolidation ceiling was 1,122,024 bytes. The checked source archive was
164,578 bytes; no CUDA Toolkit, NVIDIA runtime DLL, or LibTorch payload is
included.

## Repository transition gate

The engineering replacement is ready for cross-platform and GitHub review.
`cudaverseCUDA` must remain available until the reviewed branch and required
CI jobs are green. After that, its README will redirect to `cudaverse`, the
repository will be archived rather than deleted, and its local checkout will
move under `_archived`.
