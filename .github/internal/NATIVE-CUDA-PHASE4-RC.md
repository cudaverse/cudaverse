# Internal cudaverse 0.2 native CUDA release-candidate assessment

> Historical Phase 4 assessment: the implementation described here was later
> consolidated into the main `cudaverse` package. The pinned results remain
> the pre-consolidation baseline for the integrated-backend validation.

## Decision

**GO for integration as a cudaverse 0.2 release candidate; HOLD for any tag,
public binary release, or CRAN/Bioconductor submission.**

The implementation and evidence satisfy the bounded Phase 4 engineering gate.
Publishing remains a separate maintainer decision and is outside this
assessment.

## Candidate boundary

- `cudaverse` remains the only user-facing compute API.
- The native implementation is built into `cudaverse` and loads NVIDIA
  libraries lazily.
- `device = "auto"` prefers native only when the backend contract,
  capabilities, driver/cuBLAS/cuSOLVER/PTX runtime, and cached self-test pass.
- Explicit `device = "cuda"` never silently falls back.
- torch remains an optional compatibility backend for the 0.2 cycle.
- CPU installation and checks do not require CUDA.
- Windows and Linux provide the native backend; macOS installs a clear
  unsupported-CUDA stub.
- Dense tensor arithmetic, broadcasting, reshape, transpose, reductions, and
  matmul cover float32 and float64. Native decomposition and the R `Matrix`
  sparse workflow use float64; float32 tensor inputs are safely promoted for
  decomposition rather than being represented as float32 sparse matrices.
- Graph, community-detection, embeddings, and cudacellr work are not part of
  this candidate.

## Evidence

| Gate | Evidence | Result |
|---|---|---:|
| CPU/torch compatibility | full local `cudaverse` test suite | pass |
| Native automatic selection | contract + capabilities + runtime + cached self-test | pass |
| RTX tensor parity | float32/float64 arithmetic, broadcast, reshape, transpose, reduction, matmul | pass |
| Resident sparse workflow | normalization -> PCA -> distance -> stable top-k/kNN | pass |
| Numerical contract | CPU/native/torch parity, stable ties, PCA subspace/reconstruction | pass |
| Lifetime | dense and sparse 1,000-cycle checks; 0 tracked bytes after cleanup | pass |
| Recovery | structured R/CUDA errors and time-limit interruption; backend reusable | pass |
| Regression | median, p95, peak VRAM, and installed size against pinned Phase 3 report | pass |
| Cross-platform | Windows, macOS, Ubuntu, and R-devel package checks | pass |
| CUDA build | CUDA 12.8.1 PTX reproducibility and Linux native ABI compile | pass |
| Artifacts | Windows and Linux build, redistribution scan, clean no-CUDA install/load | pass |
| Supply chain | CycloneDX SBOM and third-party license inventory | pass |
| Local source package | `R CMD check --as-cran --no-manual` | 0 errors / 0 warnings / 1 development-metadata note |

The canonical evidence is the historical
[`cudaverseCUDA` Phase 4 report](../inst/reports/native/STAGE4.md)
and its
[machine-readable JSON](../inst/reports/native/phase4-rtx2000.json).
The report records an RTX 2000 Ada Generation, R 4.6.0, cudaverse 0.2.0.9000,
and cudaverseCUDA 0.4.0.9000. Its Phase 3 baseline SHA-256 is
`e0d7f1120c21323d6e94ca7930a797f42ab7730fc254897eb2a2c3a4da67f43a`.

## Measured outcome

For the largest `10000 x 128` sparse case, the complete host-to-result native
pipeline median was 0.22 seconds with 89.8 MiB operation-owned peak VRAM. The
same run measured 17.92 seconds for base R and 7.17 seconds for torch. These are
machine- and contract-specific measurements, not universal speed claims.

The standard installed sizes were 349,398 bytes for cudaverse and 670,917 bytes
for cudaverseCUDA. No LibTorch, CUDA Toolkit, or NVIDIA runtime binary is
bundled; the extension contains only package code and checksum-pinned,
package-owned PTX.

## Release holds

Do not create a 0.2 tag, publish CI artifacts as releases, change the frozen
0.1 CRAN branch, or submit a package to CRAN/Bioconductor without separate
maintainer authorization. Before any later public binary release, rerun the
SBOM/license gate on the exact artifact and retain its checksum with the source
commit and hardware evidence.

Before a future CRAN submission, replace the development version and rerun
CRAN incoming checks on the exact source tarball. The current local NOTE is
expected for a development RC and is not being presented as a zero-NOTE CRAN
candidate.
