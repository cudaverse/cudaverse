## Update to an existing CRAN package

This updates cudaverse 0.4.1, published on 2026-09-10, to 0.4.2.

- Correct native PCA scores for duplicate observations and avoid cancellation
  in native Euclidean distances for close pairs.
- Add an opt-in TF32 matrix-multiplication policy for supported NVIDIA GPUs.
  Standard float32 remains the default; unsupported requests fail explicitly.
- Improve benchmark validation and Windows CUDA setup documentation.

The maintainer, license, required R dependencies and optional runtime-library
model are unchanged. CUDA is optional, and NVIDIA runtimes are neither bundled
nor downloaded. Examples, tests and vignettes work without CUDA; hardware-only
tests are conditional. Optional torch use remains guarded.

## Validation and source provenance

The release is prepared from main revision f93361d, whose R checks passed on
Windows, macOS, Linux release and Linux R-devel. Each recorded 1,242 passing
assertions, 58 conditional skips, zero failures and zero test warnings.

Its unchanged R/native runtime also passed the retained full shared-A100 core,
precision and single-cell workloads. Shared-device timings are not presented
as uncontended performance claims. The documented numerical policies and all
original strict diagnostics are retained with the evidence.

The release workflow builds a source archive with R CMD build, records its
source revision and SHA-256, and checks that exact archive with R-devel
--as-cran, including vignettes and the PDF/HTML manual. The verified archive
and fresh release-check results accompany the submission; historical checks
are not substituted for the release archive's checks.

## Downstream dependencies

The CRAN package page lists no reverse dependencies as of 2026-10-16.
The separately maintained cudacellr 0.4.0 requires cudaverse >= 0.4.1 and has
passed the identifier/provenance integration contract with the release's
runtime. Its submission will be confirmed only after this cudaverse update
is accepted.
