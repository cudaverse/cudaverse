## New submission

This is the first CRAN submission of `cudaverse`.

CUDA is optional. The lightweight native backend discovers user-provided NVIDIA
runtime libraries only when requested and does not bundle or download them.
`torch` is listed only in `Suggests`, all torch-dependent code is conditional,
and the documented automatic path uses the portable R backend when no usable
CUDA backend is available. An explicit `device = "cuda"` request fails with an
actionable error instead of silently using the CPU.

## Test environments

- Windows 11, R 4.6.0 (local, NVIDIA RTX 2000 Ada)
- Windows, current R release (GitHub Actions)
- macOS, current R release (GitHub Actions)
- Ubuntu, current R release (GitHub Actions)
- Ubuntu, R-devel (GitHub Actions and the full `cran-readiness` candidate check)

## R CMD check results

0 errors | 0 warnings | 1 note

The note is the expected incoming-feasibility note for a new submission:

- `New submission`

The local Windows check used the source tarball produced by `R CMD build` and
`R CMD check --as-cran`, with all suggested packages installed. Its PDF and
HTML reference manuals, examples, tests, and vignettes passed. Because the
machine has a usable NVIDIA GPU, conditional CUDA tests also ran rather than
being skipped.

The manually dispatched `cran-readiness` workflow builds one source candidate,
records its SHA-256, passes that exact tarball to a separate R-devel job, runs a
full CRAN-style check including the reference manual, and retains the candidate
and check evidence together. The tarball uploaded to CRAN will be that verified
workflow artifact, not a locally rebuilt archive.

## Downstream dependencies

There are no CRAN reverse dependencies because this is a new submission. The
development package `cudacellr` depends on `cudaverse`, but it will not be
submitted until `cudaverse` has completed CRAN review.
