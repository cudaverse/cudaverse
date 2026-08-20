# Internal CRAN release checklist for cudaverse 0.4.0

This is the first CRAN submission of `cudaverse`.

## Candidate

- [x] Confirm that `cudaverse` conflicts with neither current nor archived CRAN
      packages nor current Bioconductor packages.
- [x] Confirm that `DESCRIPTION`, `NEWS.md`, documentation, examples, and
      vignettes describe version 0.4.0 exactly.
- [ ] Build a local preflight source tarball and record its SHA-256.
- [ ] Run local `R CMD check --as-cran` on the preflight source tarball:
      0 errors, 0 warnings, and the expected new-submission note.
- [ ] Run the GitHub R CMD check matrix on Windows, macOS, Ubuntu release, and
      Ubuntu R-devel.
- [ ] Run the manually dispatched `cran-readiness` workflow and retain the exact
      source candidate, full R-devel check log, and reference manual.
- [ ] Review spelling and URL checks.
- [ ] Submit the exact verified source tarball without rebuilding it.

## Submission

- [ ] Upload the verified tarball through the CRAN submission form.
- [ ] Accept the confirmation email sent to the `DESCRIPTION` maintainer.
- [ ] Do not submit another build while the candidate is pending.

## Acceptance

- [ ] Verify the CRAN package and check-results pages.
- [ ] Tag the accepted commit as `v0.1.0` and create the matching GitHub release.
- [ ] Update installation documentation from development installation to CRAN.
- [ ] Begin the next package submission only after this package is accepted.
