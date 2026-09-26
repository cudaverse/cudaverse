# Internal CRAN release checklist for cudaverse 0.4.1

This is the first CRAN submission of `cudaverse`.

## Publication checkpoint (2026-09-25)

CRAN published version 0.4.1 on 2026-09-10. Its package record and listed
checks are available at <https://cran.r-project.org/package=cudaverse> and
<https://cran.r-project.org/web/checks/check_results_cudaverse.html>. The CRAN
source contains the submitted package files; CRAN added its `MD5` manifest and
rewrote `DESCRIPTION` with publication metadata.

The existing `v0.4.1` GitHub tag and release point to `e292672`, before the
accepted resubmission commit `2faaa38`. The release description now identifies
the accepted CRAN source and the earlier tag and attachment without moving the
tag. The checklist below records the submission process; its
unchecked historical steps are not a claim that CRAN review is still pending.

## Candidate

- [x] Confirm that `cudaverse` conflicts with neither current nor archived CRAN
      packages nor current Bioconductor packages.
- [x] Confirm that `DESCRIPTION`, `NEWS.md`, documentation, examples, and
      vignettes describe version 0.4.1 exactly.
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

- [x] Verify the CRAN package and check-results pages.
- [x] Reconcile the existing `v0.4.1` tag and release with the accepted source
      without moving the published tag.
- [ ] Update installation documentation from development installation to CRAN.
- [ ] Begin the next package submission only after this package is accepted.
