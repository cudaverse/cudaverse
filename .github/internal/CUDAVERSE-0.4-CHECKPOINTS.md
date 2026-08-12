# cudaverse 0.4 checkpoint log

## Baseline

- Branch point: `b0ab90f8547d42a5244963bc6b15d0ba223382ff`.
- Long-lived line: `develop/0.4`.
- Frozen release refs: `main` and `release/cran-0.1.0` at
  `59e15c8c5a56d26e09a594886c875b1b8249f6f9`.
- Exact 0.3 evidence archive:
  `cudaverse/evidence/0.3.0.9000-b0ab90f`, manifest SHA-256
  `858D2691D276CD26EA225B4B57BA2219644466181869977C1AC87400D9FCDFF1`.
- Baseline surface: 38 exports, 28 S3 registrations, 179 test blocks,
  163 tracked files, and 1.48 MiB tracked payload.
- Open GitHub issues at baseline: 0.

## CP-01: diagnostics and development-line isolation

Status: implementation and local gates complete; remote PR checks pending.

Scope:

- advance development metadata to `0.4.0.9000`;
- route development installation and branch CI to `develop/0.4`;
- retain the verified compact 0.3 evidence outside the source tree;
- add `status`, `summary`, `next_steps`, and `backend_status` to
  `cuda_diagnostics()` without removing compatibility fields;
- attach the same stable reason and actionable next steps to strict explicit
  CUDA errors; and
- align README, NEWS, reference documentation, vignette setup, and the
  capability matrix.

Required gate:

- diagnostics contract, fallback, and strict-error tests pass;
- complete CPU package tests and source check pass;
- native diagnostics and public conformance pass on the RTX 2000;
- workflow boundary, capability-matrix, SBOM, and public-doc checks pass;
- source remains lightweight and no runtime binary is introduced.

Unresolved risk:

- Wording must remain actionable across Windows, Linux, macOS, native, torch,
  and CPU-only hosts without promising that installation alone fixes an
  incompatible driver or runtime.

Local evidence:

- targeted diagnostics contract tests pass;
- the complete ordinary suite passes; its 26 hardware-only cases skip only
  when `CUDAVERSE_NATIVE_TESTS` is not enabled;
- the complete explicit RTX 2000 suite passes with native enabled and no
  hardware skips, including dense and sparse 1,000-cycle lifecycle cases;
- native diagnostics select `native`, report `cuda_ready`, expose one selected
  backend row, and retain an empty action list on the healthy runtime;
- capability matrix, redistribution, SBOM/PTX/license, external-release
  boundary, benchmark-definition, historical evidence, benchmark rejection,
  candidate-composition, and candidate-evidence self-tests pass;
- pkgdown builds the 0.4 roadmap, diagnostics reference, and tutorials, and the
  public documentation boundary passes without deploying Pages; and
- source tarball `cudaverse_0.4.0.9000.tar.gz` has SHA-256
  `8592985A7CD975FBD2A8EF47E5DE9C64343B7BF2D659F9576C711105D697899A`.
  Windows `R CMD check --as-cran` has 0 errors, 0 warnings, and only the
  expected development-version incoming NOTE for `0.4.0.9000`.

Remote evidence required before merge:

- Windows, macOS, Ubuntu, and R-devel package checks;
- CPU integration, pkgdown, supply-chain, Linux/Windows artifact, CUDA 12.8.1
  ABI/PTX, and repository workflow-boundary checks; and
- RTX parity when the protected runner is enabled or explicitly dispatched.
