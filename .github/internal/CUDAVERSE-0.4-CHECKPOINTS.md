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

Status: complete at implementation commit
`01d751663ceb9a745486474b9beffacf2c16cca8`; the checkpoint-record-only
follow-up must pass the same PR gates before merge.

Review:

- draft PR: <https://github.com/cudaverse/cudaverse/pull/27>;
- target: `develop/0.4` (not `main`);
- no tag, release, CRAN submission, or Pages deployment; and
- GitHub Pages deploy job skipped as required.

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

Remote evidence at the implementation commit:

- Windows, macOS, Ubuntu, and R-devel package checks pass;
- CPU integration and pkgdown pass;
- supply-chain and repository workflow-boundary checks pass;
- Linux and Windows artifact builds pass;
- CUDA 12.8.1 ABI and PTX checks pass; and
- the GitHub hardware job skips without the protected runner, while the
  explicitly enabled local RTX 2000 parity and lifecycle suites pass with no
  hardware skips.

## CP-02A: allocation-free native reshape views

Status: complete at implementation commit
`ab971eecead4a7b04b535fe5d4d569300c9ad0a8`; the checkpoint-record-only
follow-up must pass the same PR gates before merge.

Review:

- draft PR: <https://github.com/cudaverse/cudaverse/pull/28>;
- target: `develop/0.4` (not `main`);
- no tag, release, CRAN submission, or Pages deployment; and
- GitHub Pages deploy job skipped as required.

Scope:

- separate dense view descriptors from shared device-allocation ownership;
- make native reshape metadata-only without a device-to-device copy;
- keep downstream kernels aware of each view's logical shape; and
- guarantee source, alias, and nested-view release order cannot double-free or
  invalidate a live view.

Required gate:

- existing CPU, torch, and native tensor behavior remains unchanged;
- creating and releasing 1,000 native reshape views allocates zero additional
  tracked VRAM;
- nested views remain readable after their source descriptors are released;
- matmul consumes the reshaped logical dimensions; and
- the final shared owner releases the allocation exactly once.

Unresolved scope:

- transpose still materializes a contiguous device output; general strided
  views require a separate contract because current kernels assume contiguous
  R column-major storage.

Local evidence:

- the complete ordinary package suite passes, with hardware-only cases skipped
  only when the explicit native test switch is absent;
- the complete explicitly enabled RTX 2000 suite passes with no hardware skips;
- nested reshape views allocate zero additional tracked VRAM, survive source
  release, drive matmul with their logical shape, and return to the exact
  tracked-memory baseline after 1,000 create/release cycles;
- the source package builds and Windows `R CMD check --as-cran` reports
  0 errors, 0 warnings, and only the expected development-version NOTE;
- capability, redistribution, SBOM, release-boundary, benchmark, candidate,
  report, and rejection self-tests pass; and
- pkgdown and the public documentation boundary pass from a temporary source
  copy outside the Dropbox synchronization lock, without deploying Pages.

Remote evidence at the implementation commit:

- Windows, macOS, Ubuntu, and R-devel package checks pass;
- CPU integration and pkgdown pass;
- supply-chain and repository workflow-boundary checks pass;
- Linux and Windows no-CUDA artifact builds pass; and
- CUDA 12.8.1 ABI and PTX checks pass.

## CP-02B: device-resident replacement dtype conversion

Status: implementation and local gates complete; remote PR checks pending.

Scope:

- cast replacement tensors on the same CUDA backend/device to the target
  floating dtype without materializing either tensor on the host;
- retain exact host validation for integer targets;
- record the conversion and scatter as separate provenance stages; and
- prove cast/scatter temporaries release cleanly across 1,000 cycles.

Required gate:

- base adapter contract detects any accidental host transfer;
- native integer-to-float64 and float64-to-float32 replacement parity passes;
- output dtype, device, backend, and provenance remain explicit; and
- repeated native cast/scatter returns to the exact tracked-memory baseline.

Local evidence:

- a synthetic CUDA adapter whose `to_host` always errors completes the
  same-backend replacement contract through one backend cast and scatter;
- native integer-to-float64 and float64-to-float32 replacement parity passes on
  the RTX 2000 with separate `replacement_cast` and `replacement` provenance;
- 1,000 native cast/scatter cycles return to the exact tracked-memory baseline;
- the complete ordinary and explicitly enabled RTX 2000 suites pass, with no
  hardware skips in the latter;
- source tarball SHA-256 is
  `72C65EEEBD28B0AE82D354F863BC9FF5028BD4E6745ED90F729DFE23F87FD656`;
- Windows `R CMD check --as-cran` reports 0 errors, 0 warnings, and only the
  expected development-version NOTE; and
- capability, redistribution, SBOM, release-boundary, benchmark, candidate,
  rejection, pkgdown, and public-documentation gates pass without deployment.
