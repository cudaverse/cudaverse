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

## CP-03A: explicit batched dense distance

Status: complete at implementation commit
`76325b9b783e39ef3523b9ddfd020cf3039135b6`; the checkpoint-record-only
follow-up must pass the same PR gates before merge.

Review:

- draft PR: <https://github.com/cudaverse/cudaverse/pull/32>;
- target: `develop/0.4` (not `main`);
- no tag, release, CRAN submission, or Pages deployment; and
- GitHub Pages deploy job skipped as required.

Scope:

- add one explicit `batch_size` contract to `cuda_distance()` and record the
  effective batch size and count in existing provenance parameters;
- require base, torch, and native built-in backends to honor query batching;
- keep native query/reference storage and cached reference norms resident
  across blocks, transferring only each completed distance block to R; and
- preserve the full dense host-output contract, strict CUDA semantics,
  Euclidean/cosine accuracy, dimnames, and existing provenance schema.

Required gate:

- self and cross-distance parity passes for Euclidean and cosine metrics at
  batch sizes 1, intermediate, and larger than the query;
- public validation rejects non-positive, fractional, non-finite, missing, or
  empty batch sizes before backend dispatch;
- RTX 2000 operation-owned peak VRAM scales with `batch_size * nrow(y)` rather
  than `nrow(x) * nrow(y)`;
- 1,000 repeated native batched calls return to the exact tracked-memory
  baseline; and
- interruption releases native block state and leaves the backend reusable.

Initial RTX 2000 evidence:

- for a `512 x 16` self-distance input with batch size 64, tracked peak delta
  falls from 4,268,032 bytes for the full device matrix path to 602,624 bytes
  for the batched path, an 85.9% reduction; and
- final tracked VRAM delta is zero.

Local evidence:

- base and native self/cross distance parity passes for Euclidean and cosine
  metrics at batch sizes 1, intermediate, and larger than the query;
- the backend operation contract receives one explicit effective batch size,
  while legacy third-party distance operations retain a compatibility path;
- invalid batch sizes fail before backend dispatch, and effective batch size
  plus batch count are recorded in provenance parameters;
- the native runtime self-test includes resident batched distance;
- 1,000 repeated native batched distance calls return to the exact
  tracked-memory baseline, and an interrupted run leaves the backend reusable;
- the complete ordinary and explicitly enabled RTX 2000 suites pass, with no
  hardware skips in the latter;
- source tarball SHA-256 is
  `CB7287AFD7BB9ECDDC24413C05200C58DC1EB81784AA4CB215F1903597982427`;
- Windows `R CMD check --as-cran` reports 0 errors, 0 warnings, and only the
  expected development-version NOTE; and
- capability, redistribution, SBOM, release-boundary, benchmark, candidate,
  rejection, pkgdown, and public-documentation gates pass without deployment.

Environment boundary:

- the installed torch package does not report a usable CUDA backend on this R
  runtime, so torch hardware parity remains assigned to the protected runner;
  its built-in adapter implements the same explicit block loop and is covered
  by no-CUDA loading, contract, and cross-platform checks.

Remote evidence at the implementation commit:

- Windows, macOS, Ubuntu, and R-devel package checks pass;
- CPU integration and pkgdown pass;
- supply-chain and repository workflow-boundary checks pass;
- Linux and Windows no-CUDA artifact builds pass; and
- CUDA 12.8.1 ABI and unchanged PTX checks pass.

## CP-03B: bounded-memory resident k-means

Status: complete at implementation commit
`86e1a6d3a8c79c1ae305fa7690501280dc9dc4ad`; the checkpoint-record-only
follow-up must pass the same PR gates before merge.

Review:

- draft PR: <https://github.com/cudaverse/cudaverse/pull/33>;
- target: `develop/0.4` (not `main`);
- no tag, release, CRAN submission, or Pages deployment; and
- GitHub Pages deploy job skipped as required.

Scope:

- add explicit, provenance-recorded `batch_size` control to `cuda_kmeans()`;
- prefer a new `algorithm_kmeans_batched` backend operation while retaining
  the four-argument `algorithm_kmeans` compatibility contract;
- keep native observations, centres, assignments, accumulation, and Lloyd
  updates resident while materializing only one observation-by-centre block;
- cache centre norms once per assignment pass and interrupt safely between
  blocks; and
- pass the same explicit batch size through the base/torch compatibility
  distance path.

Required gate:

- CPU and native results remain invariant across batch sizes 1, intermediate,
  and larger than the observation count;
- lowest-centre-index tie handling, empty-centre behavior, large-offset
  accuracy, names, iteration count, and convergence remain unchanged;
- the new operation is preferred when present and legacy adapters receive no
  extra argument;
- native peak temporary VRAM is lower than the full-batch path; and
- 1,000 repeated batched native runs return to the exact tracked-memory
  baseline with structured error recovery intact.

Local evidence:

- the complete ordinary suite and the complete explicitly enabled RTX 2000
  suite pass;
- CPU and native k-means parity passes across batch sizes and existing edge
  cases, with the same assignments, centres, within-cluster sums, iteration
  count, convergence, names, backend, and stage provenance;
- for `2048 x 16` data and 64 centres, batch size 64 reduces tracked peak delta
  from 2,433,792 bytes to 394,496 bytes, an 83.8% reduction;
- final tracked VRAM delta is zero, and 1,000 repeated small batched fits return
  to the exact tracked-memory baseline;
- the exact full-vignette source tarball SHA-256 is
  `F1D9B9CDD3F0363AED565B1469D9F1563C287BCF4C4E1F28FD74F85BD328E1BD`;
- Windows `R CMD check --as-cran --no-manual` reports 0 errors, 0 warnings,
  and only the expected development-version incoming NOTE; and
- capability, redistribution, SBOM, license/PTX, release-boundary, benchmark,
  candidate, rejection, pkgdown, and public-documentation gates pass without
  deployment.

Remote evidence at the implementation commit:

- Windows, macOS, Ubuntu, and R-devel package checks pass;
- CPU integration and pkgdown pass;
- supply-chain and repository workflow-boundary checks pass;
- Linux and Windows no-CUDA artifact builds pass; and
- CUDA 12.8.1 ABI and unchanged PTX checks pass.

## CP-03C: device-resident PCA prediction scores

Status: complete at implementation commit
`62a68e77b8279eb30efaee0601e3b97211adbdb2`; the checkpoint-record-only
follow-up must pass the same PR gates before merge.

Review:

- draft PR: <https://github.com/cudaverse/cudaverse/pull/34>;
- target: `develop/0.4` (not `main`);
- no tag, release, CRAN submission, or Pages deployment; and
- GitHub Pages deploy job skipped as required.

Scope:

- retain native PCA prediction scores in shared device storage after returning
  the same compatible R matrix;
- mark the resident output in the existing `cudaverse-stage/1` provenance;
- let following native distance and kNN stages reuse the storage without an
  upload; and
- preserve CPU/torch behavior, public classes, dimnames, numerical results,
  and model-device selection.

Required gate:

- native prediction matches the CPU matrix product within the float64 heavy
  numerical tolerance;
- kNN preparation shares the prediction allocation and adds only one row-norm
  vector rather than another score allocation;
- native prediction-to-kNN matches CPU and records `device_resident_input`;
- 1,000 resident prediction cycles return to the exact tracked-memory
  baseline; and
- ordinary, RTX, no-CUDA, cross-platform, source, and documentation gates stay
  green.

Local evidence:

- the complete ordinary suite and the complete explicitly enabled RTX 2000
  suite pass;
- prediction values match the centered/scaled CPU matrix product and retain
  the same R matrix, dimnames, public device attribute, and backend result;
- native kNN preparation over resident predictions increases tracked memory by
  exactly `nrow(scores) * 8` bytes for cached row norms, proving there is no
  duplicate score upload, and releases exactly to its prior baseline;
- native prediction-to-kNN matches CPU and starts provenance with backend
  `native` and reason `device_resident_input`;
- 1,000 resident prediction cycles return to the exact tracked-memory
  baseline;
- the exact full-vignette source tarball SHA-256 is
  `26CE3D1019569EC355974D248166422313B362EA2D70D5CC367144AA193D3216`;
- Windows `R CMD check --as-cran --no-manual` reports 0 errors, 0 warnings,
  and only the expected development-version incoming NOTE; and
- capability, redistribution, SBOM, license/PTX, release-boundary, benchmark,
  candidate, rejection, pkgdown, and public-documentation gates pass without
  deployment.

Remote evidence at the implementation commit:

- Windows, macOS, Ubuntu, and R-devel package checks pass;
- CPU integration and pkgdown pass;
- supply-chain and repository workflow-boundary checks pass;
- Linux and Windows no-CUDA artifact builds pass; and
- CUDA 12.8.1 ABI and unchanged PTX checks pass.

## CP-02D: allocation-free contiguous subset views

Status: complete at implementation commit
`f460ed4692e0df109fc4612f1c1160ca28ff14b6`; the checkpoint-record-only
follow-up must pass the same PR gates before merge.

Review:

- draft PR: <https://github.com/cudaverse/cudaverse/pull/31>;
- target: `develop/0.4` (not `main`);
- no tag, release, CRAN submission, or Pages deployment; and
- GitHub Pages deploy job skipped as required.

Scope:

- detect native selections whose one-based R linear indices form one
  increasing contiguous range;
- represent those selections as offset dense views sharing the source device
  allocation rather than launching a gather and allocating output VRAM;
- preserve independent source/view lifetimes, shape, dimnames, dtype,
  provenance, and downstream kernel compatibility; and
- keep non-contiguous selection on the established native gather path.

Required gate:

- contiguous and non-contiguous selections retain base R parity for integer,
  float32, and float64 tensors;
- a source can be released while its contiguous view remains usable;
- 1,000 nested contiguous views allocate zero additional tracked VRAM and the
  final owner returns to the exact baseline; and
- cuBLAS matmul consumes an offset view without materialization.

Local evidence:

- contiguous and non-contiguous native selections match base R across integer,
  float32, and float64 tensors while retaining dimnames and provenance;
- an offset view remains readable after its source descriptor is released and
  feeds cuBLAS matmul with the correct logical shape and values;
- 1,000 nested contiguous views allocate zero additional tracked VRAM and the
  final shared owner returns to the exact tracked-memory baseline;
- the complete ordinary and explicitly enabled RTX 2000 suites pass, with no
  hardware skips in the latter;
- source tarball SHA-256 is
  `65223CD058FD45CD7A5190F0BCD22177F19540CBB1991B56BEAFFE2D8882FF54`;
- Windows `R CMD check --as-cran` reports 0 errors, 0 warnings, and only the
  expected development-version NOTE; and
- capability, redistribution, SBOM, release-boundary, benchmark, candidate,
  rejection, pkgdown, and public-documentation gates pass without deployment.

Remote evidence at the implementation commit:

- Windows, macOS, Ubuntu, and R-devel package checks pass;
- CPU integration and pkgdown pass;
- supply-chain and repository workflow-boundary checks pass;
- Linux and Windows no-CUDA artifact builds pass; and
- CUDA 12.8.1 ABI and unchanged PTX checks pass.

## CP-02C: same-device tensor reconstruction casts

Status: complete at implementation commit
`49971401036f12a0bc3325fb429df6b277f7b9f2`; the checkpoint-record-only
follow-up must pass the same PR gates before merge.

Review:

- draft PR: <https://github.com/cudaverse/cudaverse/pull/30>;
- target: `develop/0.4` (not `main`);
- no tag, release, CRAN submission, or Pages deployment; and
- GitHub Pages deploy job skipped as required.

Scope:

- make `cuda_tensor(existing_tensor, dtype = ...)` use the existing backend
  cast when its selected device is unchanged;
- avoid a complete device-to-host-to-device round trip;
- retain cross-device transfer behavior and exact integer validation; and
- preserve dtype, device, backend, shape, dimnames, and cast provenance.

Required gate:

- a no-host-transfer adapter contract completes same-device reconstruction;
- native float64-to-float32 parity passes on the RTX 2000; and
- complete backend conformance remains green.

Local evidence:

- a synthetic CUDA adapter whose `to_host` always errors completes the
  same-device reconstruction through exactly one backend cast;
- native float64-to-float32 parity passes on the RTX 2000 while preserving
  device, backend, shape, dimnames, dtype, and cast provenance;
- the complete ordinary suite passes, and the explicitly enabled RTX 2000
  suite passes with no hardware skips;
- source tarball SHA-256 is
  `D126ACD14AFDA0D52C312627A131BBD912C18B2D1B81ED64BB6092F5DF7E01F8`;
- Windows `R CMD check --as-cran` reports 0 errors, 0 warnings, and only the
  expected development-version NOTE;
- capability, redistribution, SBOM, release-boundary, benchmark, candidate,
  report, and rejection self-tests pass; and
- pkgdown and the public documentation boundary pass from a temporary source
  copy without deploying Pages.

Remote evidence at the implementation commit:

- Windows, macOS, Ubuntu, and R-devel package checks pass;
- CPU integration and pkgdown pass;
- supply-chain and repository workflow-boundary checks pass;
- Linux and Windows no-CUDA artifact builds pass; and
- CUDA 12.8.1 ABI and PTX checks pass.

## CP-02B: device-resident replacement dtype conversion

Status: complete at implementation commit
`cd6e573d8c6edd54fbf76d2d151bcfa353d6cac1`; the checkpoint-record-only
follow-up must pass the same PR gates before merge.

Review:

- draft PR: <https://github.com/cudaverse/cudaverse/pull/29>;
- target: `develop/0.4` (not `main`);
- no tag, release, CRAN submission, or Pages deployment; and
- GitHub Pages deploy job skipped as required.

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

Remote evidence at the implementation commit:

- Windows, macOS, Ubuntu, and R-devel package checks pass;
- CPU integration and pkgdown pass;
- supply-chain and repository workflow-boundary checks pass;
- Linux and Windows no-CUDA artifact builds pass; and
- CUDA 12.8.1 ABI and PTX checks pass.

## CP-03D: device-resident native SVD and PCA input

Status: complete at implementation commit
`d921edec0edc293105239df73a6b66f1532620c7`; the checkpoint-record-only
follow-up must pass the same PR gates before merge.

Review:

- draft PR: <https://github.com/cudaverse/cudaverse/pull/35>;
- target: `develop/0.4` (not `main`);
- no tag, release, CRAN submission, or Pages deployment; and
- GitHub Pages deploy job skipped as required.

Scope:

- let a native CUDA `cudatensor` enter `cuda_svd()` and `cuda_pca()` through
  its existing shared device storage rather than a full host round trip;
- validate non-finite values and constant columns on the device, returning
  only two small validation flags to R;
- cast integer and float32 inputs to float64 on the device while sharing an
  existing float64 allocation;
- preserve public results, errors, dimnames, backend selection, and the
  `cudaverse-stage/1` provenance schema; and
- keep CPU, torch, third-party, cross-backend, and cross-device paths on the
  established materialization contract.

Required gate:

- integer, float32, and float64 resident SVD/PCA match the CPU reference under
  the float64 heavy-operation tolerance;
- the first computation stage records backend `native` and reason
  `device_resident_input` without downloading the input matrix;
- non-finite and constant-column inputs retain their established public
  errors;
- 1,000 native device-validation cycles return to the exact tracked-memory
  baseline; and
- ordinary, RTX, no-CUDA, cross-platform, source, supply-chain, and
  documentation gates stay green.

Local evidence:

- the complete ordinary suite and complete explicitly enabled RTX 2000 suite
  pass;
- resident SVD/PCA parity passes for integer, float32, and float64 inputs,
  while row and feature labels remain intact;
- device validation detects `NA`, `NaN`, positive/negative infinity, and
  constant columns without materializing the input matrix on the host;
- 1,000 device-validation cycles return to the exact tracked-memory baseline;
- source PTX rebuilt under CUDA 12.8.1 has SHA-256
  `08227D30A0C253B37BF11D55F93919CF556374FCD1EA19F9315C19A77F6BB327`;
- exact full-vignette source tarball SHA-256 is
  `7A9B8A98FEAA15A4DF6D5B37DAF9DABEB2CB97BB41F808BDEC550726E4A16E9C`;
- Windows `R CMD check --as-cran --no-manual` reports 0 errors, 0 warnings,
  and only the expected development-version incoming NOTE; and
- capability, redistribution, SBOM, release-boundary, benchmark, candidate,
  rejection, pkgdown, and public-documentation gates pass without deployment.

Remote evidence at the implementation commit:

- CPU integration passes;
- supply-chain and repository workflow-boundary checks pass;
- Linux and Windows no-CUDA artifact builds pass; and
- CUDA 12.8.1 ABI and regenerated PTX reproducibility checks pass.
