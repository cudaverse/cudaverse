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

## CP-04A: resident sparse normalization output

Status: complete at implementation commit
`25e9e3a2ac9c781d7be64a43bd98c45682ef10f4`; the checkpoint-record-only
follow-up must pass the same PR gates before merge.

Review:

- draft PR: <https://github.com/cudaverse/cudaverse/pull/36>;
- target: `develop/0.4` (not `main`);
- no tag, release, CRAN submission, or Pages deployment; and
- GitHub Pages deploy job skipped as required.

Scope:

- retain native sparse-normalization storage on CUDA without downloading the
  complete margin-sum or normalized-value vectors;
- validate positive finite margin sums on the device and return only one
  small status flag;
- update the required public host COO value mirror from metadata already held
  by the `cudasparse` object, without constructing a temporary Matrix object;
- preserve public values, errors, dimnames, provenance, and downstream native
  sparse PCA/kNN behavior; and
- leave CPU, torch, and third-party backend contracts compatible.

Required gate:

- row and column normalization parity, including `log1p`, remains within the
  float64 tolerance;
- a backend may return only resident sparse `storage` while the public host
  mirror remains correct;
- an empty or invalid margin retains its structured public error and leaves
  the native backend reusable;
- 1,000 resident normalization cycles return to the exact tracked-memory
  baseline and stay within the 1 MiB whole-device gate; and
- ordinary, RTX, no-CUDA, cross-platform, source, supply-chain, and
  documentation gates stay green.

Local evidence:

- the complete ordinary suite and complete explicitly enabled RTX 2000 suite
  pass;
- the storage-only backend contract and native row-normalization parity pass,
  including the existing public COO value mirror;
- native normalization continues directly into resident sparse PCA and exact
  stable kNN with established provenance;
- empty-margin validation returns a structured `cudaverse_native_error` and
  the backend remains reusable;
- 1,000 resident normalization cycles return to the exact tracked-memory
  baseline and satisfy the 1 MiB whole-device gate;
- source PTX rebuilt under CUDA 12.8.1 has SHA-256
  `839666346300D40E761B6A01608BE77806ED1D63CE36C9001BA6BA5EF5A6215F`;
- exact full-vignette source tarball SHA-256 is
  `FE7C1D97E1AFEB2FD4DAA60A384EE7C66D0C9D29B7D0789F1F96BD21FE7F8EF9`;
- Windows `R CMD check --as-cran --no-manual` reports 0 errors, 0 warnings,
  and only the expected development-version incoming NOTE; and
- capability, redistribution, SBOM, release-boundary, benchmark, candidate,
  rejection, pkgdown, and public-documentation gates pass without deployment.

Remote evidence at the implementation commit:

- CPU integration passes;
- supply-chain and repository workflow-boundary checks pass;
- Linux and Windows no-CUDA artifact builds pass; and
- CUDA 12.8.1 ABI and regenerated PTX reproducibility checks pass.

## CP-04B: shared native sparse-pattern ownership

Status: complete at implementation commit
`9ee6860a54f2d56a7860375d6566a95c177772da`; the checkpoint-record-only
follow-up must pass the same PR gates before merge.

Review:

- draft PR: <https://github.com/cudaverse/cudaverse/pull/37>;
- target: `develop/0.4` (not `main`);
- no tag, release, CRAN submission, or Pages deployment; and
- GitHub Pages deploy job skipped as required.

Scope:

- split native sparse storage ownership into independently reference-counted
  row-index, row-pointer, column-index, and value allocations;
- let normalization results share the immutable sparse pattern with their
  source while retaining independent normalized values;
- eliminate device-to-device copies of all three unchanged index arrays;
- preserve source and result validity when either object is released first;
  and
- release every shared allocation exactly once under explicit release and R
  finalization.

Required gate:

- a normalization result adds exactly `nnz * 8` resident bytes for float64
  values rather than another complete sparse allocation;
- operation peak above the live source is exactly normalized values plus one
  margin-sum vector and one four-byte validation flag;
- source-first and result-first release orders remain readable and return to
  the exact tracked-memory baseline;
- 1,000 normalization cycles and the complete RTX suite remain leak-free; and
- ordinary, no-CUDA, cross-platform, source, supply-chain, and documentation
  gates stay green.

Local evidence:

- the complete ordinary suite and complete explicitly enabled RTX 2000 suite
  pass;
- a 48-nnz normalization adds exactly 384 resident bytes, and its measured
  operation peak above the source is exactly
  `48 * 8 + 16 * 8 + 4 = 516` bytes;
- both source-first and result-first release paths preserve the live object,
  then return to the exact tracked-memory baseline;
- the existing 1,000-cycle resident-normalization and whole-device 1 MiB gates
  pass;
- exact full-vignette source tarball SHA-256 is
  `3E7CBA420B377091E37A892353CDC48F5B0F8865625F8735F353A639674766AE`;
- Windows `R CMD check --as-cran --no-manual` reports 0 errors, 0 warnings,
  and only the expected development-version incoming NOTE; and
- capability, redistribution, SBOM, release-boundary, benchmark, candidate,
  rejection, pkgdown, and public-documentation gates pass without deployment.

Remote evidence at the implementation commit:

- supply-chain and repository workflow-boundary checks pass;
- Linux and Windows no-CUDA artifact builds pass; and
- CUDA 12.8.1 ABI and unchanged PTX reproducibility checks pass.

## CP-04C: direct sparse rematerialization from COO mirrors

Status: complete at implementation commit
`f65c470ddd65519e1fb4fb93ad391d670e4f3076`; the checkpoint-record-only
follow-up must pass the same PR gates before merge.

Review:

- draft PR: <https://github.com/cudaverse/cudaverse/pull/38>;
- target: `develop/0.4` (not `main`);
- no tag, release, CRAN submission, or Pages deployment; and
- GitHub Pages deploy job skipped as required.

Scope:

- rematerialize an existing `cudasparse` input directly from its stable,
  sorted public COO mirror;
- avoid constructing a temporary Matrix object, running `summary()`, and
  sorting coordinates again during cross-device/backend transfer;
- filter explicit zeros directly from the mirror when requested;
- make same-device format-only changes reuse or share backend storage; and
- preserve shape, dimnames, public metadata, backend selection, strict CUDA
  semantics, and provenance.

Required gate:

- a backend contract errors on any attempted `.triplet_matrix()` conversion
  while direct cross-backend rematerialization succeeds;
- uploaded coordinates, values, shape, format, and labels remain exact;
- native same-device CSR-to-COO reformat adds zero tracked VRAM;
- releasing the source first leaves the shared result readable, and final
  cleanup returns to the exact tracked-memory baseline; and
- ordinary, RTX, no-CUDA, cross-platform, source, supply-chain, and
  documentation gates stay green.

Local evidence:

- the complete ordinary suite and complete explicitly enabled RTX 2000 suite
  pass with explicit zero exit status;
- the synthetic cross-backend contract completes while `.triplet_matrix()` is
  bound to an immediate error, and preserves all stable COO metadata;
- native CSR-to-COO reformat adds exactly zero tracked device bytes;
- after source storage is explicitly released, the reformatted result still
  materializes exactly, and final release returns to the tracked baseline;
- exact full-vignette source tarball SHA-256 is
  `B662EE52DF98172A5D45847D27554DA366A8C2FA1F34F303AEEDD47DB591388D`;
- Windows `R CMD check --as-cran --no-manual` reports 0 errors, 0 warnings,
  and only the expected development-version incoming NOTE; and
- capability, redistribution, SBOM, release-boundary, benchmark, candidate,
  rejection, pkgdown, and public-documentation gates pass without deployment.

Remote evidence at the implementation commit:

- supply-chain and repository workflow-boundary checks pass;
- Linux and Windows no-CUDA artifact builds pass; and
- CUDA 12.8.1 ABI and unchanged PTX reproducibility checks pass.

## CP-04D: Matrix-free sparse PCA preprocessing

Status: complete at implementation commit
`edb54161a2b8e2343d17fedf396168dee6257ec8`; the checkpoint-record-only
follow-up must pass the same PR gates before merge.

Review:

- draft PR: <https://github.com/cudaverse/cudaverse/pull/39>;
- target: `develop/0.4` (not `main`);
- no tag, release, CRAN submission, or Pages deployment; and
- GitHub Pages deploy job skipped as required.

Scope:

- skip sparse constant-feature scanning entirely when PCA scaling is disabled;
- compute column sums and squared sums directly from the stable COO mirror
  when scaling is enabled;
- transfer sparse PCA and kNN inputs through direct `cudasparse`
  rematerialization rather than a temporary Matrix object; and
- preserve constant-feature errors, numerical results, backend dispatch,
  provenance, and native device-resident decomposition/distance stages.

Required gate:

- scaled and unscaled sparse PCA dispatch while `.triplet_matrix()` is bound
  to an immediate error;
- constant sparse features are still rejected before backend upload;
- PCA projector/reconstruction and exact stable kNN parity remain green;
- complete ordinary and RTX suites retain zero failures; and
- no-CUDA, cross-platform, source, supply-chain, PTX, and documentation gates
  stay green.

Local evidence:

- the Matrix-free synthetic contract passes both scaled and unscaled sparse
  PCA and records exactly two direct COO uploads;
- a constant zero feature is rejected before a third upload;
- the complete ordinary and explicitly enabled RTX 2000 suites pass;
- exact full-vignette source tarball SHA-256 is
  `B166E51306D55E2833A139F37E07FD40986801D8F9A15906B4F073D0D32FEFC8`;
- Windows `R CMD check --as-cran --no-manual` reports 0 errors, 0 warnings,
  and only the expected development-version incoming NOTE; and
- capability, redistribution, SBOM, release-boundary, benchmark, candidate,
  rejection, pkgdown, and public-documentation gates pass without deployment.

Remote evidence at the implementation commit:

- supply-chain and repository workflow-boundary checks pass;
- Linux and Windows no-CUDA artifact builds pass; and
- CUDA 12.8.1 ABI and unchanged PTX reproducibility checks pass.

## CP-05A: resident PCA to diffusion-distance boundary

Status: complete at implementation commit
`537e176752fa57e20218f76e6041b41d12dfb25b`; the checkpoint-record-only
follow-up must pass the same PR gates before merge.

Review:

- draft PR: <https://github.com/cudaverse/cudaverse/pull/40>;
- target: `develop/0.4` (not `main`);
- no tag, release, CRAN submission, or Pages deployment; and
- graph assembly, Louvain/Leiden, UMAP, t-SNE, and diffusion kernel/eigen
  remain at their documented CPU or hybrid boundaries.

Scope:

- preserve the hidden shared native storage attached to public PCA scores
  through embedding input normalization;
- reuse those resident scores directly in native diffusion-map distance;
- retain the distance input stage in embedding provenance so the avoided
  upload is observable; and
- keep the completed dense distance result, diffusion kernel, and
  eigendecomposition explicitly on the host.

Required gate:

- an RTX contract replaces the native host-upload function with an immediate
  error after PCA and still completes diffusion distance;
- the input external pointer is identical before and after embedding input
  normalization;
- provenance begins with `distance_input` using
  `device_resident_input`, followed by a native distance output to CPU and
  explicit CPU kernel/eigendecomposition stages;
- complete ordinary and RTX suites retain zero failures; and
- no-CUDA, cross-platform, source, supply-chain, PTX, and documentation gates
  stay green.

Local evidence:

- the upload-forbidden RTX test passes and produces finite diffusion
  coordinates with hybrid compute provenance;
- the complete ordinary and explicitly enabled RTX 2000 suites pass;
- exact full-vignette source tarball SHA-256 is
  `AC57EDFD4EDB079BA379130D30E08338F729CBFA672D708F413ACDE191E78643`;
- Windows `R CMD check --as-cran --no-manual` reports 0 errors, 0 warnings,
  and only the expected development-version incoming NOTE; and
- capability, redistribution, SBOM, release-boundary, benchmark, candidate,
  rejection, pkgdown, and public-documentation gates pass without deployment.

Remote evidence at the implementation commit:

- supply-chain and repository workflow-boundary checks pass;
- Linux and Windows no-CUDA artifact builds pass; and
- CUDA 12.8.1 ABI and unchanged PTX reproducibility checks pass.

## CP-06A: backend-aware memory observability

Status: complete at implementation commit
`35e60d977f207a39c0960dca3e4520b1d4cf36e0`; the checkpoint-record-only
follow-up must pass the same PR gates before merge.

Review:

- draft PR: <https://github.com/cudaverse/cudaverse/pull/41>;
- target: `develop/0.4` (not `main`);
- no tag, release, CRAN submission, or Pages deployment; and
- the backend contract gains only an optional memory operation, so existing
  third-party or compatibility factories remain valid.

Scope:

- add `cuda_memory_info()` with stable selection, physical-memory, allocator,
  reason, and error fields;
- report CUDA-driver total/free/used bytes and cudaverse-owned current/peak
  allocations for native;
- report torch allocated/reserved current/peak bytes without estimating
  unsupported physical totals;
- keep CPU fallback, unsupported telemetry, invalid reports, and query errors
  explicit and structured; and
- document first-session self-test peak behavior and memory interpretation.

Required gate:

- native physical totals satisfy `total - free == used`;
- a 128-double allocation adds exactly 1,024 cudaverse-owned bytes and release
  returns current allocation to the exact baseline;
- an injected CUDA OOM raises the established structured condition, retains
  its temporary-allocation high-water mark, and leaves memory telemetry usable;
- ordinary CPU, automatic fallback, torch allocator, malformed-report, and
  query-error contracts pass; and
- complete ordinary, RTX, no-CUDA, source, supply-chain, and documentation
  gates remain green.

Local evidence:

- the live RTX report exposes 16.00 GiB physical total and distinct native
  current/peak ownership counters without retaining a user tensor;
- the injected-error recovery and exact 1,024-byte ownership contracts pass;
- the complete ordinary and explicitly enabled RTX 2000 suites pass;
- exact full-vignette source tarball SHA-256 is
  `9DA5A8D3202E819028FBF6E4562CEBE31B58F54D65B586380DC7FC2731FA6DF5`;
- Windows `R CMD check --as-cran --no-manual` reports 0 errors, 0 warnings,
  and only the expected development-version incoming NOTE; and
- all 39 exports pass capability coverage, while redistribution, SBOM,
  release-boundary, benchmark, candidate, rejection, installed-pkgdown, and
  public-documentation gates pass without deployment.

Remote evidence at the implementation commit:

- supply-chain and repository workflow-boundary checks pass;
- Linux and Windows no-CUDA artifact builds pass; and
- CUDA 12.8.1 ABI and unchanged PTX reproducibility checks pass.

## CP-06B: isolated native R-session lifecycle contract

Status: complete at implementation commit
`be1b3252dfb16817655a4882926ab8e71c97c180`; the RTX report is committed in
the checkpoint follow-up and both commits must pass the same PR gates.

Review:

- draft PR: <https://github.com/cudaverse/cudaverse/pull/42>;
- target: `develop/0.4` (not `main`);
- no tag, release, CRAN submission, or Pages deployment; and
- the hardware runner remains the only environment that executes this strict
  contract; ordinary source installation and checks do not require CUDA.

Scope:

- install the exact source into a new temporary R library;
- launch two distinct `Rscript --vanilla` processes against that installation;
- in each process require native selection and self-test, then run dense shared
  views, sparse normalization, resident PCA/kNN, public memory telemetry,
  injected CUDA OOM recovery, post-error matmul, and exact allocator cleanup;
- exit normally without running workspace save hooks; and
- after exit, verify both process IDs are absent from the NVIDIA compute table.

Required gate:

- both fresh processes use the exact source version and commit;
- both begin and end at zero tracked cudaverse allocation;
- injected OOM remains a structured native condition and the backend is
  reusable in the same session afterward;
- process IDs are distinct and absent from `nvidia-smi` after exit; and
- machine-readable evidence is validated on Windows and Linux no-CUDA jobs
  without rerunning the hardware workload.

Local evidence:

- session PID 7076 reports `baseline=0` and `final=0`;
- session PID 34052 reports `baseline=0` and `final=0`;
- both sessions used source commit
  `be1b3252dfb16817655a4882926ab8e71c97c180`, completed every workflow stage,
  exited successfully, and disappeared from the NVIDIA process table;
- the complete ordinary and explicitly enabled RTX 2000 suites pass;
- exact full-vignette source tarball SHA-256 is
  `273FF92D5ABBA83CECD655F145EAA672E66B31C2990D169663AF86562A3D1DB9`;
- Windows `R CMD check --as-cran --no-manual` reports 0 errors, 0 warnings,
  and only the expected development-version incoming NOTE; and
- session-report, capability, redistribution, SBOM, release-boundary,
  benchmark, candidate, rejection, installed-pkgdown, and public-documentation
  gates pass without deployment.

## CP-07A: public benchmark memory contract

Status: complete at implementation commit
`d1d4e83686d71431dc0c6a46998ca48a9e926f30`; the exact RTX smoke report is
committed in the checkpoint follow-up and both commits must pass the same PR
gates.

Review:

- draft PR: <https://github.com/cudaverse/cudaverse/pull/43>;
- target: `develop/0.4` (not `main`);
- no tag, release, CRAN submission, or Pages deployment; and
- the retained smoke report is a runner and regression gate, not a universal
  performance claim.

Scope:

- make the benchmark runner obtain native and torch allocator observations
  from the public `cuda_memory_info()` contract;
- keep native peak reset as a measurement-only maintainer action outside all
  retained timing samples;
- retain backend allocator peak, post-cleanup current allocation, whole-device
  observations, and their explicit measurement sources in the report schema;
- run the memory instrumentation as a separate execution after all timed
  samples so tracking cannot perturb retained medians; and
- make generated summary titles reflect the actual smoke or full profile
  instead of one historical checkpoint name.

Required gate:

- synthetic native, torch, base, malformed, and error memory reports pass
  positive and rejection self-tests;
- the exact installed source commit completes all smoke cases on base, native,
  and torch with numerical parity;
- every native workload returns current tracked allocation to its pre-run
  baseline;
- raw JSON and human summary identify the same source commit, report hash,
  hardware, profile, and complete state; and
- ordinary, RTX, no-CUDA, cross-platform, source, supply-chain, and
  documentation gates stay green.

Initial RTX 2000 evidence:

- report `inst/reports/benchmarks/CP07-SMOKE.json` is complete, records a clean
  source at `d1d4e83686d71431dc0c6a46998ca48a9e926f30`, and has SHA-256
  `D79BB27942994E6469D51EB0E346BCB1BE6305D83CF562643F70DCB5E6BDBA87`;
- all four smoke workloads pass base/native/torch parity, including exact kNN
  indices and heavy-operation numerical tolerances;
- native resident 64-square matmul medians are 0.000390 seconds for float32
  and 0.000324 seconds for float64 on this run;
- the isolated installed footprint is 1,384,393 bytes for cudaverse versus
  7,367,799,444 bytes for optional torch, while cudaverse bundles zero CUDA
  runtime bytes; and
- the exact smoke source tarball has SHA-256
  `7916F317991F9C526ADA5DBBF27F3B6636C547D5DE890A0B0893B54BFD9C87F3`;
- the complete ordinary suite and complete explicitly enabled RTX 2000 suite
  pass, with no hardware skips in the latter;
- the exact full-vignette checkpoint-follow-up source tarball has SHA-256
  `0A4B1A5F4425A650C48EEB26F87AA78045CE13EB0E1647E9A1CEED7D993DADE9`;
- Windows `R CMD check --as-cran --no-manual` reports 0 errors, 0 warnings,
  and only the expected development-version incoming NOTE;
- redistribution, SBOM, release-boundary, capability, benchmark, candidate,
  session-report, and rejection gates pass; and
- installed pkgdown and the public documentation boundary pass from a
  temporary clone without deploying Pages.

## CP-08A: exact 0.4 candidate evidence contract

Status: complete at implementation head
`db0935376aebd0a258f7b83760b9fdf470ca745c`; this checkpoint-record-only
follow-up must pass the same PR gates before merge.

Review:

- draft PR: <https://github.com/cudaverse/cudaverse/pull/44>;
- target: `develop/0.4` (not `main`);
- no tag, release, CRAN/Bioconductor submission, or Pages deployment; and
- the change is restricted to maintainer evidence policy, validation,
  rejection tests, and internal candidate-review documents.

Scope:

- centralize supported candidate branch/version pairs without weakening the
  exact frozen-ref or clean-source gates;
- accept the historical `0.3.0[.9000]` plus `develop/native-cuda` pair and the
  current `0.4.0[.9000]` plus `develop/0.4` pair;
- reject unsupported versions and every cross-line branch/version mismatch;
- make native-candidate validation follow the same supported release lines;
- define the requirement-by-requirement 0.4 completion audit and candidate
  decision boundary; and
- prevent those maintainer-only audit documents from entering public pkgdown.

Required gate:

- positive and rejection self-tests pass for historical 0.3 and current 0.4
  manifests and native-candidate reports;
- the retained real 0.3 manifest still validates exact source
  `b0ab90f8547d42a5244963bc6b15d0ba223382ff`;
- directly resolved remote `main` and `release/cran-0.1.0` remain frozen at
  `59e15c8c5a56d26e09a594886c875b1b8249f6f9`;
- Windows, macOS, Ubuntu, R-devel, CPU contract, pkgdown, supply-chain,
  no-CUDA artifact, ABI, and PTX checks pass; and
- pkgdown deployment stays disabled for the development PR.

Evidence at the implementation head:

- candidate/native positive and rejection self-tests pass locally;
- the external 0.3 candidate manifest revalidates its exact clean source;
- R-CMD-check run `31602965245` passes Windows, macOS, Ubuntu, and R-devel;
- pkgdown run `31602965300` builds and validates the public boundary while its
  deployment job is skipped;
- integration run `31602965792` passes the CPU and identifier contract; and
- native-integrity run `31602965261` passes supply chain, Windows/Linux
  artifacts, CUDA 12.8.1 ABI, and byte-reproducible PTX gates.

## CP-09A: higher-level workflow boundary audit

Status: complete at implementation commit
`d63cde9ec108f78a9ca6a4182b7c531933127175`; this checkpoint-record-only
follow-up must pass the same PR gates before merge.

Review:

- draft PR: <https://github.com/cudaverse/cudaverse/pull/45>;
- target: `develop/0.4` (not `main`);
- no tag, release, CRAN/Bioconductor submission, or Pages deployment; and
- the implementation changes documentation and boundary validation without
  changing a numerical algorithm or public function signature.

Scope:

- audit graph assembly, Louvain, Leiden, UMAP, t-SNE, diffusion, optional
  SingleCellExperiment reduced dimensions, and the unsupported Seurat input
  boundary against their actual implementations;
- retain graph/community, UMAP, and t-SNE as explicit CPU stages;
- retain the resident native PCA-to-diffusion-distance stage while recording
  CPU kernel construction and eigendecomposition;
- document that SingleCellExperiment reduced dimensions materialize at a host
  boundary while retaining cell identity and compatible provenance;
- remove public guidance to install a separate user package and strengthen the
  pkgdown boundary against obsolete split-package messaging; and
- define parity, determinism, provenance, memory, lifecycle, cross-platform,
  and supply-chain re-entry gates before any higher-level native stage is
  added.

Required gate:

- graph, embedding, provenance, and real SingleCellExperiment tests pass;
- all 39 exports retain valid capability-matrix coverage;
- roxygen output is synchronized with the revised UMAP/t-SNE contract;
- the rendered public site contains no obsolete separate-package message;
- cross-platform source, CPU, no-CUDA artifact, supply-chain, ABI/PTX, and
  public-documentation checks stay green; and
- development PR deployment remains skipped.

Evidence at the implementation commit:

- targeted graph/embedding/provenance/SingleCellExperiment tests pass locally;
- the capability-matrix unit and standalone checks cover all 39 exports;
- R-CMD-check run `31604756238` passes Windows, macOS, Ubuntu, and R-devel;
- pkgdown run `31604755975` builds and passes the strengthened public boundary
  while its deployment job is skipped;
- integration run `31604757080` passes the CPU and identifier contract; and
- native-integrity run `31604755982` passes supply chain, Windows/Linux
  artifacts, CUDA 12.8.1 ABI, and byte-reproducible PTX gates.

## CP-10A: benchmark substage progress observability

Status: complete at implementation commit
`9c361fdaa33d73b7855e6794e6a0519a4e475c78`; this checkpoint-record-only
follow-up must pass the same PR gates before merge.

Review:

- draft PR: <https://github.com/cudaverse/cudaverse/pull/46>;
- target: `develop/0.4` (not `main`);
- no tag, release, CRAN/Bioconductor submission, or Pages deployment; and
- the change is restricted to maintainer benchmark execution and tests, with
  no public R API, numerical, tolerance, report-schema, or resume-policy change.

Scope:

- log started and complete events for cold, every warmup, and every retained
  timed run with exact case/backend/scope and run counts;
- log the separate post-timing memory pass under the same identity;
- invoke progress callbacks outside retained timing intervals;
- report completed-run duration without including the callback or subsequent
  garbage collection; and
- document that progress messages are observability only and cannot be counted
  or resumed as benchmark evidence.

Required gate:

- callback event order, index, total, duration, and rendered messages pass
  exact self-tests;
- timing values and collected stage observations remain unchanged;
- benchmark checkpoint/recovery, report, summary, and memory rejection suites
  remain green;
- Windows, macOS, Ubuntu, R-devel, CPU, pkgdown, supply-chain, no-CUDA artifact,
  ABI, and PTX checks pass; and
- development PR deployment remains skipped.

Evidence at the implementation commit:

- timing/progress, checkpoint/recovery, report, summary, and memory tool tests
  pass locally;
- R-CMD-check run `31607054120` passes Windows, macOS, Ubuntu, and R-devel;
- pkgdown run `31607054132` builds and validates the public boundary while its
  deployment job is skipped;
- integration run `31607054521` passes the CPU and identifier contract; and
- native-integrity run `31607054148` passes supply chain, Windows/Linux
  artifacts, CUDA 12.8.1 ABI, and byte-reproducible PTX gates.

## CP-07B: complete full benchmark baseline

Status: complete for exact clean benchmark source
`deb77127eebf68e0c6f788e9e10c40c5cb8dacfc`; this is a checkpoint baseline,
not final-candidate evidence. The final clean 0.4 candidate must repeat the
same unchanged contract after all accepted performance work is merged.

Review:

- branch: `agent/16-full-benchmark-evidence`;
- target: `develop/0.4` (not `main`);
- no tag, release, CRAN/Bioconductor submission, or Pages deployment; and
- retained timings are descriptive for the recorded RTX 2000 machine,
  software, input, and provenance only.

Scope:

- retain the complete 12-case/36-backend `cudaverse-benchmark/1` report;
- preserve cold, five-warmup/ten-timed-run median and p95 distributions for
  host and resident boundaries without inferring one from the other;
- retain numerical parity, exact kNN indices, transfer semantics, public
  allocator observations, whole-device cleanup, installed size, and backend
  provenance;
- generate the human summary only from the complete report; and
- validate both the machine report and its exact generated summary with
  fail-closed checkers.

Evidence:

- `inst/reports/benchmarks/CP07-FULL.json` is complete, records clean source
  `deb77127eebf68e0c6f788e9e10c40c5cb8dacfc`, contains all 12 cases and all
  36 base/native/torch results, and has SHA-256
  `707AECBD0D72A5B1E1BD209A0892B555C41771FE9E3870F5414B7613650C380A`;
- every backend result has `status = complete` and passes its numerical gate;
  all kNN comparisons retain exact indices, including the final
  `sparse-50000x128@0.01` workload;
- for that final workload, host-boundary medians are 217.632 seconds for base,
  2.800 seconds for native, and 126.739 seconds for torch; these values locate
  the old CPU/torch stable-selection bottleneck and are not universal claims;
- the same native workload has maximum normalized, PCA-projector,
  reconstruction, and kNN-distance errors of `1.29e-16`, `8.70e-12`,
  `1.49e-11`, and `3.14e-11`; its tracked and whole-device post-cleanup
  differences are both zero;
- the installed footprint is 1,447,216 bytes for cudaverse versus
  7,367,799,444 bytes for optional torch, while cudaverse bundles zero CUDA
  runtime bytes;
- generated summary `CP07-FULL.md` has SHA-256
  `5E04846E268B6505D12B8180FEDDC8B8A48AD88C0E759B1B41C56D444B9DF315`;
- report and summary checker logs have SHA-256
  `572B9C77660EE0B17AD4AFFBE0AF09269CF3B36A86BAAD7C41786662597BDDF1`
  and `98F093CB8DB3C2CADB1C9DAEDAE4E499D236C0FC111C126071C19F27D776C392`;
- the validated-resume run log has SHA-256
  `22BBCAE3245933F5C70F8C6EFF5B3A05D13C7A7FB9B53B8668AAF4566CF68B4F`;
  the runner accepted only complete prior checkpoints and finished the final
  report atomically; and
- the exact installed benchmark source tarball has SHA-256
  `8B123AD1F948710A920415A63BAAD748D722BC622D35DF2613D963F98865058E`.
