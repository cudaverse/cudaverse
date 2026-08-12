# cudaverse 0.4.0.9000

- Added `cuda_memory_info()` for backend-aware memory
  observability. Native reports CUDA-driver totals plus cudaverse-owned current
  and peak bytes; torch reports allocated and reserved allocator bytes, while
  unsupported counters remain explicit `NA` values.
- Diffusion maps now retain the device-resident input stage in provenance.
  Native PCA scores are reused directly by CUDA distance instead of being
  uploaded again; kernel construction and eigendecomposition remain explicit
  CPU stages.
- Started the isolated 0.4 development line from the exact review-ready 0.3
  candidate while keeping the 0.3 source and evidence unchanged.
- Extended `cuda_diagnostics()` with a concise health `status`, human-readable
  `summary`, actionable `next_steps`, and a backend comparison table. Strict
  CUDA-unavailable conditions now retain the same reason and guidance.
- Added the ordered 0.4 roadmap and enabled development-line CI coverage.
- Made native CUDA reshape allocation-free by separating dense view metadata
  from shared device-allocation ownership. Nested reshape views remain valid
  after their sources are released and free the allocation exactly once.
- Kept same-backend CUDA tensor replacement device-resident when the replacement
  needs a compatible floating dtype cast, with the cast recorded in provenance.
- Made `cuda_tensor()` cast an existing tensor through its current backend when
  the requested device is unchanged, avoiding a download/upload round trip.
- Made contiguous native CUDA subsets allocation-free shared views while
  retaining device gather for non-contiguous selections.
- Added explicit, provenance-recorded distance batching across base, torch,
  and native backends. Native execution retains input/reference storage and
  cached reference norms across blocks, bounding peak device memory.
- Added explicit k-means batching with a backward-compatible backend contract.
  Native Lloyd iterations now keep data, centres, assignments, and updates on
  the GPU while bounding temporary distance storage by observation batch.
- Kept native PCA prediction scores in shared device storage after returning
  their compatible R matrix, allowing following distance and kNN stages to
  reuse the scores without an upload.
- Added device-side finite/constant validation and resident SVD/PCA dispatch
  for native `cudatensor` inputs. Float32 and integer inputs cast on the GPU;
  the full input matrix is no longer downloaded and uploaded for decomposition.
- Kept native sparse-normalization output resident without downloading the
  margin-sum or normalized-value vectors. The required public COO mirror is
  updated from its existing host metadata, while device validation returns
  only one small status flag.
- Made native sparse normalization share immutable CSR/COO index allocations
  with its source through independent reference counting. Normalized results
  allocate only new values, remain valid after either release order, and free
  shared pattern storage exactly once.
- Made `cuda_sparse()` rematerialize an existing `cudasparse` object directly
  from its stable COO mirror. Same-device format changes share storage, while
  cross-device transfers and zero filtering avoid a temporary Matrix object,
  summary pass, and redundant coordinate sort.
- Removed Matrix construction from sparse PCA preprocessing and sparse
  algorithm transfers. Constant-column checks run directly on the COO mirror
  only when scaling is requested; unscaled PCA skips that scan entirely.

# cudaverse 0.3.0.9000

- Added reproducible full-benchmark Markdown generation and validation. The
  retained assessment is bound to the exact machine-readable report by its
  SHA-256 and source commit, and incomplete drafts cannot pass the final
  summary gate.
- Added a machine-readable 0.3 benchmark contract and report runner for
  synchronized base, torch, and native measurements. The full profile fixes
  workload sizes, warmup/run counts, transfer boundaries, numerical gates,
  peak-memory sources, installed size, and provenance without treating an
  inseparable dense-PCA upload as a measured transfer duration.
- Added an executable public-backend conformance matrix. Every exported
  function is assigned to a diagnostics, tensor, sparse, algorithm, graph, or
  embedding contract case, and the shared suite runs the same small workflows
  on base, torch, and native when CUDA hardware coverage is required.
- Added `t()` for `cudasparse` matrices. Native CUDA transposes CSR backing
  storage on device while preserving stable COO metadata, logical format,
  dimnames, shared ownership, and same-device provenance; compatibility
  backends rebuild storage from the already-public COO metadata.
- Added registry-driven resident native CUDA k-means. Observations and centres
  are uploaded once; distance, stable assignment, accumulation, and Lloyd
  centre updates remain on device, with compact convergence and final-result
  transfers. The base and torch compatibility paths are unchanged.
- Added registry-driven, device-native tensor subsetting and replacement for
  the native CUDA backend, including dtype preservation, dimnames, R recycling,
  and deterministic last-write handling for duplicate indices.
- Added a backend registry and integrated the lightweight native CUDA
  implementation into `cudaverse` without changing the public device API.
- Added native dense casts, reductions, SVD/PCA, exact distance blocks, and
  deterministic top-k/kNN integration.
- Added shared-ownership native COO/CSR storage, sparse Matrix conversion,
  sparse matrix-vector/matrix multiplication, row/column reductions, and
  sparse-preserving normalization.
- Added sparse inputs to `cuda_pca()` and `cuda_knn()`; the native path expands
  them on the GPU and continues through the existing resident dense pipeline.
- Kept `PCA -> distance -> top-k` intermediate data on the GPU for the native
  path and retained the `cudaverse-stage/1` provenance schema.
- Preserved portable CPU behavior and the optional torch compatibility backend.
- Added capability-gated automatic native selection. Native is preferred only
  when the native contract, complete tensor/algorithm capability set,
  driver/cuBLAS/cuSOLVER/PTX runtime, and cached runtime self-test all pass;
  otherwise torch or the recorded CPU fallback retains compatibility.
- Added native float32 matmul plus device-native element-wise arithmetic,
  trailing-dimension broadcasting, reshape, and transpose coverage required by
  the global automatic-selection gate.
- Required SVD and PCA prediction compatibility in that same fail-closed gate,
  and published the bounded Phase 4 release-candidate assessment with links to
  the checksum-pinned RTX evidence.

# cudaverse 0.1.0

- Establishes the general-purpose cudaverse API.
- Adds dense tensor and sparse matrix functionality.
- Adds numerical algorithms, graph workflows, and embedding workflows.
- Preserves the canonical `cuda_provenance()` protocol across all modules.
- Keeps single-cell-specific workflows in the separate `cudacellr` package.
- Fixes CUDA indexing, R column-major reshape semantics, and exact
  self-distance diagonals for compatibility with R torch 0.17.
- Documents the measured, benchmark-gated roadmap toward a lightweight native
  CUDA backend while retaining the current portable CPU fallback.
