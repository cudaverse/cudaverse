# cudaverse 0.2.0.9000

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
- Removed the cross-repository `cudaverseCUDA` dependency. Native runtime
  libraries are still discovered lazily, so CPU-only installation and checks
  do not require CUDA or a CUDA toolkit.
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
