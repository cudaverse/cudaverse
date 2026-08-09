# cudaverse 0.2.0.9000

- Added a backend registry and lazy discovery of the optional
  `cudaverseCUDA` extension without changing the public device API.
- Added native dense casts, reductions, SVD/PCA, exact distance blocks, and
  deterministic top-k/kNN integration.
- Kept `PCA -> distance -> top-k` intermediate data on the GPU for the native
  path and retained the `cudaverse-stage/1` provenance schema.
- Preserved portable CPU behavior and the optional torch compatibility backend;
  native remains opt-in pending the complete Phase 2 release audit.

# cudaverse 0.1.0

- Establishes one user-facing package for the general-purpose cudaverse API.
- Incorporates dense tensor functionality from `cudatensr`.
- Incorporates sparse matrix functionality from `cudasparsr`.
- Incorporates numerical algorithms from `cudalearnr`.
- Incorporates graph workflows from `cudagraphR`.
- Incorporates embedding workflows from `cudaembedr`.
- Preserves the canonical `cuda_provenance()` protocol across all modules.
- Keeps single-cell-specific workflows in the separate `cudacellr` package.
- Fixes CUDA indexing, R column-major reshape semantics, and exact
  self-distance diagonals for compatibility with R torch 0.17.
- Documents the measured, benchmark-gated roadmap toward a lightweight native
  CUDA backend while retaining the current portable CPU fallback.
