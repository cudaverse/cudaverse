# Lightweight native CUDA roadmap

## Positioning

The long-term CUDA identity of cudaverse is a lightweight, R-native execution
layer rather than a wrapper around a general deep-learning framework:

> Lightweight native CUDA for R -- no Python, no PyTorch, no Conda.

This sentence is a target for the native-backend release. It must not be used
as a claim about cudaverse 0.1.0, whose optional CUDA backend is currently
implemented with R torch.

## Why build a native backend?

The native backend is intended to provide five concrete benefits:

1. **Smaller installation.** CUDA users should not need the complete LibTorch
   runtime. The R torch CUDA installation measured on the Windows RTX 2000
   development machine occupied 6.86 GB. Native releases will publish their
   own measured installed size and list every downloaded component.
2. **Stable R semantics.** cudaverse will own the conversion between R's
   one-based, column-major arrays and device memory instead of adapting to
   changes in another R tensor API's indexing and reshape behavior.
3. **Device-resident workflows.** PCA, pairwise distances, and top-k neighbour
   selection should share device buffers so intermediate results do not make
   avoidable round trips through host memory.
4. **Controlled objects and provenance.** `cudatensor` and `cudasparse` memory
   layout, allocation, finalization, backend version, and stage-level compute
   provenance will be defined by cudaverse contracts.
5. **Backend independence.** The public API and returned object contracts will
   remain stable when native CUDA, CPU, or a future backend is selected.

## Architecture

The public R API will call an internal backend contract rather than a specific
runtime. The initial choices will be `native`, `torch`, and `cpu`, with `auto`
selecting the best validated backend. During migration, torch remains an
optional compatibility backend and the CPU path remains the portable fallback.

The optional
[`cudaverseCUDA`](https://github.com/cudaverse/cudaverseCUDA) extension now
implements the native sparse Phase 3 prototype. The main package discovers it
lazily; it remains installable and fully checkable without the extension or
CUDA. The native implementation uses NVIDIA's focused numerical libraries
where they are a good fit:

- cuBLAS for dense matrix multiplication;
- cuSOLVER for decompositions; and
- small cudaverse-owned kernels for reductions, sparse operations, distance
  calculation, broadcasting, and top-k selection. A future cuSPARSE adapter
  remains an optimization option, not a runtime requirement.

## Delivery stages

1. **Complete:** extract and test the internal backend contract without
   changing the public API.
2. **Complete (Stage 1):** device diagnostics,
   allocation/finalization, transfer, synchronization, and cuBLAS matrix
   multiplication.
3. **Complete (dense Phase 2):** casts, reductions, cuSOLVER SVD/PCA,
   pairwise distance blocks, and stable top-k/kNN. PCA scores feed distance and
   top-k through shared device storage; only ordinary PCA result fields and the
   final compact neighbour result are materialized in R.
4. **Complete (sparse Phase 3):** shared-ownership COO/CSR storage, Matrix
   conversion, sparse matvec/matmul, row/column reductions, normalization, and
   sparse-input PCA/kNN through a device-resident dense continuation.
5. Make `native` the automatic CUDA choice only after every operation reachable
   through automatic device selection passes the same CPU/GPU parity contract.
   Selecting native globally before element-wise and broadcasting coverage
   would still regress existing workflows.
6. Retain torch for one compatibility cycle, then reassess whether it still
   belongs in `Suggests`.

## Claims require evidence

Lightweight is a measurable contract, not an adjective. Before the native
backend becomes the default, a release candidate must publish reproducible
evidence for:

- clean-machine download and installed size;
- number and identity of external runtime components;
- cold-start time;
- CPU/native/torch parity across supported operations;
- end-to-end benchmarks that include transfers; and
- Windows and Linux installation and hardware checks.

The benchmark report must distinguish resident GPU kernel timing from complete
R workflow timing. It must also report CPU-only and hybrid stages so users can
see where acceleration actually occurred.

Stage-one, dense Phase 2, and sparse Phase 3 parity, provenance, error recovery,
interruption, allocation high-water, and benchmark contracts are versioned in
the `cudaverseCUDA` repository. The CycloneDX SBOM and third-party license
inventory explicitly distinguish package-owned PTX from dynamically discovered
NVIDIA runtime libraries. Native can be released as an explicit backend, but
global automatic preference remains gated on the uncovered tensor arithmetic
and broadcasting surface.

The completed
[RTX 2000 sparse Phase 3 report](https://github.com/cudaverse/cudaverseCUDA/blob/main/inst/reports/STAGE3.md)
links its human-readable summary to the full machine-readable timing, parity,
provenance, error-recovery, and lifecycle evidence.
