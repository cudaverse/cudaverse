# cudaverse: Lightweight CUDA numerical computing for R

Provides a lightweight interface to CUDA-accelerated numerical computing
in R. Dense tensors, sparse matrices, decompositions, distances, exact
nearest neighbours, clustering, graph workflows, and embeddings use one
consistent R API. The native backend discovers NVIDIA driver, cuBLAS,
and cuSOLVER libraries at runtime without bundling LibTorch or a CUDA
runtime. Stage-level provenance records the backend, device, and data
transfers used by each result. A portable implementation supports
package validation on systems without CUDA.

## See also

Useful links:

- <https://cudaverse.github.io/cudaverse/>

- <https://github.com/cudaverse/cudaverse>

- Report bugs at <https://github.com/cudaverse/cudaverse/issues>

## Author

**Maintainer**: Yaoxiang Li <liyaoxiang@outlook.com>

Authors:

- Yaoxiang Li <liyaoxiang@outlook.com>
