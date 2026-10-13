# cudaverse: Lightweight CUDA numerical computing for R

Provides a lightweight interface to graphics processing unit
(GPU)-accelerated numerical computing using 'CUDA'. Dense tensors,
sparse matrices, decompositions, distances, exact nearest neighbours,
clustering, graph workflows, and embeddings share one consistent
interface. The native backend discovers the 'NVIDIA CUDA Driver API',
'cuBLAS', and 'cuSOLVER' libraries at runtime without bundling
'LibTorch' or the 'CUDA Runtime'. Stage-level provenance records the
backend, device, and data transfers used by each result. A portable
implementation supports package validation on systems without 'CUDA'.
Background for the included Leiden community detection and uniform
manifold approximation and projection methods is given by Traag, Waltman
and van Eck (2019)
[doi:10.1038/s41598-019-41695-z](https://doi.org/10.1038/s41598-019-41695-z)
and McInnes et al. (2018)
[doi:10.21105/joss.00861](https://doi.org/10.21105/joss.00861) ,
respectively.

## See also

Useful links:

- <https://cudaverse.github.io/cudaverse/>

- <https://github.com/cudaverse/cudaverse>

- Report bugs at <https://github.com/cudaverse/cudaverse/issues>

## Author

**Maintainer**: Yaoxiang Li <liyaoxiang@outlook.com>

Authors:

- Yaoxiang Li <liyaoxiang@outlook.com>
