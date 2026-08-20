# Package index

## Device and provenance

- [`cuda_available()`](https://cudaverse.github.io/cudaverse/reference/cuda_available.md)
  : Detect a usable CUDA backend
- [`cuda_diagnostics()`](https://cudaverse.github.io/cudaverse/reference/cuda_diagnostics.md)
  : Diagnose the optional CUDA runtime
- [`cuda_memory_info()`](https://cudaverse.github.io/cudaverse/reference/cuda_memory_info.md)
  : Inspect CUDA memory
- [`cuda_select_device()`](https://cudaverse.github.io/cudaverse/reference/cuda_select_device.md)
  : Select a computation device without hiding fallback
- [`cuda_stage()`](https://cudaverse.github.io/cudaverse/reference/cuda_stage.md)
  : Record one compute stage
- [`cuda_provenance()`](https://cudaverse.github.io/cudaverse/reference/cuda_provenance.md)
  : Inspect actual compute provenance

## Dense tensors

- [`cuda_tensor()`](https://cudaverse.github.io/cudaverse/reference/cuda_tensor.md)
  : Create a GPU-aware tensor
- [`Ops(`*`<cudatensor>`*`)`](https://cudaverse.github.io/cudaverse/reference/cudatensor-operators.md)
  [`` `%*%`( ``*`<cudatensor>`*`)`](https://cudaverse.github.io/cudaverse/reference/cudatensor-operators.md)
  : Arithmetic operators for GPU-aware tensors
- [`` `[`( ``*`<cudatensor>`*`)`](https://cudaverse.github.io/cudaverse/reference/cudatensor-subset.md)
  [`` `[<-`( ``*`<cudatensor>`*`)`](https://cudaverse.github.io/cudaverse/reference/cudatensor-subset.md)
  : Subset and replace tensor values
- [`dimnames(`*`<cudatensor>`*`)`](https://cudaverse.github.io/cudaverse/reference/dimnames.cudatensor.md)
  : Inspect tensor dimension labels
- [`tensor_broadcast_to()`](https://cudaverse.github.io/cudaverse/reference/tensor_broadcast_to.md)
  : Broadcast a tensor to a compatible shape
- [`tensor_device()`](https://cudaverse.github.io/cudaverse/reference/tensor_device.md)
  : Inspect tensor device and backend
- [`tensor_matmul()`](https://cudaverse.github.io/cudaverse/reference/tensor_matmul.md)
  : Matrix multiplication for tensors
- [`tensor_reshape()`](https://cudaverse.github.io/cudaverse/reference/tensor_reshape.md)
  : Reshape a tensor without changing its values
- [`tensor_shape()`](https://cudaverse.github.io/cudaverse/reference/tensor_shape.md)
  : Inspect tensor shape
- [`tensor_sum()`](https://cudaverse.github.io/cudaverse/reference/tensor_sum.md)
  [`tensor_mean()`](https://cudaverse.github.io/cudaverse/reference/tensor_sum.md)
  : Tensor reductions
- [`to_cpu()`](https://cudaverse.github.io/cudaverse/reference/to_cpu.md)
  : Transfer tensor data to base R
- [`to_device()`](https://cudaverse.github.io/cudaverse/reference/to_device.md)
  : Transfer a tensor to a device

## Sparse matrices

- [`cuda_sparse()`](https://cudaverse.github.io/cudaverse/reference/cuda_sparse.md)
  : Create a GPU-aware sparse matrix
- [`as_coo()`](https://cudaverse.github.io/cudaverse/reference/as_coo.md)
  [`as_csr()`](https://cudaverse.github.io/cudaverse/reference/as_coo.md)
  : Convert sparse storage format
- [`dimnames(`*`<cudasparse>`*`)`](https://cudaverse.github.io/cudaverse/reference/dimnames.cudasparse.md)
  : Inspect sparse matrix dimension labels
- [`t(`*`<cudasparse>`*`)`](https://cudaverse.github.io/cudaverse/reference/t.cudasparse.md)
  : Transpose a GPU-aware sparse matrix
- [`to_dgCMatrix()`](https://cudaverse.github.io/cudaverse/reference/to_dgCMatrix.md)
  : Convert to an R sparse matrix
- [`sparse_info()`](https://cudaverse.github.io/cudaverse/reference/sparse_info.md)
  : Inspect sparse matrix metadata
- [`sparse_matmul_dense()`](https://cudaverse.github.io/cudaverse/reference/sparse_matmul_dense.md)
  : Sparse matrix by dense matrix multiplication
- [`sparse_matvec()`](https://cudaverse.github.io/cudaverse/reference/sparse_matvec.md)
  : Sparse matrix-vector multiplication
- [`sparse_normalize()`](https://cudaverse.github.io/cudaverse/reference/sparse_normalize.md)
  : Normalize sparse rows or columns without densifying
- [`sparse_row_sums()`](https://cudaverse.github.io/cudaverse/reference/sparse_row_sums.md)
  [`sparse_col_sums()`](https://cudaverse.github.io/cudaverse/reference/sparse_row_sums.md)
  : Sparse row and column reductions

## Numerical algorithms

- [`cuda_svd()`](https://cudaverse.github.io/cudaverse/reference/cuda_svd.md)
  : GPU-aware singular value decomposition
- [`cuda_pca()`](https://cudaverse.github.io/cudaverse/reference/cuda_pca.md)
  : GPU-aware principal component analysis
- [`cuda_distance()`](https://cudaverse.github.io/cudaverse/reference/cuda_distance.md)
  : Pairwise distances with an optional CUDA backend
- [`cuda_knn()`](https://cudaverse.github.io/cudaverse/reference/cuda_knn.md)
  : k-nearest neighbours
- [`cuda_kmeans()`](https://cudaverse.github.io/cudaverse/reference/cuda_kmeans.md)
  : GPU-aware k-means clustering
- [`predict(`*`<cuda_pca>`*`)`](https://cudaverse.github.io/cudaverse/reference/predict.cuda_pca.md)
  : Project observations with a fitted CUDA-aware PCA model
- [`predict(`*`<cuda_kmeans>`*`)`](https://cudaverse.github.io/cudaverse/reference/predict.cuda_kmeans.md)
  : Assign observations with a fitted CUDA-aware k-means model

## Graph workflows

- [`cuda_knn_graph()`](https://cudaverse.github.io/cudaverse/reference/cuda_knn_graph.md)
  : Build a sparse graph from nearest neighbours
- [`as_adjacency_matrix()`](https://cudaverse.github.io/cudaverse/reference/as_adjacency_matrix.md)
  : Extract a graph adjacency matrix
- [`cuda_louvain()`](https://cudaverse.github.io/cudaverse/reference/cuda_louvain.md)
  : Cluster a graph with Louvain
- [`cuda_leiden()`](https://cudaverse.github.io/cudaverse/reference/cuda_leiden.md)
  : Cluster a graph with Leiden

## Embeddings

- [`cuda_umap()`](https://cudaverse.github.io/cudaverse/reference/cuda_umap.md)
  : UMAP embedding
- [`cuda_tsne()`](https://cudaverse.github.io/cudaverse/reference/cuda_tsne.md)
  : t-SNE embedding
- [`cuda_diffusion_map()`](https://cudaverse.github.io/cudaverse/reference/cuda_diffusion_map.md)
  : Diffusion-map-style embedding
- [`embedding_coordinates()`](https://cudaverse.github.io/cudaverse/reference/embedding_coordinates.md)
  : Extract embedding coordinates
