# Backend support and release boundaries

`cudaverse` has one public API and three possible computation backends.
The portable base backend is always present. The optional torch backend
uses a CUDA-enabled torch installation. The lightweight native backend
dynamically loads an NVIDIA driver, cuBLAS, cuSOLVER, and package PTX
only when CUDA is diagnosed or requested.

Installing the package does not download a CUDA runtime or LibTorch.
Explicit `device = "cuda"` requests are strict and never silently fall
back to CPU. `device = "auto"` may fall back, and the reason is
preserved in provenance.

## Read the matrix

The table below covers every exported function. Its status vocabulary
is:

- `direct`: the main numerical operation is implemented by that backend;
- `hybrid`: meaningful stages are split between that backend and CPU
  code;
- `cpu_only`: the operation deliberately runs on CPU for the current
  surface;
- `metadata`: validation or metadata without a new numerical kernel;
- `probe`: backend diagnosis or selection; and
- `host`: an explicit host materialization.

`direct` does not imply that the public return value remains on the
device. The `result_location` column states whether the result is
host-resident, device-resident, cached for a native continuation, or
backend-dependent. The `contract_case` column connects every export to
one of six executable public-API workflows: diagnostics, tensor, sparse,
algorithm, graph, or embedding.

| api | category | contract_case | cpu_base | torch_cuda | native_cuda | result_location | notes |
|:---|:---|:---|:---|:---|:---|:---|:---|
| as_adjacency_matrix | graph | graph | metadata | metadata | metadata | host | Returns the already CPU-resident sparse adjacency matrix |
| as_coo | sparse | sparse | metadata | metadata | metadata | same_device | Changes the logical public format view without numerical work |
| as_csr | sparse | sparse | metadata | metadata | metadata | same_device | Changes the logical public format view without numerical work |
| cuda_available | diagnostics | diagnostics | probe | probe | probe | host | Reports whether native or torch can satisfy an explicit CUDA request |
| cuda_diagnostics | diagnostics | diagnostics | probe | probe | probe | host | Reports compatibility fields plus backend health summary status table and actionable next steps |
| cuda_diffusion_map | embedding | embedding | direct | hybrid | hybrid | host | Native reuses device-resident PCA scores for distance while kernel construction and eigendecomposition are CPU; optional SingleCellExperiment reduced dimensions begin at a host boundary |
| cuda_distance | algorithm | algorithm | direct | direct | direct | host | Distance kernels use the selected backend and return an R matrix |
| cuda_kmeans | algorithm | algorithm | direct | hybrid | direct | host | Native keeps repeated distance assignment accumulation and centre updates on device; torch retains CPU updates |
| cuda_knn | algorithm | algorithm | direct | direct | direct | host | Native and torch with stable-sort support select deterministic top-k on device and transfer only final indices and distances; older torch APIs use the recorded compatibility path |
| cuda_knn_graph | graph | graph | direct | cpu_only | cpu_only | host | Graph assembly always uses Matrix on CPU from a completed kNN result |
| cuda_leiden | graph | graph | direct | cpu_only | cpu_only | host | Community detection always uses igraph on CPU |
| cuda_louvain | graph | graph | direct | cpu_only | cpu_only | host | Community detection always uses igraph on CPU |
| cuda_memory_info | diagnostics | diagnostics | metadata | direct | direct | host | Reports native driver and package allocator bytes or torch allocator bytes without estimating unsupported counters; first selection can run the cached self-test |
| cuda_pca | algorithm | algorithm | direct | hybrid | direct | host_with_native_cache | Torch sparse input materializes before dense PCA while native has a sparse path |
| cuda_provenance | provenance | diagnostics | metadata | metadata | metadata | host | Returns the recorded stage table without new numerical work |
| cuda_select_device | diagnostics | diagnostics | probe | probe | probe | host | Selects native then torch for auto and records CPU fallback |
| cuda_sparse | sparse | sparse | direct | direct | direct | same_device | Constructs Matrix torch COO or native shared storage |
| cuda_stage | provenance | diagnostics | metadata | metadata | metadata | host | Validates and constructs one provenance stage |
| cuda_svd | algorithm | algorithm | direct | direct | direct | host | SVD uses the selected backend and materializes the public result |
| cuda_tensor | tensor | tensor | direct | direct | direct | same_device | Constructs storage in the selected backend |
| cuda_tsne | embedding | embedding | direct | cpu_only | cpu_only | host | Always uses Rtsne on CPU and preserves upstream provenance; optional SingleCellExperiment reduced dimensions are materialized on the host |
| cuda_umap | embedding | embedding | direct | cpu_only | cpu_only | host | Always uses uwot on CPU and preserves upstream provenance; optional SingleCellExperiment reduced dimensions are materialized on the host |
| embedding_coordinates | embedding | embedding | metadata | metadata | metadata | host | Returns coordinates already stored in the embedding result |
| sparse_col_sums | sparse | sparse | direct | cpu_only | direct | host | Torch uses the host metadata mirror while native reduces on device |
| sparse_info | sparse | sparse | metadata | metadata | metadata | host | Returns public sparse metadata only |
| sparse_matmul_dense | sparse | sparse | direct | hybrid | direct | backend_dependent | Torch computes on CUDA then returns CPU while native keeps a dense tensor resident |
| sparse_matvec | sparse | sparse | direct | hybrid | hybrid | host | GPU multiplication is followed by explicit host vector materialization |
| sparse_normalize | sparse | sparse | direct | hybrid | direct | same_device | Torch normalizes the host mirror then uploads while native normalizes device storage |
| sparse_row_sums | sparse | sparse | direct | cpu_only | direct | host | Torch uses the host metadata mirror while native reduces on device |
| tensor_broadcast_to | tensor | tensor | direct | direct | direct | same_device | Uses the selected tensor backend |
| tensor_device | tensor | tensor | metadata | metadata | metadata | host | Returns tensor metadata only |
| tensor_matmul | tensor | tensor | direct | direct | direct | same_device | Uses the selected tensor backend |
| tensor_mean | tensor | tensor | direct | direct | direct | same_device | Uses the selected tensor backend |
| tensor_reshape | tensor | tensor | direct | direct | direct | same_device | Native reshape shares storage ownership |
| tensor_shape | tensor | tensor | metadata | metadata | metadata | host | Returns tensor metadata only |
| tensor_sum | tensor | tensor | direct | direct | direct | same_device | Uses the selected tensor backend |
| to_cpu | transfer | tensor | host | host | host | host | Explicitly materializes a base R object |
| to_device | transfer | tensor | direct | direct | direct | same_device | CPU uses base while CUDA uses the selected compatible backend |
| to_dgCMatrix | transfer | sparse | host | host | host | host | Explicitly materializes a Matrix dgCMatrix |

The installed matrix and `NAMESPACE` are checked together: a new or
removed export fails the package tests until this table is updated. The
same compact contract workflows always run against base. On the
protected hardware gate, they must also pass against both torch and
native without skipped cases. The suite checks numerical parity,
metadata, provenance, intentional CPU/hybrid boundaries, missing-runtime
behavior, invalid inputs, and backend reuse after an error.
Backend-specific ownership, injected CUDA failure, interruption, and
1,000-cycle lifecycle tests continue to run beside this public surface
suite.

## Intentional CPU and hybrid boundaries

The following boundaries are deliberate and are not reasons to install
more packages:

- native CUDA tensor subsetting and replacement keep tensor values on
  the device; missing indices and compatibility backends without
  indexing operations use a recorded CPU round trip;
- [`t()`](https://rdrr.io/r/base/t.html) preserves `cudasparse` device,
  format, and labels; native backing storage is transposed on CUDA while
  compatibility backends rebuild from the stable public COO metadata;
- native CUDA k-means keeps distance, stable assignment, accumulation,
  and centre updates on the device; the torch compatibility path uses
  GPU distance and CPU assignment/centre updates;
- exact kNN keeps each distance block and stable top-k selection on the
  native backend, and does the same on torch versions that expose stable
  device sort; only the final index and distance matrices cross the host
  boundary;
- kNN graph construction uses `Matrix`, while Louvain and Leiden use
  `igraph`;
- UMAP and t-SNE use `uwot` and `Rtsne` on CPU; and
- diffusion maps may calculate distance on CUDA, but construct the
  kernel and eigendecomposition on CPU.

`SingleCellExperiment` support is optional. Embedding entry points
materialize the selected reduced dimension as a host matrix and retain
its cell names and compatible source provenance. UMAP and t-SNE remain
CPU operations. Diffusion maps can move that matrix through the
documented CUDA distance stage, followed by the intentional CPU kernel
and eigendecomposition stages. Seurat objects are not currently a public
input type.

Use
[`cuda_provenance()`](https://cudaverse.github.io/cudaverse/reference/cuda_provenance.md)
on a result to inspect each actual compute and transfer stage. A hybrid
result is not presented as fully GPU-resident.

## Native acceptance gate

On 2026-08-10, source commit `fdceea66d13c84c4adf731b8cd4fd0826c738ca6`
passed the complete explicitly gated native test file on an NVIDIA RTX
2000 Ada Generation GPU with 16 GB VRAM and driver 595.97. No hardware
case was skipped. The gate covered:

- float32/float64 transfer, matmul, arithmetic, reductions, reshape,
  broadcasting, transpose, subsetting, and replacement;
- SVD, PCA, PCA prediction, distance, deterministic top-k, and kNN;
- COO/CSR ownership, sparse multiplication, reductions, normalization,
  sparse PCA, and sparse kNN;
- structured CUDA and R error recovery, time-limit interruption, and
  backend reuse; and
- shared ownership, allocation telemetry, and 1,000-cycle dense and
  sparse lifecycle tests.

The test process referenced user-provided `cublas64_12.dll` and
`cusolver64_11.dll` files already present on that machine. These
libraries are not copied into or redistributed with `cudaverse`. Exact
release-candidate evidence is regenerated at the final candidate commit
rather than inferred from this checkpoint.

## Inspect a local installation

``` r

library(cudaverse)

diagnostics <- cuda_diagnostics()
diagnostics$selected_backend
#> [1] "base"
diagnostics$backend_diagnostics$native$capabilities
#>  [1] "driver-detection"     "allocation"           "transfer"            
#>  [4] "cast"                 "matmul"               "reduce"              
#>  [7] "arithmetic"           "reshape"              "broadcast"           
#> [10] "transpose"            "subset"               "replacement"         
#> [13] "matrix-validation"    "svd"                  "svd-resident"        
#> [16] "pca"                  "pca-resident"         "pca-predict"         
#> [19] "distance"             "distance-batched"     "kmeans"              
#> [22] "kmeans-batched"       "knn"                  "stable-topk"         
#> [25] "sparse"               "sparse-coo"           "sparse-csr"          
#> [28] "sparse-transpose"     "sparse-normalize"     "sparse-matmul"       
#> [31] "sparse-reduce"        "sparse-pca"           "sparse-knn"          
#> [34] "synchronize"          "shared-ownership"     "memory-observability"
#> [37] "dtype-float32"        "dtype-float64"        "runtime-self-test"
diagnostics$backend_diagnostics$native$operations
#>  [1] "algorithm_distance"           "algorithm_distance_batched"  
#>  [3] "algorithm_kmeans"             "algorithm_kmeans_batched"    
#>  [5] "algorithm_knn_block"          "algorithm_knn_prepare"       
#>  [7] "algorithm_knn_select"         "algorithm_matrix_validate"   
#>  [9] "algorithm_pca"                "algorithm_pca_predict"       
#> [11] "algorithm_pca_storage"        "algorithm_sparse_knn_prepare"
#> [13] "algorithm_sparse_pca"         "algorithm_svd"               
#> [15] "algorithm_svd_storage"        "binary"                      
#> [17] "broadcast"                    "cast"                        
#> [19] "from_host"                    "matmul"                      
#> [21] "memory_info"                  "reduce"                      
#> [23] "release"                      "replace"                     
#> [25] "reshape"                      "sparse_from_coo"             
#> [27] "sparse_matmul_dense"          "sparse_normalize"            
#> [29] "sparse_reduce"                "sparse_release"              
#> [31] "sparse_share"                 "sparse_to_dense"             
#> [33] "sparse_to_host"               "sparse_transpose"            
#> [35] "subset"                       "synchronize"                 
#> [37] "test_inject_cuda_error"       "to_host"                     
#> [39] "transpose"
```

`capabilities` are user-facing feature declarations. `operations` are
the callable internal registry entries. Native automatic selection
additionally requires the compatible contract schema, complete runtime
components, and a passing cached self-test.
