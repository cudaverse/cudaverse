# cudaverse 0.4 higher-level workflow boundary audit

This audit records the implemented graph, clustering, embedding, and optional
SingleCellExperiment boundaries. It is evidence for scope decisions, not a
performance claim and not authorization for a tag, release, Pages deployment,
or repository submission.

## Decision table

| Public workflow | Current execution | Output boundary | 0.4 decision |
|---|---|---|---|
| `cuda_knn_graph()` | consumes stable kNN output and constructs the weighted sparse graph with `Matrix` on CPU | host graph with upstream kNN provenance | retain CPU construction; the graph object is a host interoperability boundary and no native graph storage contract exists |
| `as_adjacency_matrix()` | metadata conversion only | host `Matrix` object | retain host conversion |
| `cuda_louvain()` | `igraph` Louvain on CPU | host membership/modularity plus graph and upstream kNN provenance | retain CPU dependency backend; do not imply CUDA execution |
| `cuda_leiden()` | `igraph` Leiden on CPU | host membership/modularity plus graph and upstream kNN provenance | retain CPU dependency backend; do not imply CUDA execution |
| `cuda_umap()` | `uwot` on CPU | host coordinates with source lineage and an explicit fixed-CPU embedding stage | retain CPU adapter; a native UMAP implementation is outside the current numerical kernel contract |
| `cuda_tsne()` | `Rtsne` on CPU | host coordinates with source lineage and an explicit fixed-CPU embedding stage | retain CPU adapter for the same reason |
| `cuda_diffusion_map()` | distance uses the selected cudaverse backend; kernel construction and eigendecomposition run on CPU | host coordinates; provenance is CPU or hybrid according to the actual distance stage | retain the CP-05A resident PCA-to-distance input boundary; do not describe the full workflow as GPU-resident |
| optional `SingleCellExperiment` input | selects and materializes one reduced dimension as a finite host matrix while retaining cell names and compatible source provenance | UMAP/t-SNE stay CPU; diffusion may upload for distance and then returns to CPU | keep optional through `Suggests`; require explicit `reduced_dim` unless compatible metadata or one unique `PCA` resolves it |
| Seurat input | no public adapter | unsupported with a documented conversion boundary | do not add a large optional API without an object-version, identity, assay/layer, and provenance contract |

## Evidence coverage

- `tests/testthat/test-graph.R` covers graph weights, labels, clustering, and
  invalid inputs.
- `tests/testthat/test-provenance-graph.R` proves fixed-CPU graph/community
  stages and preservation of upstream kNN provenance.
- `tests/testthat/test-embeddings.R` covers common result shape, optional
  backend adapters, composition from PCA, and invalid parameters.
- `tests/testthat/test-provenance-embeddings.R` proves fixed-CPU UMAP/t-SNE,
  strict explicit CUDA, recorded automatic fallback, hybrid diffusion, and
  the resident native distance-input stage.
- `tests/testthat/test-sce-input.R` covers real SingleCellExperiment input,
  stable selection, ambiguity, stale and invalid metadata, cell identity,
  source lineage, and all three embedding entry points.
- `inst/reports/backend-capability-matrix.csv` records the same CPU/hybrid and
  host-output boundaries for every public export.

## Re-entry gates for more device residency

Graph/community or embedding stages move to native CUDA only after all of the
following exist:

1. a package-owned numerical and object contract that does not expose a second
   user package or silently replace dependency semantics;
2. deterministic or explicitly bounded stochastic parity, including stable
   graph ties, labels, seeds, and disconnected/degenerate cases;
3. stage-level provenance and strict explicit-CUDA error behavior;
4. bounded-memory execution and allocator/whole-device cleanup evidence on the
   RTX 2000; and
5. cross-platform no-CUDA installation, source, ABI/PTX, SBOM, license, and
   compact-artifact gates.

Until then the CPU and hybrid stages above are intentional public boundaries,
not missing fallback disclosures. The final 0.4 candidate must re-run the
public conformance suite and verify that its rendered documentation agrees
with this table.
