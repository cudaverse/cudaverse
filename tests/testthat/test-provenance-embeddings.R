.embedding_test_matrix <- function(n = 12L) {
  matrix(
    sin(seq_len(n * 3L) / 5),
    nrow = n,
    ncol = 3L,
    dimnames = list(paste0("cell_", seq_len(n)), paste0("feature_", 1:3))
  )
}

test_that("provenance inspection re-exports the canonical generic", {
  expect_identical(cuda_provenance, cuda_provenance)
  expect_true(utils::isS3stdGeneric(cuda_provenance))
})

test_that("diffusion maps expose actual stage-level provenance", {
  fit <- cuda_diffusion_map(
    .embedding_test_matrix(),
    n_components = 2,
    device = "cpu"
  )
  provenance <- cuda_provenance(fit)

  expect_identical(
    provenance$stage,
    c("distance", "kernel", "eigendecomposition")
  )
  expect_identical(provenance$device, c("cpu", "cpu", "cpu"))
  expect_identical(
    provenance$requested_device,
    c("cpu", "fixed-cpu", "fixed-cpu")
  )
  expect_identical(fit$provenance_schema, "cudaverse-stage/1")
  expect_identical(fit$compute_device, "cpu")
  expect_identical(fit$parameters$n_components, 2L)
  expect_identical(fit$parameters$requested_device, "cpu")
})

test_that("diffusion provenance retains a resident distance input stage", {
  input_stage <- cuda_stage(
    requested_device = "inherited",
    device = "cuda",
    backend = "native",
    selection_reason = "device_resident_input",
    fallback = FALSE,
    output_device = "cuda"
  )
  distance_stage <- cuda_stage(
    requested_device = "cuda",
    device = "cuda",
    backend = "native",
    selection_reason = "explicit_cuda",
    fallback = FALSE,
    output_device = "cpu"
  )
  distances <- matrix(0, 3L, 3L)
  attr(distances, "compute_stages") <- list(
    input_x_materialization = input_stage,
    distance = distance_stage
  )
  attr(distances, "provenance_schema") <- "cudaverse-stage/1"

  stages <- cudaverse:::.embedding_diffusion_distance_stages(distances)

  expect_identical(names(stages), c("distance_input", "distance"))
  expect_identical(
    stages$distance_input$selection_reason,
    "device_resident_input"
  )
  expect_identical(stages$distance$output_device, "cpu")
})

test_that("upstream PCA provenance and compute summary are retained", {
  pca <- cuda_pca(
    .embedding_test_matrix(),
    n_components = 2,
    device = "cpu"
  )
  fit <- cuda_diffusion_map(pca, n_components = 2, device = "cpu")

  expect_identical(fit$source_class, "cuda_pca")
  expect_identical(fit$source_device, "cpu")
  expect_identical(fit$source_compute_device, "cpu")
  expect_s3_class(fit$source_provenance, "cuda_provenance")
  expect_identical(
    fit$source_provenance$stage,
    c("preprocessing", "decomposition")
  )
  expect_identical(
    rownames(embedding_coordinates(fit)),
    rownames(pca$x)
  )
})

test_that("matrix source provenance is validated and derives compute summary", {
  plain <- cuda_diffusion_map(
    .embedding_test_matrix(),
    n_components = 2,
    device = "cpu"
  )
  expect_null(plain$source_provenance)
  expect_identical(plain$source_compute_device, "cpu")

  source <- .embedding_test_matrix()
  attr(source, "compute_stages") <- list(
    gpu_stage = cuda_stage(
      requested_device = "cuda",
      device = "cuda",
      backend = "torch",
      selection_reason = "explicit_cuda",
      output_device = "cpu"
    ),
    cpu_stage = cuda_stage(
      requested_device = "fixed-cpu",
      device = "cpu",
      backend = "base",
      selection_reason = "algorithm_cpu_only"
    )
  )
  attr(source, "provenance_schema") <- "cudaverse-stage/1"
  attr(source, "compute_device") <- "hybrid"

  fit <- cuda_diffusion_map(
    source,
    n_components = 2,
    device = "cpu"
  )
  expect_s3_class(fit$source_provenance, "cuda_provenance")
  expect_identical(fit$source_compute_device, "hybrid")

  attr(source, "provenance_schema") <- "cudaverse-stage/99"
  condition <- tryCatch(
    cuda_diffusion_map(
      source,
      n_components = 2,
      device = "cpu"
    ),
    error = identity
  )
  expect_s3_class(condition, "cudaverse_provenance_schema_error")
})

test_that("automatic diffusion fallback is explicit and CUDA is strict", {
  unavailable <- structure(
    list(
      torch_installed = FALSE,
      torch_version = NA_character_,
      cuda_available = FALSE,
      cuda_device_count = 0L,
      reason = "torch_not_installed",
      detection_error = NULL
    ),
    class = "cuda_diagnostics"
  )
  testthat::local_mocked_bindings(
    cuda_diagnostics = function() unavailable,
    .package = "cudaverse"
  )

  automatic <- cuda_diffusion_map(
    .embedding_test_matrix(),
    device = "auto"
  )
  distance_stage <- cuda_provenance(automatic)[1L, , drop = FALSE]
  expect_identical(distance_stage$requested_device, "auto")
  expect_identical(distance_stage$device, "cpu")
  expect_identical(distance_stage$selection_reason, "torch_not_installed")
  expect_true(distance_stage$fallback)
  expect_identical(automatic$compute_device, "cpu")

  condition <- tryCatch(
    cuda_diffusion_map(
      .embedding_test_matrix(),
      device = "cuda"
    ),
    error = identity
  )
  expect_s3_class(condition, "cudaverse_cuda_unavailable")
})

test_that("fixed-CPU adapters use the same provenance schema", {
  skip_if_not_installed("Rtsne")
  fit <- cuda_tsne(
    .embedding_test_matrix(20),
    perplexity = 3,
    seed = 1
  )
  provenance <- cuda_provenance(fit)

  expect_identical(provenance$stage, "embedding")
  expect_identical(provenance$requested_device, "fixed-cpu")
  expect_identical(provenance$device, "cpu")
  expect_identical(provenance$backend, "Rtsne")
  expect_identical(provenance$selection_reason, "algorithm_cpu_only")
  expect_identical(fit$parameters$seed, 1L)
  expect_identical(fit$parameters$n_components, 2L)
})
