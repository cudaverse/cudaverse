.provenance_matrix <- function() {
  matrix(
    c(
      1, 2, 3,
      2, 3, 4,
      4, 2, 1,
      5, 3, 2,
      8, 7, 9,
      9, 8, 8
    ),
    ncol = 3,
    byrow = TRUE
  )
}

test_that("provenance inspection re-exports the canonical generic", {
  expect_identical(cuda_provenance, cuda_provenance)
  expect_true(utils::isS3stdGeneric(cuda_provenance))
})

test_that("all numerical result types expose one provenance schema", {
  x <- .provenance_matrix()

  svd_fit <- cuda_svd(x, device = "cpu")
  expect_identical(cuda_provenance(svd_fit)$stage, "decomposition")
  expect_identical(svd_fit$compute_device, "cpu")
  expect_identical(svd_fit$backend, "base")
  expect_identical(svd_fit$requested_device, "cpu")

  pca_fit <- cuda_pca(x, n_components = 2, device = "cpu")
  expect_identical(
    cuda_provenance(pca_fit)$stage,
    c("preprocessing", "decomposition")
  )
  expect_identical(pca_fit$compute_device, "cpu")
  expect_identical(pca_fit$backend, "stats")
  expect_identical(
    pca_fit$parameters,
    list(n_components = 2L, center = TRUE, scale = FALSE)
  )

  distance <- cuda_distance(x, device = "cpu")
  distance_provenance <- cuda_provenance(distance)
  expect_identical(distance_provenance$stage, "distance")
  expect_identical(distance_provenance$output_device, "cpu")
  expect_identical(attr(distance, "compute_device"), "cpu")
  expect_identical(attr(distance, "requested_device"), "cpu")
  expect_identical(attr(distance, "parameters")$batch_size, nrow(x))
  expect_identical(attr(distance, "parameters")$batches, 1L)

  knn <- cuda_knn(x, k = 2, batch_size = 3, device = "cpu")
  expect_identical(
    cuda_provenance(knn)$stage,
    c("distance", "neighbor_selection")
  )
  expect_identical(knn$compute_device, "cpu")
  expect_identical(knn$parameters$batch_size, 3L)

  kmeans <- cuda_kmeans(
    x,
    centers = x[c(1L, 4L), , drop = FALSE],
    device = "cpu"
  )
  expect_identical(
    cuda_provenance(kmeans)$stage,
    c("initialization", "distance", "assignment", "center_update")
  )
  expect_identical(kmeans$compute_device, "cpu")
  expect_identical(kmeans$backend, "base")
  expect_identical(kmeans$parameters$batch_size, nrow(x))
  expect_identical(kmeans$parameters$batches, 1L)
})

test_that("post-fit predictions expose their actual compute stages", {
  x <- .provenance_matrix()
  colnames(x) <- paste0("feature_", seq_len(ncol(x)))
  rownames(x) <- paste0("row_", seq_len(nrow(x)))
  newdata <- x[1:2, rev(colnames(x)), drop = FALSE]

  pca <- cuda_pca(x, n_components = 2, device = "cpu")
  projected <- predict(pca, newdata, device = "cpu")
  projected_provenance <- cuda_provenance(projected)
  expect_identical(projected_provenance$stage, "projection")
  expect_identical(projected_provenance$device, "cpu")
  expect_identical(attr(projected, "compute_device"), "cpu")
  expect_identical(attr(projected, "backend"), "base")
  expect_identical(
    attr(projected, "parameters"),
    list(n_components = 2L)
  )
  expect_identical(attr(projected, "source_class"), "matrix")

  kmeans <- cuda_kmeans(x, centers = 2, seed = 1, device = "cpu")
  assigned <- predict(kmeans, newdata, device = "cpu")
  assigned_provenance <- cuda_provenance(assigned)
  expect_identical(
    assigned_provenance$stage,
    c("distance", "assignment")
  )
  expect_identical(assigned_provenance$device, c("cpu", "cpu"))
  expect_identical(attr(assigned, "compute_device"), "cpu")
  expect_identical(attr(assigned, "backend"), "base")
  expect_identical(
    attr(assigned, "parameters"),
    list(type = "cluster", metric = "euclidean")
  )

  projected_from_model <- predict(pca, newdata)
  projected_model_provenance <- cuda_provenance(projected_from_model)
  expect_identical(
    projected_model_provenance$requested_device,
    "inherited"
  )
  expect_identical(
    projected_model_provenance$selection_reason,
    "model_device"
  )
  expect_false(projected_model_provenance$fallback)
  expect_identical(
    attr(projected_from_model, "requested_device"),
    "inherited"
  )

  assigned_from_model <- predict(kmeans, newdata)
  assigned_model_provenance <- cuda_provenance(assigned_from_model)
  expect_identical(
    assigned_model_provenance$requested_device,
    c("inherited", "fixed-cpu")
  )
  expect_identical(
    assigned_model_provenance$selection_reason,
    c("model_device", "algorithm_cpu_only")
  )
  expect_false(any(assigned_model_provenance$fallback))
  expect_identical(
    attr(assigned_from_model, "requested_device"),
    "inherited"
  )

  model_distances <- predict(kmeans, newdata, type = "distance")
  distance_model_provenance <- cuda_provenance(model_distances)
  expect_identical(
    distance_model_provenance$requested_device,
    "inherited"
  )
  expect_identical(
    distance_model_provenance$selection_reason,
    "model_device"
  )

  expect_null(attr(predict(pca), "compute_stages", exact = TRUE))
  expect_null(attr(predict(kmeans), "compute_stages", exact = TRUE))
})

test_that("automatic fallback is visible and explicit CUDA remains strict", {
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

  fit <- cuda_knn(.provenance_matrix(), k = 2, device = "auto")
  provenance <- cuda_provenance(fit)
  expect_identical(fit$requested_device, "auto")
  expect_identical(provenance$requested_device[[1L]], "auto")
  expect_identical(provenance$selection_reason[[1L]], "torch_not_installed")
  expect_true(provenance$fallback[[1L]])
  expect_identical(provenance$device, c("cpu", "cpu"))
  expect_s3_class(
    tryCatch(
      cuda_pca(.provenance_matrix(), device = "cuda"),
      error = identity
    ),
    "cudaverse_cuda_unavailable"
  )
})

test_that("print methods disclose hybrid-aware compute metadata", {
  svd_fit <- cuda_svd(.provenance_matrix(), device = "cpu")
  pca_fit <- cuda_pca(.provenance_matrix(), device = "cpu")
  knn_fit <- cuda_knn(.provenance_matrix(), k = 2, device = "cpu")
  kmeans_fit <- cuda_kmeans(.provenance_matrix(), centers = 2, seed = 1,
                            device = "cpu")

  expect_output(print(svd_fit), "compute=cpu")
  expect_output(print(pca_fit), "compute=cpu")
  expect_output(print(knn_fit), "distance_device=cpu compute=cpu")
  expect_output(print(kmeans_fit), "distance_device=cpu compute=cpu")
})
