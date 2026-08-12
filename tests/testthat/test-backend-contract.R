test_that("backend registry exposes the stable operation contract", {
  backend <- cudaverse:::.backend_get("base")

  expect_identical(backend$name, "base")
  expect_identical(backend$device, "cpu")
  expect_true(all(c(
    "diagnostics", "capabilities", "from_host", "to_host", "cast",
    "matmul", "reduce", "synchronize", "release", "error_translate"
  ) %in% names(backend)))
  expect_true(all(vapply(
    backend[setdiff(names(backend), c("name", "device"))],
    is.function,
    logical(1)
  )))
})

test_that("diagnostics extend rather than replace legacy fields", {
  diagnostics <- cuda_diagnostics()

  expect_true(all(c(
    "torch_installed", "torch_version", "cuda_available",
    "cuda_device_count", "reason", "detection_error"
  ) %in% names(diagnostics)))
  expect_type(diagnostics$available_backends, "character")
  expect_true("base" %in% diagnostics$available_backends)
  expect_type(diagnostics$auto_eligible_backends, "character")
  expect_type(diagnostics$auto_selection_reason, "character")
  expect_length(diagnostics$auto_selection_reason, 1L)
  expect_type(diagnostics$selected_backend, "character")
  expect_length(diagnostics$selected_backend, 1L)
  expect_named(diagnostics$backend_diagnostics, c("torch", "native"))
  expect_identical(
    diagnostics$status,
    if (diagnostics$cuda_available) "cuda_ready" else "cpu_only"
  )
  expect_type(diagnostics$summary, "character")
  expect_length(diagnostics$summary, 1L)
  expect_type(diagnostics$next_steps, "character")
  expect_s3_class(diagnostics$backend_status, "data.frame")
  expect_named(
    diagnostics$backend_status,
    c(
      "backend", "device", "installed", "available", "auto_eligible",
      "selected", "reason", "error"
    )
  )
  expect_identical(
    diagnostics$backend_status$backend,
    c("base", "torch", "native")
  )
  expect_identical(
    diagnostics$backend_status$device,
    c("cpu", "cuda", "cuda")
  )
  expect_false(diagnostics$backend_status$auto_eligible[[1L]])
  expect_identical(sum(diagnostics$backend_status$selected), 1L)
  expect_identical(
    diagnostics$backend_status$backend[diagnostics$backend_status$selected],
    diagnostics$selected_backend
  )
  expect_true(all(c(
    "capabilities", "operations"
  ) %in% names(diagnostics$backend_diagnostics$torch)))
  expect_true("algorithm_pca_predict" %in%
                diagnostics$backend_diagnostics$torch$operations)
})

test_that("diagnostic guidance maps stable runtime reasons to actions", {
  expect_match(
    cudaverse:::.cuda_diagnostic_next_steps("driver_unavailable"),
    "NVIDIA display driver",
    fixed = TRUE
  )
  expect_match(
    cudaverse:::.cuda_diagnostic_next_steps("native_runtime_incomplete"),
    "CUDAVERSE_CUBLAS_PATH",
    fixed = TRUE
  )
  expect_match(
    cudaverse:::.cuda_diagnostic_next_steps("native_self_test_failed"),
    "self_test$error",
    fixed = TRUE
  )
})

test_that("strict CUDA conditions retain actionable diagnostics", {
  unavailable <- structure(
    list(
      cuda_available = FALSE,
      auto_selection_reason = "driver_unavailable",
      reason = "torch_not_installed",
      selected_backend = "base",
      next_steps = "Install or update the NVIDIA display driver."
    ),
    class = "cuda_diagnostics"
  )
  testthat::local_mocked_bindings(cuda_diagnostics = function() unavailable)

  condition <- tryCatch(cuda_select_device("cuda"), error = identity)
  expect_s3_class(condition, "cudaverse_cuda_unavailable")
  expect_identical(condition$reason, "driver_unavailable")
  expect_identical(condition$next_steps, unavailable$next_steps)
  expect_match(conditionMessage(condition), "Next:", fixed = TRUE)
})

test_that("capabilities and callable operations have distinct contracts", {
  base <- cudaverse:::.backend_get("base")

  expect_true(all(c(
    "svd", "pca", "pca-predict", "distance", "distance-batched", "knn"
  ) %in% base$capabilities()))
  expect_true(all(c(
    "algorithm_svd", "algorithm_pca", "algorithm_pca_predict",
    "algorithm_distance", "algorithm_distance_batched",
    "algorithm_knn_prepare", "algorithm_knn_block"
  ) %in% cudaverse:::.backend_operations(base)))
})

test_that("invalid capability declarations fail registration", {
  factory <- cudaverse:::.base_backend_factory()
  factory$name <- "invalid-capabilities"
  factory$capabilities <- function() c("matmul", "matmul")

  condition <- tryCatch(
    cudaverse:::.backend_register(factory, replace = TRUE),
    error = identity
  )
  expect_s3_class(condition, "cudaverse_backend_contract_error")
  expect_identical(condition$backend, "invalid-capabilities")
  expect_identical(condition$operation, "capabilities")
})

test_that("native probe failures do not change the legacy detection field", {
  testthat::local_mocked_bindings(
    .backend_diagnostics = function(name) {
      if (identical(name, "torch")) {
        return(list(
          installed = FALSE, available = FALSE, device_count = 0L,
          version = NA_character_, reason = "torch_not_installed",
          detection_error = NULL
        ))
      }
      list(
        installed = TRUE, available = FALSE, device_count = 0L,
        version = NA_character_, reason = "backend_error",
        detection_error = "injected native probe failure"
      )
    }
  )

  diagnostics <- cuda_diagnostics()
  expect_null(diagnostics$detection_error)
  expect_identical(
    diagnostics$backend_diagnostics$native$detection_error,
    "injected native probe failure"
  )
})

test_that("CPU tensor and algorithm adapters match their R references", {
  x <- matrix(seq_len(6), 2, 3)
  y <- matrix(seq_len(6), 3, 2)
  product <- tensor_matmul(
    cuda_tensor(x, device = "cpu"),
    cuda_tensor(y, device = "cpu")
  )

  expect_equal(to_cpu(product), x %*% y)
  expect_identical(tensor_device(product), c(device = "cpu", backend = "base"))
  expect_equal(
    as.vector(cuda_distance(x, device = "cpu")),
    as.vector(as.matrix(stats::dist(x)))
  )
})

test_that("same-backend replacement casts remain device-resident", {
  factory <- cudaverse:::.base_backend_factory()
  factory$name <- "resident-cast-test"
  factory$device <- "cuda"
  original_cast <- factory$cast
  cast_calls <- 0L
  factory$cast <- function(storage, dtype) {
    cast_calls <<- cast_calls + 1L
    original_cast(storage, dtype)
  }
  factory$to_host <- function(storage) {
    stop("unexpected host transfer", call. = FALSE)
  }
  cudaverse:::.backend_register(factory, replace = TRUE)
  on.exit(
    rm(list = factory$name, envir = cudaverse:::.cudaverse_backends),
    add = TRUE
  )

  x <- cudaverse:::.new_cudatensor(
    array(c(1, 2, 3, 4), dim = 4L),
    "cuda", factory$name, "float64", 4L
  )
  replacement <- cudaverse:::.new_cudatensor(
    array(c(8L, 9L), dim = 2L),
    "cuda", factory$name, "integer", 2L
  )

  x[c(2L, 4L)] <- replacement
  expect_identical(cast_calls, 1L)
  expect_equal(as.vector(x$storage), c(1, 8, 3, 9), tolerance = 0)
  expect_identical(x$dtype, "float64")
  expect_identical(
    cuda_provenance(x)$stage,
    c("replacement_cast", "replacement")
  )
  expect_identical(
    cuda_provenance(x)$selection_reason,
    c("replacement_dtype_conversion", "inherited_device")
  )
})

test_that("same-device tensor reconstruction casts without host transfer", {
  factory <- cudaverse:::.base_backend_factory()
  factory$name <- "resident-construction-cast-test"
  factory$device <- "cuda"
  original_cast <- factory$cast
  cast_calls <- 0L
  factory$cast <- function(storage, dtype) {
    cast_calls <<- cast_calls + 1L
    original_cast(storage, dtype)
  }
  factory$to_host <- function(storage) {
    stop("unexpected host transfer", call. = FALSE)
  }
  cudaverse:::.backend_register(factory, replace = TRUE)
  on.exit(
    rm(list = factory$name, envir = cudaverse:::.cudaverse_backends),
    add = TRUE
  )
  testthat::local_mocked_bindings(
    cuda_select_device = function(device) {
      list(
        requested_device = device,
        device = "cuda",
        backend = factory$name,
        selection_reason = "contract_test",
        fallback = FALSE
      )
    }
  )

  x <- cudaverse:::.new_cudatensor(
    array(c(1, 2, 3, 4), dim = c(2L, 2L)),
    "cuda", factory$name, "float64", c(2L, 2L),
    dimnames = list(c("r1", "r2"), c("c1", "c2"))
  )
  result <- cuda_tensor(x, device = "cuda", dtype = "float32")

  expect_identical(cast_calls, 1L)
  expect_identical(result$dtype, "float32")
  expect_identical(result$storage, array(c(1, 2, 3, 4), c(2L, 2L)))
  expect_identical(tensor_device(result), c(
    device = "cuda", backend = factory$name
  ))
  expect_identical(cuda_provenance(result)$stage, "cast")
  expect_identical(
    dimnames(result), list(c("r1", "r2"), c("c1", "c2"))
  )
})

test_that("missing backend capabilities return structured conditions", {
  factory <- list(
    name = "contract-test",
    device = "cuda",
    diagnostics = function() list(available = FALSE),
    capabilities = function() "matmul",
    from_host = identity,
    to_host = identity,
    matmul = function(x, y) x,
    synchronize = function() invisible(TRUE),
    release = function(x) invisible(TRUE),
    error_translate = cudaverse:::.backend_default_error_translate(
      "contract-test"
    )
  )
  cudaverse:::.backend_register(factory, replace = TRUE)
  on.exit(
    rm(list = "contract-test", envir = cudaverse:::.cudaverse_backends),
    add = TRUE
  )

  condition <- tryCatch(
    cudaverse:::.backend_call("contract-test", "reduce", NULL),
    error = identity
  )
  expect_s3_class(condition, "cudaverse_backend_capability_error")
  expect_s3_class(condition, "cudaverse_backend_error")
  expect_identical(condition$backend, "contract-test")
  expect_identical(condition$operation, "reduce")
})

test_that("native auto-selection requires its contract, capabilities, and self-test", {
  old_option <- options(cudaverse.cuda_backends = NULL)
  on.exit(options(old_option), add = TRUE)

  registry <- cudaverse:::.cudaverse_backends
  had_native <- exists("native", envir = registry, inherits = FALSE)
  old_native <- if (had_native) get("native", envir = registry) else NULL
  on.exit({
    if (had_native) {
      assign("native", old_native, envir = registry)
    } else if (exists("native", envir = registry, inherits = FALSE)) {
      rm(list = "native", envir = registry)
    }
  }, add = TRUE)

  capabilities <- cudaverse:::.native_auto_required_capabilities
  factory <- cudaverse:::.base_backend_factory()
  factory$name <- "native"
  factory$device <- "cuda"
  factory$contract <- function() list(schema = "cudaverse-backend/1")
  factory$capabilities <- function() capabilities
  cudaverse:::.backend_register(factory, replace = TRUE)

  healthy <- list(
    installed = TRUE,
    available = TRUE,
    device_count = 1L,
    version = "test",
    reason = "cuda_available",
    detection_error = NULL,
    runtime_complete = TRUE,
    self_test = list(passed = TRUE)
  )
  torch <- list(
    installed = TRUE,
    available = TRUE,
    device_count = 1L,
    version = "test",
    reason = "cuda_available",
    detection_error = NULL
  )

  status <- cudaverse:::.backend_selection_status("native", healthy)
  expect_true(status$capability_compatible)
  expect_true(status$self_test_passed)
  expect_true(status$auto_eligible)
  expect_length(status$missing_auto_capabilities, 0L)
  expect_identical(
    cudaverse:::.backend_select_cuda(list(
      backend_diagnostics = list(torch = torch, native = healthy)
    )),
    "native"
  )

  capabilities <- setdiff(capabilities, "stable-topk")
  incompatible <- cudaverse:::.backend_selection_status("native", healthy)
  expect_false(incompatible$auto_eligible)
  expect_identical(
    incompatible$auto_selection_reason,
    "native_capability_incompatible"
  )
  expect_identical(incompatible$missing_auto_capabilities, "stable-topk")
  expect_identical(
    cudaverse:::.backend_select_cuda(list(
      backend_diagnostics = list(torch = torch, native = healthy)
    )),
    "torch"
  )

  capabilities <- setdiff(
    cudaverse:::.native_auto_required_capabilities,
    "pca-predict"
  )
  incomplete_prediction <- cudaverse:::.backend_selection_status(
    "native", healthy
  )
  expect_false(incomplete_prediction$auto_eligible)
  expect_identical(
    incomplete_prediction$missing_auto_capabilities,
    "pca-predict"
  )

  capabilities <- cudaverse:::.native_auto_required_capabilities
  failed <- healthy
  failed$self_test <- list(passed = FALSE, error = "injected")
  failed_status <- cudaverse:::.backend_selection_status("native", failed)
  expect_false(failed_status$auto_eligible)
  expect_identical(
    failed_status$auto_selection_reason,
    "native_self_test_failed"
  )
})

test_that("optional backend operations are discovered without changing contract", {
  expect_false(cudaverse:::.backend_has_operation("base", "missing_method"))
  expect_true(cudaverse:::.backend_has_operation("base", "algorithm_distance"))
})

test_that("distance dispatches one explicit backend batching contract", {
  values <- matrix(seq_len(30) / 7, 10L, 3L)
  calls <- new.env(parent = emptyenv())
  calls$batched <- 0L
  factory <- cudaverse:::.base_backend_factory()
  factory$name <- "distance-batch-contract"
  factory$device <- "cuda"
  factory$algorithm_distance <- function(...) {
    stop("unexpected unbatched distance dispatch", call. = FALSE)
  }
  factory$algorithm_distance_batched <- function(
      x, y, metric, batch_size, source_x, source_y) {
    calls$batched <- calls$batched + 1L
    expect_identical(x, values)
    expect_identical(y, values)
    expect_identical(source_x, values)
    expect_identical(source_y, values)
    expect_identical(metric, "euclidean")
    expect_identical(batch_size, 3L)
    as.matrix(stats::dist(values))
  }
  cudaverse:::.backend_register(factory, replace = TRUE)
  on.exit(
    rm(list = factory$name, envir = cudaverse:::.cudaverse_backends),
    add = TRUE
  )
  testthat::local_mocked_bindings(
    .learn_device = function(device) list(
      requested_device = "cuda",
      device = "cuda",
      backend = factory$name,
      selection_reason = "contract_test",
      fallback = FALSE
    )
  )

  result <- cuda_distance(values, device = "cuda", batch_size = 3L)

  expect_identical(calls$batched, 1L)
  expect_equal(result, as.matrix(stats::dist(values)), ignore_attr = TRUE)
  expect_identical(attr(result, "backend"), factory$name)
  expect_identical(attr(result, "parameters")$batch_size, 3L)
  expect_identical(attr(result, "parameters")$batches, 4L)
})

test_that("sparse algorithms dispatch by operation rather than backend name", {
  dense <- matrix(c(1, 0, 2, 0, 3, 1, 4, 0), nrow = 4, byrow = TRUE)
  calls <- new.env(parent = emptyenv())
  calls$pca <- 0L
  calls$knn <- 0L
  factory <- cudaverse:::.base_backend_factory()
  factory$name <- "sparse-contract-test"
  factory$device <- "cuda"
  factory$algorithm_sparse_pca <- function(storage, shape, n_components,
                                           center, scale) {
    calls$pca <- calls$pca + 1L
    cudaverse:::.base_backend_factory()$algorithm_pca(
      dense, n_components, center, scale
    )
  }
  factory$algorithm_sparse_knn_prepare <- function(storage, shape, metric) {
    calls$knn <- calls$knn + 1L
    dense
  }
  factory$algorithm_knn_select <- function(storage, values, k, metric,
                                           batch_size) {
    distances <- as.matrix(stats::dist(storage))
    diag(distances) <- Inf
    index <- t(vapply(
      seq_len(nrow(storage)),
      function(row) order(distances[row, ], seq_len(nrow(storage)))[seq_len(k)],
      integer(k)
    ))
    selected <- matrix(
      distances[cbind(rep(seq_len(nrow(storage)), each = k), as.vector(t(index)))],
      nrow = nrow(storage),
      ncol = k,
      byrow = TRUE
    )
    list(index = index, distance = selected)
  }
  cudaverse:::.backend_register(factory, replace = TRUE)
  on.exit(
    rm(list = "sparse-contract-test", envir = cudaverse:::.cudaverse_backends),
    add = TRUE
  )
  testthat::local_mocked_bindings(
    .learn_device = function(device) list(
      requested_device = "cuda",
      device = "cuda",
      backend = "sparse-contract-test",
      selection_reason = "contract_test",
      fallback = FALSE
    )
  )

  sparse <- cuda_sparse(dense, device = "cpu")
  sparse$device <- "cuda"
  sparse$backend <- "sparse-contract-test"
  sparse$backend_id <- "sparse-contract-test"
  sparse$storage <- list(test = TRUE)

  pca <- cuda_pca(sparse, n_components = 1L, device = "cuda")
  knn <- cuda_knn(sparse, k = 2L, device = "cuda")

  expect_identical(calls$pca, 1L)
  expect_identical(calls$knn, 1L)
  expect_identical(pca$backend, "sparse-contract-test")
  expect_identical(knn$backend, "sparse-contract-test")
  expect_identical(knn$device, "cuda")
})

test_that("k-means dispatches resident updates by backend operation", {
  values <- rbind(c(0, 0), c(0, 1), c(10, 10), c(10, 11))
  initial <- values[c(1L, 3L), , drop = FALSE]
  calls <- new.env(parent = emptyenv())
  calls$kmeans <- 0L
  factory <- cudaverse:::.base_backend_factory()
  factory$name <- "kmeans-contract-test"
  factory$device <- "cuda"
  factory$algorithm_kmeans <- function(x, centers, iter_max, tolerance) {
    calls$kmeans <- calls$kmeans + 1L
    expect_identical(x, values)
    expect_identical(centers, initial)
    expect_identical(iter_max, 10L)
    expect_identical(tolerance, 1e-8)
    list(
      cluster = c(1L, 1L, 2L, 2L),
      centers = rbind(c(0, 0.5), c(10, 10.5)),
      withinss = c(0.5, 0.5),
      iter = 2L,
      converged = TRUE
    )
  }
  cudaverse:::.backend_register(factory, replace = TRUE)
  on.exit(
    rm(list = "kmeans-contract-test", envir = cudaverse:::.cudaverse_backends),
    add = TRUE
  )
  testthat::local_mocked_bindings(
    .learn_device = function(device) list(
      requested_device = "cuda",
      device = "cuda",
      backend = "kmeans-contract-test",
      selection_reason = "contract_test",
      fallback = FALSE
    )
  )

  fit <- cuda_kmeans(
    values,
    centers = initial,
    iter.max = 10L,
    tolerance = 1e-8,
    device = "cuda"
  )

  expect_identical(calls$kmeans, 1L)
  expect_identical(fit$cluster, c(1L, 1L, 2L, 2L))
  expect_equal(fit$centers, rbind(c(0, 0.5), c(10, 10.5)))
  expect_identical(fit$backend, "kmeans-contract-test")
  expect_identical(fit$compute_device, "hybrid")
  expect_identical(fit$compute_stages$assignment$output_device, "cuda")
  expect_identical(fit$compute_stages$center_update$output_device, "cuda")
  expect_identical(fit$compute_stages$finalization$output_device, "cpu")
})

test_that("k-means prefers the bounded-memory backend contract", {
  values <- rbind(c(0, 0), c(0, 1), c(10, 10), c(10, 11))
  initial <- values[c(1L, 3L), , drop = FALSE]
  calls <- new.env(parent = emptyenv())
  calls$legacy <- 0L
  calls$batched <- 0L
  factory <- cudaverse:::.base_backend_factory()
  factory$name <- "kmeans-batched-contract-test"
  factory$device <- "cuda"
  result <- list(
    cluster = c(1L, 1L, 2L, 2L),
    centers = rbind(c(0, 0.5), c(10, 10.5)),
    withinss = c(0.5, 0.5),
    iter = 2L,
    converged = TRUE
  )
  factory$algorithm_kmeans <- function(...) {
    calls$legacy <- calls$legacy + 1L
    result
  }
  factory$algorithm_kmeans_batched <- function(
      x, centers, iter_max, tolerance, batch_size) {
    calls$batched <- calls$batched + 1L
    expect_identical(x, values)
    expect_identical(centers, initial)
    expect_identical(iter_max, 10L)
    expect_identical(tolerance, 1e-8)
    expect_identical(batch_size, 2L)
    result
  }
  cudaverse:::.backend_register(factory, replace = TRUE)
  on.exit(rm(
    list = "kmeans-batched-contract-test",
    envir = cudaverse:::.cudaverse_backends
  ), add = TRUE)
  testthat::local_mocked_bindings(
    .learn_device = function(device) list(
      requested_device = "cuda",
      device = "cuda",
      backend = "kmeans-batched-contract-test",
      selection_reason = "contract_test",
      fallback = FALSE
    )
  )

  fit <- cuda_kmeans(
    values, centers = initial, iter.max = 10L, tolerance = 1e-8,
    batch_size = 2L, device = "cuda"
  )

  expect_identical(calls$batched, 1L)
  expect_identical(calls$legacy, 0L)
  expect_identical(fit$parameters$batch_size, 2L)
  expect_identical(fit$parameters$batches, 2L)
})

test_that("sparse transpose dispatches by backend operation", {
  calls <- new.env(parent = emptyenv())
  calls$transpose <- 0L
  factory <- cudaverse:::.base_backend_factory()
  factory$name <- "sparse-transpose-contract"
  factory$device <- "cuda"
  factory$sparse_transpose <- function(storage) {
    calls$transpose <- calls$transpose + 1L
    expect_identical(storage, list(marker = "source"))
    list(marker = "transposed")
  }
  cudaverse:::.backend_register(factory, replace = TRUE)
  on.exit(
    rm(
      list = "sparse-transpose-contract",
      envir = cudaverse:::.cudaverse_backends
    ),
    add = TRUE
  )

  source <- matrix(c(1, 0, 2, 0, 3, 0), nrow = 2L)
  sparse <- cuda_sparse(source, device = "cpu")
  sparse$device <- "cuda"
  sparse$backend <- "sparse-transpose-contract"
  sparse$backend_id <- "sparse-transpose-contract"
  sparse$storage <- list(marker = "source")
  result <- t(sparse)

  expect_identical(calls$transpose, 1L)
  expect_identical(result$storage, list(marker = "transposed"))
  expect_identical(result$shape, c(3L, 2L))
  expect_equal(as.matrix(to_dgCMatrix(result)), t(source))
  expect_identical(result$compute_stages$sparse_transpose$device, "cuda")
})
