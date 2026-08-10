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
  expect_true(all(c(
    "capabilities", "operations"
  ) %in% names(diagnostics$backend_diagnostics$torch)))
  expect_true("algorithm_pca_predict" %in%
                diagnostics$backend_diagnostics$torch$operations)
})

test_that("capabilities and callable operations have distinct contracts", {
  base <- cudaverse:::.backend_get("base")

  expect_true(all(c(
    "svd", "pca", "pca-predict", "distance", "knn"
  ) %in% base$capabilities()))
  expect_true(all(c(
    "algorithm_svd", "algorithm_pca", "algorithm_pca_predict",
    "algorithm_distance", "algorithm_knn_prepare", "algorithm_knn_block"
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
