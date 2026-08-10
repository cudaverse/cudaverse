test_that("native diagnostics and factory follow the integrated contract", {
  diagnostics <- cudaverse:::.native_diagnostics()
  factory <- cudaverse:::.native_backend_factory()

  expect_named(
    diagnostics,
    c(
      "installed", "available", "device_count", "version", "reason",
      "detection_error", "driver_version", "cublas_loaded",
      "cusolver_loaded", "cusolver_error", "kernels_loaded",
      "runtime_complete", "self_test", "auto_eligible"
    )
  )
  expect_named(
    diagnostics$self_test,
    c("passed", "reason", "error", "checks", "duration_ms")
  )
  expect_true(diagnostics$installed)
  expect_identical(factory$name, "native")
  expect_identical(factory$device, "cuda")
  expect_identical(factory$contract()$schema, "cudaverse-backend/1")
  expect_true(all(c(
    "contract", "diagnostics", "capabilities", "from_host", "to_host", "cast",
    "reshape", "subset", "replace", "broadcast", "binary", "transpose",
    "matmul", "reduce",
    "sparse_from_coo", "sparse_to_host",
    "sparse_reduce", "sparse_normalize", "sparse_matmul_dense",
    "algorithm_pca_predict", "algorithm_sparse_pca", "algorithm_kmeans",
    "algorithm_sparse_knn_prepare",
    "synchronize", "release", "error_translate"
  ) %in% names(factory)))
  expect_true(all(c(
    "sparse-coo", "sparse-csr", "sparse-normalize", "sparse-matmul",
    "sparse-reduce", "sparse-pca", "sparse-knn", "pca-predict", "kmeans",
    "dtype-float32",
    "dtype-float64", "runtime-self-test", "arithmetic", "reshape",
    "broadcast", "transpose", "subset", "replacement"
  ) %in% factory$capabilities()))
})

test_that("native runtime self-test is cached and releases its allocations", {
  skip_if_not(identical(Sys.getenv("CUDAVERSE_NATIVE_TESTS"), "true"))
  skip_if_not(isTRUE(cudaverse:::.native_diagnostics()$available))
  factory <- cudaverse:::.native_backend_factory()
  factory$synchronize()
  gc()
  baseline <- cudaverse:::.native_memory_tracker(reset = TRUE)$current

  first <- cudaverse:::.native_self_test(reset = TRUE)
  second <- cudaverse:::.native_self_test()
  factory$synchronize()
  gc()
  final <- cudaverse:::.native_memory_tracker()

  expect_true(first$passed)
  expect_identical(second, first)
  expect_true(all(c(
    "float64-transfer-matmul-reduce",
    "float32-transfer-matmul-reduce",
    "arithmetic-reshape-broadcast-transpose",
    "device-indexing",
    "resident-kmeans",
    "sparse-transfer-normalize"
  ) %in% first$checks))
  expect_identical(final$current, baseline)
})

test_that("native tensor surface matches R for both floating dtypes", {
  skip_if_not(identical(Sys.getenv("CUDAVERSE_NATIVE_TESTS"), "true"))
  skip_if_not(isTRUE(cudaverse:::.native_diagnostics()$auto_eligible))
  old <- options(cudaverse.cuda_backends = "native")
  on.exit(options(old), add = TRUE)

  values <- matrix(c(1, 2, 3, 4, 5, 6), 2, 3)
  other <- matrix(c(2, 4, 6, 8, 10, 12), 2, 3)
  vector <- c(2, 3, 4)
  operations <- list(
    "+" = function(x, y) x + y,
    "-" = function(x, y) x - y,
    "*" = function(x, y) x * y,
    "/" = function(x, y) x / y,
    "^" = function(x, y) x^y
  )

  for (dtype in c("float32", "float64")) {
    x <- cudaverse::cuda_tensor(values, dtype = dtype, device = "cuda")
    y <- cudaverse::cuda_tensor(other, dtype = dtype, device = "cuda")
    tolerance <- if (identical(dtype, "float32")) 1e-5 else 1e-10
    for (operation in names(operations)) {
      actual <- operations[[operation]](x, y)
      expected <- operations[[operation]](values, other)
      expect_equal(cudaverse::to_cpu(actual), expected, tolerance = tolerance)
      expect_identical(
        cudaverse::tensor_device(actual),
        c(device = "cuda", backend = "native")
      )
    }

    broadcast <- x + cudaverse::cuda_tensor(
      vector, dtype = dtype, device = "cuda"
    )
    expect_equal(
      cudaverse::to_cpu(broadcast),
      sweep(values, 2L, vector, "+"),
      tolerance = tolerance
    )
    reshaped <- cudaverse::tensor_reshape(x, c(3L, 2L))
    expect_equal(
      cudaverse::to_cpu(reshaped),
      array(values, dim = c(3L, 2L)),
      tolerance = tolerance
    )
    expect_equal(cudaverse::to_cpu(t(x)), t(values), tolerance = tolerance)
  }
})

test_that("native subsetting and replacement keep tensor values on device", {
  skip_if_not(identical(Sys.getenv("CUDAVERSE_NATIVE_TESTS"), "true"))
  skip_if_not(isTRUE(cudaverse:::.native_diagnostics()$available))
  old <- options(cudaverse.cuda_backends = "native")
  on.exit(options(old), add = TRUE)

  for (dtype in c("integer", "float32", "float64")) {
    source <- matrix(
      1:20,
      4,
      5,
      dimnames = list(
        sample = paste0("s", 1:4),
        feature = paste0("f", 1:5)
      )
    )
    x <- cudaverse::cuda_tensor(source, device = "cuda", dtype = dtype)
    selected <- x[c("s4", "s1"), c(5L, 2L), drop = FALSE]
    tolerance <- if (identical(dtype, "float32")) 1e-6 else 0
    expect_equal(
      cudaverse::to_cpu(selected),
      source[c("s4", "s1"), c(5L, 2L), drop = FALSE],
      tolerance = tolerance
    )
    expect_identical(
      cudaverse::tensor_device(selected),
      c(device = "cuda", backend = "native")
    )
    expect_identical(
      cudaverse::cuda_provenance(selected)$stage,
      "subset"
    )
    expect_identical(cudaverse::cuda_provenance(selected)$device, "cuda")

    x[c(1L, 1L, 4L), 3L] <- c(101, 202, 303)
    source[c(1L, 1L, 4L), 3L] <- c(101, 202, 303)
    expect_equal(cudaverse::to_cpu(x), source, tolerance = tolerance)
    expect_identical(cudaverse::cuda_provenance(x)$stage, "replacement")
    expect_identical(cudaverse::cuda_provenance(x)$backend, "native")

    replacement <- cudaverse::cuda_tensor(
      c(404, 505), device = "cuda", dtype = dtype
    )
    x[c(2L, 3L), 4L] <- replacement
    source[c(2L, 3L), 4L] <- c(404, 505)
    expect_equal(cudaverse::to_cpu(x), source, tolerance = tolerance)
    expect_identical(
      cudaverse::cuda_provenance(x)$output_device,
      "cuda"
    )
  }
})

test_that("native indexing errors recover and repeated buffers do not leak", {
  skip_if_not(identical(Sys.getenv("CUDAVERSE_NATIVE_TESTS"), "true"))
  skip_if_not(isTRUE(cudaverse:::.native_diagnostics()$available))
  factory <- cudaverse:::.native_backend_factory()
  source <- factory$from_host(matrix(1:64, 8, 8), "float64", c(8L, 8L))
  replacement <- factory$from_host(c(100, 200), "float64", 2L)
  on.exit(factory$release(source), add = TRUE)
  on.exit(factory$release(replacement), add = TRUE)
  factory$synchronize()
  gc()
  baseline <- cudaverse:::.native_memory_info()$used
  tracked_baseline <- cudaverse:::.native_memory_tracker(reset = TRUE)$current

  condition <- tryCatch(
    cudaverse:::.backend_call("native", "subset", source, 65L, 1L),
    error = identity
  )
  expect_s3_class(condition, "cudaverse_native_error")
  expect_identical(condition$operation, "subset")

  for (iteration in seq_len(1000L)) {
    gathered <- factory$subset(source, c(64L, 1L), 2L)
    replaced <- factory$replace(
      source, c(2L, 63L), replacement, c(1L, 2L)
    )
    factory$release(gathered)
    factory$release(replaced)
  }
  factory$synchronize()
  gc()
  final <- cudaverse:::.native_memory_info()$used
  tracked_final <- cudaverse:::.native_memory_tracker()

  expect_lte(abs(final - baseline), 1024^2)
  expect_identical(tracked_final$current, tracked_baseline)
  expect_equal(factory$to_host(source), 1:64, tolerance = 0)
})

test_that("native reductions match R across dimensions and dtypes", {
  skip_if_not(identical(Sys.getenv("CUDAVERSE_NATIVE_TESTS"), "true"))
  skip_if_not(isTRUE(cudaverse:::.native_diagnostics()$available))
  old <- options(cudaverse.cuda_backends = "native")
  on.exit(options(old), add = TRUE)

  values <- array(seq_len(24) / 7, dim = c(2, 3, 4))
  dimnames(values) <- list(
    sample = c("sample_a", "sample_b"),
    feature = paste0("feature_", 1:3),
    batch = paste0("batch_", 1:4)
  )
  tensor <- cudaverse::cuda_tensor(
    values, device = "cuda", dtype = "float64"
  )
  expect_equal(
    cudaverse::to_cpu(cudaverse::tensor_sum(tensor)),
    array(sum(values), dim = 1L),
    tolerance = 1e-10
  )
  expect_equal(
    cudaverse::to_cpu(cudaverse::tensor_sum(tensor, dim = c(1, 3))),
    array(
      apply(values, 2, sum),
      dim = 3L,
      dimnames = list(feature = paste0("feature_", 1:3))
    ),
    tolerance = 1e-10
  )
  expect_equal(
    cudaverse::to_cpu(cudaverse::tensor_mean(tensor, dim = 2, keepdim = TRUE)),
    array(
      apply(values, c(1, 3), mean),
      dim = c(2, 1, 4),
      dimnames = list(
        sample = c("sample_a", "sample_b"),
        feature = NULL,
        batch = paste0("batch_", 1:4)
      )
    ),
    tolerance = 1e-10
  )
  reduced <- cudaverse::tensor_sum(tensor, dim = 2, keepdim = TRUE)
  expect_identical(
    dimnames(cudaverse::to_cpu(reduced)),
    list(
      sample = c("sample_a", "sample_b"),
      feature = NULL,
      batch = paste0("batch_", 1:4)
    )
  )

  float_values <- matrix(seq_len(20) / 11, 4, 5)
  float_tensor <- cudaverse::cuda_tensor(
    float_values, device = "cuda", dtype = "float32"
  )
  expect_equal(
    cudaverse::to_cpu(cudaverse::tensor_sum(float_tensor, dim = 1)),
    array(colSums(float_values), dim = 5L),
    tolerance = 1e-5
  )
  expect_equal(
    cudaverse::to_cpu(cudaverse::tensor_mean(float_tensor, dim = 2)),
    array(rowMeans(float_values), dim = 4L),
    tolerance = 1e-5
  )

  integer_tensor <- cudaverse::cuda_tensor(
    array(1:12, c(3, 4)), device = "cuda", dtype = "integer"
  )
  integer_sum <- cudaverse::tensor_sum(integer_tensor, dim = 1)
  expect_identical(integer_sum$dtype, "float64")
  expect_equal(
    cudaverse::to_cpu(integer_sum),
    array(colSums(matrix(1:12, 3, 4)), dim = 4L),
    tolerance = 0
  )
})

test_that("native cast and integer reduction kernels preserve values", {
  skip_if_not(identical(Sys.getenv("CUDAVERSE_NATIVE_TESTS"), "true"))
  skip_if_not(isTRUE(cudaverse:::.native_diagnostics()$available))
  factory <- cudaverse:::.native_backend_factory()

  integer_storage <- factory$from_host(1:12, "integer", c(3L, 4L))
  on.exit(factory$release(integer_storage), add = TRUE)
  integer_reduction <- factory$reduce(
    integer_storage, 1L, FALSE, "sum"
  )
  on.exit(factory$release(integer_reduction$storage), add = TRUE)
  expect_identical(
    factory$to_host(integer_reduction$storage),
    as.integer(c(6, 15, 24, 33))
  )

  float_storage <- factory$cast(integer_storage, "float32")
  on.exit(factory$release(float_storage), add = TRUE)
  double_storage <- factory$cast(float_storage, "float64")
  on.exit(factory$release(double_storage), add = TRUE)
  expect_equal(factory$to_host(double_storage), as.numeric(1:12), tolerance = 0)

  non_finite <- factory$from_host(
    c(1, Inf, NaN), "float64", 3L
  )
  on.exit(factory$release(non_finite), add = TRUE)
  non_finite_sum <- factory$reduce(non_finite, NULL, FALSE, "sum")
  on.exit(factory$release(non_finite_sum$storage), add = TRUE)
  expect_true(is.nan(factory$to_host(non_finite_sum$storage)))
})

test_that("native cuSOLVER SVD matches reconstruction for tall and wide data", {
  skip_if_not(identical(Sys.getenv("CUDAVERSE_NATIVE_TESTS"), "true"))
  skip_if_not(isTRUE(cudaverse:::.native_diagnostics()$available))
  skip_if_not(nzchar(Sys.getenv("CUDAVERSE_CUSOLVER_PATH")))
  old <- options(cudaverse.cuda_backends = "native")
  on.exit(options(old), add = TRUE)

  set.seed(20260809)
  for (shape in list(c(12L, 5L), c(5L, 12L), c(5L, 1L))) {
    values <- matrix(rnorm(prod(shape)), nrow = shape[[1L]])
    actual <- cudaverse::cuda_svd(values, device = "cuda")
    expected <- base::svd(values)
    reconstruction <- actual$u %*% diag(actual$d, nrow = length(actual$d)) %*%
      t(actual$v)

    expect_equal(unname(actual$d), expected$d, tolerance = 1e-8)
    expect_equal(reconstruction, values, tolerance = 1e-8)
  }
})

test_that("native PCA matches R subspaces and retains device scores", {
  skip_if_not(identical(Sys.getenv("CUDAVERSE_NATIVE_TESTS"), "true"))
  skip_if_not(isTRUE(cudaverse:::.native_diagnostics()$available))
  skip_if_not(nzchar(Sys.getenv("CUDAVERSE_CUSOLVER_PATH")))
  old <- options(cudaverse.cuda_backends = "native")
  on.exit(options(old), add = TRUE)

  set.seed(2718)
  values <- matrix(rnorm(200), 40, 5)
  for (center in c(FALSE, TRUE)) {
    for (scale in c(FALSE, TRUE)) {
      actual <- cudaverse::cuda_pca(
        values, n_components = 3, center = center, scale. = scale,
        device = "cuda"
      )
      expected <- stats::prcomp(
        values, rank. = 3, center = center, scale. = scale
      )
      actual_reconstruction <- actual$x %*% t(actual$rotation)
      expected_reconstruction <- expected$x[, 1:3, drop = FALSE] %*%
        t(expected$rotation[, 1:3, drop = FALSE])

      expect_equal(unname(actual$sdev), expected$sdev[1:3], tolerance = 1e-8)
      expect_equal(actual_reconstruction, expected_reconstruction,
                   tolerance = 1e-8)
      expect_equal(actual$center, expected$center, tolerance = 1e-10)
      expect_equal(actual$scale, expected$scale, tolerance = 1e-10)
      state <- attr(actual$x, "cudaverse_native_state", exact = TRUE)
      expect_identical(state$backend, "native")
      expect_identical(state$shape, c(40L, 3L))
      expect_type(state$storage, "externalptr")
    }
  }
})

test_that("native distances and stable device top-k match CPU ordering", {
  skip_if_not(identical(Sys.getenv("CUDAVERSE_NATIVE_TESTS"), "true"))
  skip_if_not(isTRUE(cudaverse:::.native_diagnostics()$available))
  old <- options(cudaverse.cuda_backends = "native")
  on.exit(options(old), add = TRUE)

  values <- rbind(
    c(0, 0), c(1, 0), c(-1, 0), c(0, 1), c(0, -1), c(1, 0)
  )
  cpu_distance <- cudaverse::cuda_distance(values, device = "cpu")
  native_distance <- cudaverse::cuda_distance(values, device = "cuda")
  expect_equal(
    as.vector(native_distance), as.vector(cpu_distance), tolerance = 1e-10
  )

  factory <- cudaverse:::.native_backend_factory()
  state <- factory$algorithm_knn_prepare(values)
  native <- factory$algorithm_knn_select(
    state, values, 4L, "euclidean", 2L
  )
  cpu <- cudaverse::cuda_knn(
    values, k = 4, device = "cpu", batch_size = 2
  )
  expect_identical(native$index, cpu$index)
  expect_equal(native$distance, cpu$distance, tolerance = 1e-10)

  cosine_values <- rbind(
    c(1, 0), c(2, 0), c(0, 1), c(0, -1), c(-1, 0), c(1, 1)
  )
  unit_values <- cosine_values / sqrt(rowSums(cosine_values^2))
  cosine_state <- factory$algorithm_knn_prepare(unit_values)
  native_cosine <- factory$algorithm_knn_select(
    cosine_state, unit_values, 3L, "cosine", 3L
  )
  cpu_cosine <- cudaverse::cuda_knn(
    cosine_values, k = 3, metric = "cosine", device = "cpu", batch_size = 3
  )
  expect_identical(native_cosine$index, cpu_cosine$index)
  expect_equal(native_cosine$distance, cpu_cosine$distance, tolerance = 1e-10)

  set.seed(33)
  general_values <- matrix(rnorm(135), 45, 3)
  general_state <- factory$algorithm_knn_prepare(general_values)
  native_general <- factory$algorithm_knn_select(
    general_state, general_values, 40L, "euclidean", 11L
  )
  cpu_general <- cudaverse::cuda_knn(
    general_values, k = 40, device = "cpu", batch_size = 11
  )
  expect_identical(native_general$index, cpu_general$index)
  expect_equal(native_general$distance, cpu_general$distance, tolerance = 1e-10)
})

test_that("native k-means keeps Lloyd updates resident and matches CPU", {
  skip_if_not(identical(Sys.getenv("CUDAVERSE_NATIVE_TESTS"), "true"))
  skip_if_not(isTRUE(cudaverse:::.native_diagnostics()$available))
  old <- options(cudaverse.cuda_backends = "native")
  on.exit(options(old), add = TRUE)

  cases <- list(
    separated = list(
      values = rbind(
        c(0, 0), c(0, 1), c(1, 0),
        c(10, 10), c(10, 11), c(11, 10)
      ),
      centers = rbind(c(0, 0), c(10, 10))
    ),
    tied_empty = list(
      values = rbind(c(0, 0), c(0, 0), c(5, 5), c(10, 10), c(10, 10)),
      centers = rbind(c(0, 0), c(0, 0), c(10, 10))
    ),
    large_offset = list(
      values = rbind(
        c(1e12, 1e12), c(1e12 + 1, 1e12),
        c(1e12 + 100, 1e12 + 100), c(1e12 + 101, 1e12 + 100)
      ),
      centers = rbind(c(1e12, 1e12), c(1e12 + 100, 1e12 + 100))
    )
  )

  for (case in cases) {
    cpu <- cudaverse::cuda_kmeans(
      case$values, centers = case$centers, iter.max = 20L,
      tolerance = 1e-10, device = "cpu"
    )
    native <- cudaverse::cuda_kmeans(
      case$values, centers = case$centers, iter.max = 20L,
      tolerance = 1e-10, device = "cuda"
    )
    expect_identical(native$cluster, cpu$cluster)
    expect_equal(native$centers, cpu$centers, tolerance = 1e-8)
    expect_equal(native$withinss, cpu$withinss, tolerance = 1e-8)
    expect_equal(native$tot.withinss, cpu$tot.withinss, tolerance = 1e-8)
    expect_identical(native$iter, cpu$iter)
    expect_identical(native$converged, cpu$converged)
    expect_identical(native$backend, "native")
    expect_identical(native$compute_stages$distance$output_device, "cuda")
    expect_identical(native$compute_stages$assignment$output_device, "cuda")
    expect_identical(native$compute_stages$center_update$output_device, "cuda")
    expect_identical(native$compute_stages$finalization$output_device, "cpu")
  }
})

test_that("native k-means errors recover and repeated updates do not leak", {
  skip_if_not(identical(Sys.getenv("CUDAVERSE_NATIVE_TESTS"), "true"))
  skip_if_not(isTRUE(cudaverse:::.native_diagnostics()$available))
  old <- options(cudaverse.cuda_backends = "native")
  on.exit(options(old), add = TRUE)
  factory <- cudaverse:::.native_backend_factory()
  values <- matrix(seq_len(40) / 7, 20, 2)
  centers <- values[c(1L, 20L), , drop = FALSE]

  factory$synchronize()
  gc()
  baseline <- cudaverse:::.native_memory_info()$used
  tracked_baseline <- cudaverse:::.native_memory_tracker(reset = TRUE)$current

  condition <- tryCatch(
    cudaverse:::.backend_call(
      "native", "algorithm_kmeans", values, values, 2L, 1e-8
    ),
    error = identity
  )
  expect_s3_class(condition, "cudaverse_native_error")
  expect_identical(condition$operation, "algorithm_kmeans")

  for (iteration in seq_len(1000L)) {
    fit <- factory$algorithm_kmeans(values, centers, 3L, 1e-8)
    expect_identical(length(fit$cluster), nrow(values))
  }
  rm(fit)
  factory$synchronize()
  gc()
  final <- cudaverse:::.native_memory_info()$used
  tracked_final <- cudaverse:::.native_memory_tracker()

  expect_lte(abs(final - baseline), 1024^2)
  expect_identical(tracked_final$current, tracked_baseline)

  probe <- factory$algorithm_kmeans(values, centers, 3L, 1e-8)
  expect_identical(length(probe$cluster), nrow(values))
})

test_that("native algorithm failures are structured and release temporaries", {
  skip_if_not(identical(Sys.getenv("CUDAVERSE_NATIVE_TESTS"), "true"))
  skip_if_not(isTRUE(cudaverse:::.native_diagnostics()$available))
  skip_if_not(nzchar(Sys.getenv("CUDAVERSE_CUSOLVER_PATH")))
  old <- options(cudaverse.cuda_backends = "native")
  on.exit(options(old), add = TRUE)

  warmup <- matrix(rnorm(24), 8, 3)
  cudaverse::cuda_pca(warmup, 2, device = "cuda")
  factory <- cudaverse:::.native_backend_factory()
  factory$synchronize()
  gc()
  baseline <- cudaverse:::.native_memory_info()$used

  constant <- matrix(1, 8, 3)
  condition <- tryCatch(
    cudaverse:::.backend_call(
      "native", "algorithm_pca", constant, 2L, TRUE, TRUE
    ),
    error = identity
  )
  expect_s3_class(condition, "cudaverse_native_error")
  expect_s3_class(condition, "cudaverse_backend_error")
  expect_identical(condition$backend, "native")
  expect_identical(condition$operation, "algorithm_pca")

  for (iteration in seq_len(100L)) {
    try(
      cudaverse:::.backend_call(
        "native", "algorithm_pca", constant, 2L, TRUE, TRUE
      ),
      silent = TRUE
    )
  }
  factory$synchronize()
  gc()
  final <- cudaverse:::.native_memory_info()$used
  expect_lte(abs(final - baseline), 1024^2)
})

test_that("injected CUDA failures unwind temporary allocations", {
  skip_if_not(identical(Sys.getenv("CUDAVERSE_NATIVE_TESTS"), "true"))
  skip_if_not(isTRUE(cudaverse:::.native_diagnostics()$available))
  factory <- cudaverse:::.native_backend_factory()
  factory$synchronize()
  gc()
  baseline <- cudaverse:::.native_memory_tracker(reset = TRUE)$current

  condition <- tryCatch(
    cudaverse:::.backend_call(
      "native", "test_inject_cuda_error", 4096L
    ),
    error = identity
  )
  expect_s3_class(condition, "cudaverse_native_error")
  expect_s3_class(condition, "cudaverse_backend_error")
  expect_identical(condition$backend, "native")
  expect_identical(condition$operation, "test_inject_cuda_error")
  expect_match(conditionMessage(condition), "OUT_OF_MEMORY|out of memory")

  for (iteration in seq_len(100L)) {
    try(
      cudaverse:::.backend_call(
        "native", "test_inject_cuda_error", 4096L
      ),
      silent = TRUE
    )
  }
  factory$synchronize()
  gc()
  final <- cudaverse:::.native_memory_tracker()
  expect_identical(final$current, baseline)
  expect_gte(final$peak - baseline, 4096)
})

test_that("repeated dense pipelines release all operation-owned VRAM", {
  skip_if_not(identical(Sys.getenv("CUDAVERSE_NATIVE_TESTS"), "true"))
  skip_if_not(isTRUE(cudaverse:::.native_diagnostics()$available))
  skip_if_not(nzchar(Sys.getenv("CUDAVERSE_CUSOLVER_PATH")))
  factory <- cudaverse:::.native_backend_factory()
  values <- matrix(rnorm(400), 50, 8)

  warmup <- factory$algorithm_pca(values, 4L, TRUE, FALSE)
  state <- factory$algorithm_knn_prepare(warmup$x)
  factory$algorithm_knn_select(state, warmup$x, 5L, "euclidean", 13L)
  rm(warmup, state)
  factory$synchronize()
  gc()
  baseline <- cudaverse:::.native_memory_info()$used

  for (iteration in seq_len(100L)) {
    fit <- factory$algorithm_pca(values, 4L, TRUE, FALSE)
    state <- factory$algorithm_knn_prepare(fit$x)
    result <- factory$algorithm_knn_select(
      state, fit$x, 5L, "euclidean", 13L
    )
    expect_identical(dim(result$index), c(50L, 5L))
  }
  rm(fit, state, result)
  factory$synchronize()
  gc()
  final <- cudaverse:::.native_memory_info()$used
  expect_lte(abs(final - baseline), 1024^2)
})

test_that("R time-limit interruption leaves the native backend reusable", {
  skip_if_not(identical(Sys.getenv("CUDAVERSE_NATIVE_TESTS"), "true"))
  skip_if_not(isTRUE(cudaverse:::.native_diagnostics()$available))
  factory <- cudaverse:::.native_backend_factory()
  values <- matrix(rnorm(40000), 2000, 20)
  factory$synchronize()
  gc()
  baseline <- cudaverse:::.native_memory_info()$used

  state <- factory$algorithm_knn_prepare(values)
  on.exit(setTimeLimit(cpu = Inf, elapsed = Inf, transient = FALSE), add = TRUE)
  setTimeLimit(elapsed = 0.02, transient = TRUE)
  condition <- tryCatch(
    factory$algorithm_knn_select(
      state, values, 15L, "euclidean", 1L
    ),
    error = identity
  )
  setTimeLimit(cpu = Inf, elapsed = Inf, transient = FALSE)
  expect_s3_class(condition, "error")
  expect_match(conditionMessage(condition), "time limit|elapsed")

  factory$synchronize()
  probe <- factory$from_host(matrix(1:4, 2), "float64", c(2L, 2L))
  expect_equal(factory$to_host(probe), as.numeric(1:4), tolerance = 0)
  factory$release(probe)
  gc()
  final <- cudaverse:::.native_memory_info()$used
  expect_lte(abs(final - baseline), 1024^2)
})
test_that("native errors translate into structured cudaverse conditions", {
  factory <- cudaverse:::.native_backend_factory()
  raw <- simpleError("injected CUDA failure")
  condition <- factory$error_translate(raw, "matmul")

  expect_s3_class(condition, "cudaverse_native_error")
  expect_s3_class(condition, "cudaverse_backend_error")
  expect_identical(condition$backend, "native")
  expect_identical(condition$operation, "matmul")
  expect_identical(condition$parent, raw)
})

test_that("native float32 and float64 transfer and cuBLAS matmul match base R", {
  skip_if_not(identical(Sys.getenv("CUDAVERSE_NATIVE_TESTS"), "true"))
  skip_if_not(isTRUE(cudaverse:::.native_diagnostics()$available))
  old <- options(cudaverse.cuda_backends = "native")
  on.exit(options(old), add = TRUE)

  left <- matrix(seq_len(15) / 7, 5, 3)
  right <- matrix(seq_len(12) / 11, 3, 4)
  for (dtype in c("float32", "float64")) {
    actual <- cudaverse::tensor_matmul(
      cudaverse::cuda_tensor(left, device = "cuda", dtype = dtype),
      cudaverse::cuda_tensor(right, device = "cuda", dtype = dtype)
    )

    expect_identical(
      cudaverse::tensor_device(actual),
      c(device = "cuda", backend = "native")
    )
    tolerance <- if (identical(dtype, "float32")) 1e-5 else 1e-10
    expect_equal(
      cudaverse::to_cpu(actual),
      left %*% right,
      tolerance = tolerance
    )
    provenance <- cudaverse::cuda_provenance(actual)
    expect_identical(provenance$backend, "native")
    expect_identical(provenance$output_device, "cuda")
  }
})

test_that("native PCA prediction remains compatible with automatic selection", {
  skip_if_not(identical(Sys.getenv("CUDAVERSE_NATIVE_TESTS"), "true"))
  skip_if_not(nzchar(Sys.getenv("CUDAVERSE_CUSOLVER_PATH")))
  skip_if_not(isTRUE(cudaverse:::.native_diagnostics()$auto_eligible))
  old <- options(cudaverse.cuda_backends = NULL)
  on.exit(options(old), add = TRUE)

  training <- matrix(seq_len(48) / 11, 12, 4)
  prediction <- training[c(2L, 7L, 11L), , drop = FALSE]
  fit <- cudaverse::cuda_pca(
    training, n_components = 3L, center = TRUE, scale. = TRUE,
    device = "auto"
  )
  scores <- predict(fit, prediction, device = "auto")
  transformed <- sweep(prediction, 2L, fit$center, "-")
  transformed <- sweep(transformed, 2L, fit$scale, "/")
  reference <- transformed %*% fit$rotation

  expect_equal(as.vector(scores), as.vector(reference), tolerance = 1e-8)
  expect_identical(attr(scores, "device", exact = TRUE), "cuda")
  expect_true(all(
    cudaverse::cuda_provenance(scores)$backend == "native"
  ))
})

test_that("shared native ownership frees an allocation exactly once", {
  skip_if_not(identical(Sys.getenv("CUDAVERSE_NATIVE_TESTS"), "true"))
  skip_if_not(isTRUE(cudaverse:::.native_diagnostics()$available))
  factory <- cudaverse:::.native_backend_factory()

  original <- factory$from_host(as.double(1:8), "float64", 8L)
  shared <- cudaverse:::.native_share(original)
  factory$release(original)
  expect_equal(factory$to_host(shared), as.double(1:8))
  expect_true(factory$release(shared))
  expect_error(factory$to_host(shared), "released")
})

test_that("operation-owned allocation telemetry reports a high-water mark", {
  skip_if_not(identical(Sys.getenv("CUDAVERSE_NATIVE_TESTS"), "true"))
  skip_if_not(isTRUE(cudaverse:::.native_diagnostics()$available))
  factory <- cudaverse:::.native_backend_factory()
  gc()
  baseline <- cudaverse:::.native_memory_tracker(reset = TRUE)$current

  first <- factory$from_host(double(128), "float64", 128L)
  second <- factory$from_host(double(256), "float64", 256L)
  during <- cudaverse:::.native_memory_tracker()
  expect_identical(during$current - baseline, 384 * 8)
  expect_gte(during$peak - baseline, 384 * 8)

  factory$release(first)
  factory$release(second)
  factory$synchronize()
  gc()
  final <- cudaverse:::.native_memory_tracker()
  expect_identical(final$current, baseline)
  expect_gte(final$peak - baseline, 384 * 8)
})

test_that("one thousand allocate-transfer-free cycles do not leak VRAM", {
  skip_if_not(identical(Sys.getenv("CUDAVERSE_NATIVE_TESTS"), "true"))
  skip_if_not(isTRUE(cudaverse:::.native_diagnostics()$available))
  factory <- cudaverse:::.native_backend_factory()
  factory$synchronize()
  gc()
  baseline <- cudaverse:::.native_memory_info()$used
  tracked_baseline <- cudaverse:::.native_memory_tracker(reset = TRUE)$current

  values <- matrix(seq_len(64) / 13, 8, 8)
  for (iteration in seq_len(1000L)) {
    pointer <- factory$from_host(values, "float64", dim(values))
    expect_equal(factory$to_host(pointer), as.vector(values), tolerance = 0)
    factory$release(pointer)
  }
  factory$synchronize()
  gc()
  final <- cudaverse:::.native_memory_info()$used
  tracked_final <- cudaverse:::.native_memory_tracker()

  expect_lte(abs(final - baseline), 1024^2)
  expect_identical(tracked_final$current, tracked_baseline)
  expect_gte(tracked_final$peak - tracked_baseline, length(values) * 8)
})

test_that("native COO and CSR storage round-trip with shared ownership", {
  skip_if_not(identical(Sys.getenv("CUDAVERSE_NATIVE_TESTS"), "true"))
  skip_if_not(isTRUE(cudaverse:::.native_diagnostics()$available))
  old <- options(cudaverse.cuda_backends = "native")
  on.exit(options(old), add = TRUE)
  factory <- cudaverse:::.native_backend_factory()
  source <- Matrix::sparseMatrix(
    i = c(1L, 1L, 2L, 4L),
    j = c(1L, 3L, 2L, 3L),
    x = c(2, 1, 4, 5),
    dims = c(4L, 3L),
    dimnames = list(paste0("r", 1:4), paste0("c", 1:3))
  )
  csr <- cudaverse::cuda_sparse(source, format = "csr", device = "cuda")
  expect_type(csr$storage, "externalptr")
  expect_equal(
    as.matrix(cudaverse::to_dgCMatrix(csr)),
    as.matrix(source),
    tolerance = 0
  )

  coo <- cudaverse::as_coo(csr)
  expect_type(coo$storage, "externalptr")
  factory$sparse_release(csr$storage)
  expect_equal(
    as.matrix(cudaverse::to_dgCMatrix(coo)),
    as.matrix(source),
    tolerance = 0
  )
  expect_true(factory$sparse_release(coo$storage))
  expect_error(factory$sparse_to_host(coo$storage), "released")
})

test_that("native sparse matmul, matvec, and reductions match Matrix", {
  skip_if_not(identical(Sys.getenv("CUDAVERSE_NATIVE_TESTS"), "true"))
  skip_if_not(isTRUE(cudaverse:::.native_diagnostics()$available))
  old <- options(cudaverse.cuda_backends = "native")
  on.exit(options(old), add = TRUE)
  set.seed(3001)
  source <- Matrix::rsparsematrix(37, 19, density = 0.18)
  dense <- matrix(stats::rnorm(19 * 7), 19, 7)
  sparse <- cudaverse::cuda_sparse(source, device = "cuda")

  product <- cudaverse::sparse_matmul_dense(sparse, dense)
  expect_identical(
    cudaverse::tensor_device(product),
    c(device = "cuda", backend = "native")
  )
  expect_equal(
    cudaverse::to_cpu(product),
    as.matrix(source %*% dense),
    tolerance = 1e-10
  )
  vector <- stats::rnorm(ncol(source))
  expect_equal(
    as.numeric(cudaverse::sparse_matvec(sparse, vector)),
    as.vector(source %*% vector),
    tolerance = 1e-10
  )
  expect_equal(
    as.numeric(cudaverse::sparse_row_sums(sparse)),
    as.numeric(Matrix::rowSums(source)),
    tolerance = 1e-10
  )
  expect_equal(
    as.numeric(cudaverse::sparse_col_sums(sparse)),
    as.numeric(Matrix::colSums(source)),
    tolerance = 1e-10
  )
  expect_identical(
    cudaverse::cuda_provenance(product)$output_device,
    "cuda"
  )
  expect_identical(
    cudaverse::cuda_provenance(cudaverse::sparse_row_sums(sparse))$backend,
    "native"
  )
})

test_that("native sparse normalization feeds resident PCA and stable kNN", {
  skip_if_not(identical(Sys.getenv("CUDAVERSE_NATIVE_TESTS"), "true"))
  skip_if_not(isTRUE(cudaverse:::.native_diagnostics()$available))
  old <- options(cudaverse.cuda_backends = "native")
  on.exit(options(old), add = TRUE)
  set.seed(3002)
  values <- matrix(stats::rpois(96 * 24, lambda = 2), 96, 24)
  values[, 1L] <- values[, 1L] + 1
  values[rowSums(values) == 0, 1L] <- 1
  rownames(values) <- paste0("sample_", seq_len(nrow(values)))
  colnames(values) <- paste0("feature_", seq_len(ncol(values)))
  source <- Matrix::Matrix(values, sparse = TRUE)
  sparse <- cudaverse::cuda_sparse(source, device = "cuda")
  normalized <- cudaverse::sparse_normalize(
    sparse, margin = "rows", scale_factor = 1000, log1p = TRUE
  )
  dense <- log1p(values * 1000 / rowSums(values))
  expect_equal(
    as.matrix(cudaverse::to_dgCMatrix(normalized)),
    dense,
    tolerance = 1e-10
  )

  native_pca <- cudaverse::cuda_pca(
    normalized, 8L, center = TRUE, scale. = FALSE, device = "cuda"
  )
  cpu_pca <- cudaverse::cuda_pca(
    dense, 8L, center = TRUE, scale. = FALSE, device = "cpu"
  )
  expect_equal(
    tcrossprod(native_pca$rotation),
    tcrossprod(cpu_pca$rotation),
    tolerance = 1e-8
  )
  state <- attr(native_pca$x, "cudaverse_native_state", exact = TRUE)
  expect_type(state$storage, "externalptr")
  pca_stages <- cudaverse::cuda_provenance(native_pca)
  expect_true(all(c(
    "normalization", "sparse_to_dense", "preprocessing",
    "decomposition", "scores_resident"
  ) %in% pca_stages$stage))
  expect_identical(attr(pca_stages, "schema", exact = TRUE),
                   "cudaverse-stage/1")

  native_knn <- cudaverse::cuda_knn(
    native_pca$x, k = 15L, device = "cuda", batch_size = 31L
  )
  cpu_knn <- cudaverse::cuda_knn(
    native_pca$x, k = 15L, device = "cpu", batch_size = 31L
  )
  expect_identical(native_knn$index, cpu_knn$index)
  expect_equal(native_knn$distance, cpu_knn$distance, tolerance = 1e-8)
  expect_identical(
    cudaverse::cuda_provenance(native_knn)$stage,
    c("input_materialization", "distance", "neighbor_selection")
  )

  direct_native <- cudaverse::cuda_knn(
    normalized, k = 7L, device = "cuda", batch_size = 23L
  )
  direct_cpu <- cudaverse::cuda_knn(
    dense, k = 7L, device = "cpu", batch_size = 23L
  )
  expect_identical(direct_native$index, direct_cpu$index)
  expect_equal(direct_native$distance, direct_cpu$distance, tolerance = 1e-8)
  direct_stages <- cudaverse::cuda_provenance(direct_native)
  expect_true(all(c(
    "normalization", "sparse_to_dense", "distance", "neighbor_selection"
  ) %in% direct_stages$stage))
})

test_that("native sparse failures are structured and backend remains reusable", {
  skip_if_not(identical(Sys.getenv("CUDAVERSE_NATIVE_TESTS"), "true"))
  skip_if_not(isTRUE(cudaverse:::.native_diagnostics()$available))
  old <- options(cudaverse.cuda_backends = "native")
  on.exit(options(old), add = TRUE)
  source <- Matrix::Diagonal(4, x = 1:4)
  sparse <- cudaverse::cuda_sparse(source, device = "cuda")
  broken <- sparse
  broken$shape <- c(4L, 3L)
  condition <- tryCatch(
    cudaverse::sparse_matmul_dense(broken, matrix(1, 3, 2)),
    error = identity
  )
  expect_s3_class(condition, "cudaverse_native_error")
  expect_identical(condition$backend, "native")
  expect_identical(condition$operation, "sparse_matmul_dense")

  expect_equal(
    as.numeric(cudaverse::sparse_row_sums(sparse)),
    1:4,
    tolerance = 0
  )
})

test_that("one thousand sparse allocate-normalize-free cycles do not leak", {
  skip_if_not(identical(Sys.getenv("CUDAVERSE_NATIVE_TESTS"), "true"))
  skip_if_not(isTRUE(cudaverse:::.native_diagnostics()$available))
  factory <- cudaverse:::.native_backend_factory()
  factory$synchronize()
  gc()
  baseline <- cudaverse:::.native_memory_info()$used
  tracked_baseline <- cudaverse:::.native_memory_tracker(reset = TRUE)$current
  i <- rep(1:16, each = 3L)
  j <- rep(c(1L, 4L, 8L), 16L)
  values <- rep(c(1, 2, 3), 16L)

  for (iteration in seq_len(1000L)) {
    storage <- factory$sparse_from_coo(
      i, j, values, c(16L, 8L), "csr"
    )
    normalized <- factory$sparse_normalize(
      storage, 0L, 100, TRUE
    )
    factory$sparse_release(normalized$storage)
    factory$sparse_release(storage)
  }
  factory$synchronize()
  gc()
  final <- cudaverse:::.native_memory_info()$used
  tracked_final <- cudaverse:::.native_memory_tracker()
  expect_lte(abs(final - baseline), 1024^2)
  expect_identical(tracked_final$current, tracked_baseline)
  expect_gt(tracked_final$peak, tracked_baseline)
})

test_that("R time-limit interruption leaves sparse native state reusable", {
  skip_if_not(identical(Sys.getenv("CUDAVERSE_NATIVE_TESTS"), "true"))
  skip_if_not(isTRUE(cudaverse:::.native_diagnostics()$available))
  old <- options(cudaverse.cuda_backends = "native")
  on.exit(options(old), add = TRUE)
  factory <- cudaverse:::.native_backend_factory()
  source <- Matrix::rsparsematrix(512, 256, density = 0.05)
  sparse <- cudaverse::cuda_sparse(abs(source), device = "cuda")
  dense <- matrix(stats::rnorm(256 * 64), 256, 64)
  factory$synchronize()
  gc()
  tracked_baseline <- cudaverse:::.native_memory_tracker(reset = TRUE)$current

  condition <- tryCatch({
    setTimeLimit(elapsed = 0.01, transient = TRUE)
    repeat {
      product <- cudaverse::sparse_matmul_dense(sparse, dense)
      cudaverse::to_cpu(product)
    }
  }, error = identity, finally = setTimeLimit(cpu = Inf, elapsed = Inf,
                                               transient = FALSE))
  expect_s3_class(condition, "error")
  if (exists("product", inherits = FALSE)) rm(product)
  gc()
  factory$synchronize()
  expect_identical(
    cudaverse:::.native_memory_tracker()$current,
    tracked_baseline
  )
  expect_equal(
    as.numeric(cudaverse::sparse_row_sums(sparse)),
    as.numeric(Matrix::rowSums(abs(source))),
    tolerance = 1e-10
  )
})


