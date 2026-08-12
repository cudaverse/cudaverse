.native_kernel_state <- new.env(parent = emptyenv())
.native_kernel_state$loaded <- FALSE
.native_self_test_state <- new.env(parent = emptyenv())
.native_self_test_state$result <- NULL

.native_kernel_path <- function() {
  system.file(
    "kernels",
    "cudaverse_dense_kernels.ptx",
    package = "cudaverse",
    mustWork = FALSE
  )
}

.native_ensure_kernels <- function() {
  if (isTRUE(.native_kernel_state$loaded)) return(invisible(TRUE))
  path <- .native_kernel_path()
  if (!nzchar(path) || !file.exists(path)) {
    stop(
      "The CUDA 12.8.1 dense-kernel PTX artifact is missing.",
      call. = FALSE
    )
  }
  .Call(C_cudaverse_cuda_load_kernels, normalizePath(path, mustWork = TRUE))
  .native_kernel_state$loaded <- TRUE
  invisible(TRUE)
}

.native_self_test <- function(reset = FALSE) {
  if (isTRUE(reset)) .native_self_test_state$result <- NULL
  if (!is.null(.native_self_test_state$result)) {
    return(.native_self_test_state$result)
  }

  started <- unname(proc.time()[["elapsed"]])
  dense_resources <- list()
  sparse_resources <- list()
  keep_dense <- function(storage) {
    dense_resources[[length(dense_resources) + 1L]] <<- storage
    storage
  }
  keep_sparse <- function(storage) {
    sparse_resources[[length(sparse_resources) + 1L]] <<- storage
    storage
  }
  on.exit({
    for (storage in rev(dense_resources)) {
      tryCatch(.native_release(storage), error = function(error) NULL)
    }
    for (storage in rev(sparse_resources)) {
      tryCatch(.native_sparse_release(storage), error = function(error) NULL)
    }
  }, add = TRUE)

  checks <- character()
  result <- tryCatch({
    .native_ensure_kernels()
    values <- matrix(c(1, 2, 3, 4), nrow = 2L)
    identity <- diag(2)

    left64 <- keep_dense(.native_from_host(values, "float64", c(2L, 2L)))
    right64 <- keep_dense(.native_from_host(identity, "float64", c(2L, 2L)))
    product64 <- keep_dense(.native_matmul(left64, right64))
    actual64 <- matrix(.native_to_host(product64), nrow = 2L)
    if (!isTRUE(all.equal(actual64, values, tolerance = 1e-12))) {
      stop("float64 transfer/matmul parity failed", call. = FALSE)
    }
    sum64 <- .native_reduce(product64, NULL, FALSE, "sum")
    keep_dense(sum64$storage)
    if (!isTRUE(all.equal(.native_to_host(sum64$storage), 10,
                          tolerance = 1e-12))) {
      stop("float64 reduction parity failed", call. = FALSE)
    }
    checks <- c(checks, "float64-transfer-matmul-reduce")

    left32 <- keep_dense(.native_from_host(values, "float32", c(2L, 2L)))
    right32 <- keep_dense(.native_from_host(identity, "float32", c(2L, 2L)))
    product32 <- keep_dense(.native_matmul(left32, right32))
    actual32 <- matrix(.native_to_host(product32), nrow = 2L)
    if (!isTRUE(all.equal(actual32, values, tolerance = 1e-6))) {
      stop("float32 transfer/matmul parity failed", call. = FALSE)
    }
    sum32 <- .native_reduce(product32, NULL, FALSE, "sum")
    keep_dense(sum32$storage)
    if (!isTRUE(all.equal(.native_to_host(sum32$storage), 10,
                          tolerance = 1e-5))) {
      stop("float32 reduction parity failed", call. = FALSE)
    }
    checks <- c(checks, "float32-transfer-matmul-reduce")

    binary32 <- keep_dense(.native_binary(product32, product32, "+"))
    if (!isTRUE(all.equal(
      matrix(.native_to_host(binary32), nrow = 2L),
      values * 2,
      tolerance = 1e-5
    ))) {
      stop("float32 arithmetic parity failed", call. = FALSE)
    }
    reshaped32 <- keep_dense(.native_reshape(
      product32, c(2L, 2L), 4L
    ))
    if (!isTRUE(all.equal(
      .native_to_host(reshaped32),
      as.vector(values),
      tolerance = 1e-6
    ))) {
      stop("float32 reshape parity failed", call. = FALSE)
    }
    transposed32 <- keep_dense(.native_transpose(product32))
    if (!isTRUE(all.equal(
      matrix(.native_to_host(transposed32), nrow = 2L),
      t(values),
      tolerance = 1e-6
    ))) {
      stop("float32 transpose parity failed", call. = FALSE)
    }
    vector32 <- keep_dense(.native_from_host(c(5, 7), "float32", 2L))
    broadcast32 <- keep_dense(.native_broadcast(
      vector32, 2L, c(2L, 2L)
    ))
    if (!isTRUE(all.equal(
      matrix(.native_to_host(broadcast32), nrow = 2L),
      matrix(c(5, 5, 7, 7), nrow = 2L),
      tolerance = 1e-6
    ))) {
      stop("float32 broadcast parity failed", call. = FALSE)
    }
    checks <- c(checks, "arithmetic-reshape-broadcast-transpose")

    gathered64 <- keep_dense(.native_subset(product64, c(4L, 1L), 2L))
    if (!isTRUE(all.equal(
      .native_to_host(gathered64), c(4, 1), tolerance = 1e-12
    ))) {
      stop("device gather parity failed", call. = FALSE)
    }
    replaced64 <- keep_dense(.native_replace(
      product64, c(2L, 4L), left64, c(1L, 2L)
    ))
    if (!isTRUE(all.equal(
      matrix(.native_to_host(replaced64), nrow = 2L),
      matrix(c(1, 1, 3, 2), nrow = 2L),
      tolerance = 1e-12
    ))) {
      stop("device replacement parity failed", call. = FALSE)
    }
    checks <- c(checks, "device-indexing")

    validation <- .native_algorithm_matrix_validate(left64, FALSE)
    if (!identical(validation$finite, TRUE) ||
        !identical(validation$constant, FALSE)) {
      stop("resident matrix validation failed", call. = FALSE)
    }
    resident_svd <- .native_algorithm_svd_storage(
      left64, c(2L, 2L), "float64", 2L, 2L
    )
    reconstructed <- resident_svd$u %*%
      diag(resident_svd$d, nrow = 2L) %*% t(resident_svd$v)
    if (!isTRUE(all.equal(reconstructed, values, tolerance = 1e-10))) {
      stop("resident SVD parity failed", call. = FALSE)
    }
    pca_values <- rbind(c(0, 0), c(0, 1), c(1, 0), c(1, 1))
    pca_storage <- keep_dense(.native_from_host(
      pca_values, "float64", c(4L, 2L)
    ))
    resident_pca <- .native_algorithm_pca_storage(
      pca_storage, c(4L, 2L), "float64", 2L, TRUE, FALSE
    )
    pca_state <- attr(resident_pca$x, "cudaverse_native_state", exact = TRUE)
    keep_dense(pca_state$storage)
    if (!identical(dim(resident_pca$x), c(4L, 2L)) ||
        !identical(dim(resident_pca$rotation), c(2L, 2L)) ||
        any(!is.finite(resident_pca$x))) {
      stop("resident PCA parity failed", call. = FALSE)
    }
    checks <- c(checks, "resident-matrix-validation-svd-pca")

    distance_values <- rbind(c(0, 0), c(1, 0), c(0, 1), c(1, 1))
    distance <- .native_algorithm_distance_batched(
      distance_values, distance_values, "euclidean", 2L
    )
    if (!isTRUE(all.equal(
      distance,
      as.matrix(stats::dist(distance_values)),
      tolerance = 1e-10,
      check.attributes = FALSE
    ))) {
      stop("batched distance parity failed", call. = FALSE)
    }
    checks <- c(checks, "resident-batched-distance")

    kmeans_values <- rbind(c(0, 0), c(0, 1), c(10, 10), c(10, 11))
    kmeans <- .native_algorithm_kmeans(
      kmeans_values,
      kmeans_values[c(1L, 3L), , drop = FALSE],
      10L,
      1e-8
    )
    if (!identical(kmeans$cluster, c(1L, 1L, 2L, 2L)) ||
        !isTRUE(all.equal(
          kmeans$centers,
          rbind(c(0, 0.5), c(10, 10.5)),
          tolerance = 1e-10
        ))) {
      stop("resident k-means parity failed", call. = FALSE)
    }
    checks <- c(checks, "resident-kmeans")

    sparse <- keep_sparse(.native_sparse_from_coo(
      c(1L, 2L), c(2L, 3L), c(2, 4), c(2L, 3L), "csr"
    ))
    sparse_transpose <- keep_sparse(.native_sparse_transpose(sparse))
    transpose_host <- .native_sparse_to_host(sparse_transpose)
    transpose_order <- order(transpose_host$i, transpose_host$j)
    if (!identical(transpose_host$shape, c(3L, 2L)) ||
        !identical(transpose_host$i[transpose_order], c(2L, 3L)) ||
        !identical(transpose_host$j[transpose_order], c(1L, 2L)) ||
        !isTRUE(all.equal(
          transpose_host$values[transpose_order], c(2, 4), tolerance = 0
        ))) {
      stop("sparse transpose parity failed", call. = FALSE)
    }
    checks <- c(checks, "sparse-transpose")

    normalized <- .native_sparse_normalize(sparse, 0L, 1, FALSE)
    keep_sparse(normalized$storage)
    normalized_host <- .native_sparse_to_host(normalized$storage)
    if (!isTRUE(all.equal(normalized_host$values, c(1, 1),
                          tolerance = 1e-12))) {
      stop("sparse normalization parity failed", call. = FALSE)
    }
    checks <- c(checks, "sparse-transfer-normalize")

    .native_synchronize()
    list(
      passed = TRUE,
      reason = "self_test_passed",
      error = NULL,
      checks = checks
    )
  }, error = function(error) {
    list(
      passed = FALSE,
      reason = "self_test_failed",
      error = conditionMessage(error),
      checks = checks
    )
  })
  result$duration_ms <- max(
    0,
    (unname(proc.time()[["elapsed"]]) - started) * 1000
  )
  .native_self_test_state$result <- result
  result
}

.native_diagnostics <- function() {
  diagnostics <- .Call(C_cudaverse_cuda_diagnostics)
  kernel_error <- NULL
  if (isTRUE(diagnostics$available)) {
    tryCatch(
      .native_ensure_kernels(),
      error = function(error) kernel_error <<- conditionMessage(error)
    )
  }
  diagnostics$kernels_loaded <- isTRUE(.native_kernel_state$loaded)
  if (!is.null(kernel_error)) {
    diagnostics$available <- FALSE
    diagnostics$reason <- "kernel_unavailable"
    diagnostics$detection_error <- kernel_error
  }
  diagnostics$runtime_complete <- isTRUE(diagnostics$available) &&
    isTRUE(diagnostics$cublas_loaded) &&
    isTRUE(diagnostics$cusolver_loaded) &&
    isTRUE(diagnostics$kernels_loaded)
  diagnostics$self_test <- if (isTRUE(diagnostics$available) &&
                               isTRUE(diagnostics$kernels_loaded)) {
    .native_self_test()
  } else {
    list(
      passed = FALSE,
      reason = "backend_unavailable",
      error = diagnostics$detection_error,
      checks = character(),
      duration_ms = 0
    )
  }
  diagnostics$auto_eligible <- diagnostics$runtime_complete &&
    isTRUE(diagnostics$self_test$passed)
  diagnostics
}

.native_capabilities <- function() {
  c(
    "driver-detection",
    "allocation",
    "transfer",
    "cast",
    "matmul",
    "reduce",
    "arithmetic",
    "reshape",
    "broadcast",
    "transpose",
    "subset",
    "replacement",
    "matrix-validation",
    "svd",
    "svd-resident",
    "pca",
    "pca-resident",
    "pca-predict",
    "distance",
    "distance-batched",
    "kmeans",
    "kmeans-batched",
    "knn",
    "stable-topk",
    "sparse",
    "sparse-coo",
    "sparse-csr",
    "sparse-transpose",
    "sparse-normalize",
    "sparse-matmul",
    "sparse-reduce",
    "sparse-pca",
    "sparse-knn",
    "synchronize",
    "shared-ownership",
    "memory-observability",
    "dtype-float32",
    "dtype-float64",
    "runtime-self-test"
  )
}

.native_contract <- function() {
  list(
    schema = "cudaverse-backend/1",
    provider = "cudaverse",
    capabilities = .native_capabilities()
  )
}

.native_from_host <- function(x, dtype, shape, dimnames = NULL) {
  .Call(
    C_cudaverse_cuda_from_host,
    x,
    as.character(dtype),
    as.integer(shape)
  )
}

.native_to_host <- function(storage) {
  .Call(C_cudaverse_cuda_to_host, storage)
}

.native_sparse_from_coo <- function(i, j, values, shape, format = "csr") {
  .native_ensure_kernels()
  .Call(
    C_cudaverse_cuda_sparse_from_coo,
    as.integer(i),
    as.integer(j),
    as.numeric(values),
    as.integer(shape),
    as.character(format)
  )
}

.native_sparse_to_host <- function(storage) {
  .Call(C_cudaverse_cuda_sparse_to_host, storage)
}

.native_sparse_transpose <- function(storage) {
  .native_ensure_kernels()
  .Call(C_cudaverse_cuda_sparse_transpose, storage)
}

.native_sparse_reduce <- function(storage, margin) {
  .native_ensure_kernels()
  .Call(C_cudaverse_cuda_sparse_reduce, storage, as.integer(margin))
}

.native_sparse_normalize <- function(storage, margin, scale_factor, log1p) {
  .native_ensure_kernels()
  .Call(
    C_cudaverse_cuda_sparse_normalize,
    storage,
    as.integer(margin),
    as.numeric(scale_factor),
    as.logical(log1p)
  )
}

.native_sparse_matmul_dense <- function(storage, i, j, values, shape, dense,
                                        dense_storage = NULL,
                                        dense_shape = dim(dense)) {
  .native_ensure_kernels()
  owned <- is.null(dense_storage)
  if (owned) {
    dense_storage <- .native_from_host(dense, "float64", dense_shape)
    on.exit(.native_release(dense_storage), add = TRUE)
  }
  result <- .Call(
    C_cudaverse_cuda_sparse_matmul_dense,
    storage,
    dense_storage
  )
  list(
    storage = result,
    shape = as.integer(c(shape[[1L]], dense_shape[[2L]])),
    dtype = "float64",
    device_resident = TRUE
  )
}

.native_sparse_to_dense <- function(storage) {
  .native_ensure_kernels()
  .Call(C_cudaverse_cuda_sparse_to_dense, storage)
}

.native_cast <- function(storage, dtype) {
  .native_ensure_kernels()
  .Call(C_cudaverse_cuda_cast, storage, as.character(dtype))
}

.native_reshape <- function(storage, source_shape, target_shape) {
  .Call(C_cudaverse_cuda_reshape, storage, as.integer(target_shape))
}

.native_subset <- function(storage, indices, shape) {
  .native_ensure_kernels()
  .Call(
    C_cudaverse_cuda_gather,
    storage,
    as.integer(indices),
    as.integer(shape)
  )
}

.native_replace <- function(storage, indices, replacement,
                            replacement_indices) {
  .native_ensure_kernels()
  .Call(
    C_cudaverse_cuda_scatter,
    storage,
    as.integer(indices),
    replacement,
    as.integer(replacement_indices)
  )
}

.native_broadcast <- function(storage, source_shape, target_shape) {
  .native_ensure_kernels()
  .Call(C_cudaverse_cuda_broadcast, storage, as.integer(target_shape))
}

.native_binary <- function(x, y, operator) {
  .native_ensure_kernels()
  .Call(C_cudaverse_cuda_binary, x, y, as.character(operator))
}

.native_transpose <- function(storage) {
  .native_ensure_kernels()
  .Call(C_cudaverse_cuda_transpose, storage)
}

.native_reduce <- function(storage, dim, keepdim, method) {
  .native_ensure_kernels()
  .Call(
    C_cudaverse_cuda_reduce,
    storage,
    if (is.null(dim)) integer() else as.integer(dim),
    as.logical(keepdim),
    as.character(method)
  )
}

.native_algorithm_matrix_validate <- function(storage,
                                              check_constant = FALSE) {
  .native_ensure_kernels()
  .Call(
    C_cudaverse_cuda_matrix_validate,
    storage,
    as.logical(check_constant)
  )
}

.native_svd_from_storage <- function(storage, shape, nu, nv) {
  .native_ensure_kernels()
  result <- .Call(
    C_cudaverse_cuda_svd,
    storage,
    as.integer(nu),
    as.integer(nv)
  )
  result$u <- matrix(result$u, nrow = shape[[1L]], ncol = nu)
  result$v <- matrix(result$v, nrow = shape[[2L]], ncol = nv)
  result
}

.native_algorithm_svd_storage <- function(storage, shape, dtype, nu, nv) {
  dtype <- match.arg(dtype, c("integer", "float32", "float64"))
  storage64 <- .native_cast(storage, "float64")
  on.exit(.native_release(storage64), add = TRUE)
  .native_svd_from_storage(storage64, as.integer(shape), nu, nv)
}

.native_algorithm_svd <- function(x, nu, nv) {
  storage <- .native_from_host(x, "float64", dim(x))
  on.exit(.native_release(storage), add = TRUE)
  .native_svd_from_storage(storage, dim(x), nu, nv)
}

.native_pca_from_storage <- function(storage, shape, n_components,
                                     center, scale) {
  result <- .Call(
    C_cudaverse_cuda_pca,
    storage,
    as.integer(n_components),
    as.logical(center),
    as.logical(scale)
  )
  result$rotation <- matrix(
    result$rotation,
    nrow = shape[[2L]],
    ncol = n_components
  )
  result$x <- matrix(result$x, nrow = shape[[1L]], ncol = n_components)
  attr(result$x, "cudaverse_native_state") <- list(
    storage = result$scores_storage,
    shape = as.integer(c(shape[[1L]], n_components)),
    dtype = "float64",
    backend = "native"
  )
  result$scores_storage <- NULL
  result$center <- if (isTRUE(center)) result$center else FALSE
  result$scale <- if (isTRUE(scale)) result$scale else FALSE
  result
}

.native_algorithm_pca <- function(x, n_components, center, scale) {
  .native_ensure_kernels()
  storage <- .native_from_host(x, "float64", dim(x))
  on.exit(.native_release(storage), add = TRUE)
  .native_pca_from_storage(
    storage, dim(x), n_components, center, scale
  )
}

.native_algorithm_pca_storage <- function(storage, shape, dtype,
                                          n_components, center, scale) {
  dtype <- match.arg(dtype, c("integer", "float32", "float64"))
  storage64 <- .native_cast(storage, "float64")
  on.exit(.native_release(storage64), add = TRUE)
  .native_pca_from_storage(
    storage64, as.integer(shape), n_components, center, scale
  )
}

.native_algorithm_pca_predict <- function(values, center, scale, rotation) {
  .native_ensure_kernels()
  transformed <- values
  if (is.numeric(center)) {
    transformed <- sweep(transformed, 2L, center, "-")
  }
  if (is.numeric(scale)) {
    transformed <- sweep(transformed, 2L, scale, "/")
  }
  values_storage <- .native_from_host(
    transformed, "float64", dim(transformed)
  )
  on.exit(.native_release(values_storage), add = TRUE)
  rotation_storage <- .native_from_host(
    rotation, "float64", dim(rotation)
  )
  on.exit(.native_release(rotation_storage), add = TRUE)
  scores_storage <- .native_matmul(values_storage, rotation_storage)
  release_scores <- TRUE
  on.exit({
    if (release_scores) .native_release(scores_storage)
  }, add = TRUE)
  scores <- matrix(
    .native_to_host(scores_storage),
    nrow = nrow(values),
    ncol = ncol(rotation)
  )
  attr(scores, "cudaverse_native_state") <- list(
    storage = scores_storage,
    shape = as.integer(dim(scores)),
    dtype = "float64",
    backend = "native"
  )
  release_scores <- FALSE
  scores
}

.native_algorithm_sparse_pca <- function(storage, shape, n_components,
                                         center, scale) {
  dense <- .native_sparse_to_dense(storage)
  on.exit(.native_release(dense), add = TRUE)
  .native_pca_from_storage(
    dense, as.integer(shape), n_components, center, scale
  )
}

.native_matrix_storage <- function(values) {
  state <- attr(values, "cudaverse_native_state", exact = TRUE)
  valid_state <- is.list(state) && identical(state$backend, "native") &&
    identical(state$dtype, "float64") &&
    identical(state$shape, as.integer(dim(values))) &&
    typeof(state$storage) == "externalptr"
  if (valid_state) {
    return(.native_share(state$storage))
  }
  .native_from_host(values, "float64", dim(values))
}

.native_distance_storage <- function(values, source_values, metric) {
  source_state <- attr(
    source_values, "cudaverse_native_state", exact = TRUE
  )
  resident_source <- is.list(source_state) &&
    identical(source_state$backend, "native") &&
    typeof(source_state$storage) == "externalptr"
  if (identical(metric, "cosine") && resident_source) {
    raw <- .native_matrix_storage(source_values)
    on.exit(.native_release(raw), add = TRUE)
    return(.Call(C_cudaverse_cuda_normalize_rows, raw))
  }
  .native_matrix_storage(values)
}

.native_algorithm_distance <- function(x, y, metric,
                                       source_x = x, source_y = y) {
  .native_ensure_kernels()
  self <- identical(source_x, source_y)
  x_storage <- .native_distance_storage(x, source_x, metric)
  on.exit(.native_release(x_storage), add = TRUE)
  y_storage <- if (self) {
    x_storage
  } else {
    .native_distance_storage(y, source_y, metric)
  }
  if (!self) on.exit(.native_release(y_storage), add = TRUE)
  distance_storage <- .Call(
    C_cudaverse_cuda_distance,
    x_storage,
    y_storage,
    as.character(metric),
    self
  )
  on.exit(.native_release(distance_storage), add = TRUE)
  matrix(
    .native_to_host(distance_storage),
    nrow = nrow(x),
    ncol = nrow(y)
  )
}

.native_algorithm_distance_batched <- function(x, y, metric, batch_size,
                                                source_x = x,
                                                source_y = y) {
  .native_ensure_kernels()
  self <- identical(source_x, source_y)
  query_storage <- .native_distance_storage(x, source_x, metric)
  on.exit(.native_release(query_storage), add = TRUE)
  reference_storage <- if (self) {
    query_storage
  } else {
    .native_distance_storage(y, source_y, metric)
  }
  if (!self) on.exit(.native_release(reference_storage), add = TRUE)
  reference_norms <- if (identical(metric, "euclidean")) {
    .Call(C_cudaverse_cuda_row_norms, reference_storage)
  } else {
    NULL
  }
  if (!is.null(reference_norms)) {
    on.exit(.native_release(reference_norms), add = TRUE)
  }

  result <- matrix(NA_real_, nrow = nrow(x), ncol = nrow(y))
  starts <- seq.int(1L, nrow(x), by = batch_size)
  for (start in starts) {
    count <- min(batch_size, nrow(x) - start + 1L)
    rows <- seq.int(start, length.out = count)
    block <- .Call(
      C_cudaverse_cuda_distance_block,
      query_storage,
      reference_storage,
      reference_norms,
      as.integer(start - 1L),
      as.integer(count),
      as.character(metric),
      self
    )
    result[rows, ] <- matrix(block, nrow = count, ncol = nrow(y))
  }
  result
}

.native_algorithm_kmeans <- function(x, centers, iter_max, tolerance,
                                     batch_size = 256L) {
  .native_ensure_kernels()
  input_storage <- .native_from_host(x, "float64", dim(x))
  on.exit(.native_release(input_storage), add = TRUE)
  center_storage <- .native_from_host(centers, "float64", dim(centers))
  on.exit(.native_release(center_storage), add = TRUE)
  result <- .Call(
    C_cudaverse_cuda_kmeans,
    input_storage,
    center_storage,
    as.integer(iter_max),
    as.numeric(tolerance),
    as.integer(batch_size)
  )
  result$centers <- matrix(
    result$centers,
    nrow = nrow(centers),
    ncol = ncol(centers)
  )
  result
}

.native_knn_prepare <- function(values, metric = "euclidean",
                                source_values = values) {
  .native_ensure_kernels()
  storage <- .native_distance_storage(values, source_values, metric)
  norms <- .Call(C_cudaverse_cuda_row_norms, storage)
  list(
    storage = storage,
    norms = norms,
    rows = nrow(values),
    columns = ncol(values)
  )
}

.native_sparse_knn_prepare <- function(storage, shape,
                                       metric = "euclidean") {
  .native_ensure_kernels()
  dense <- .native_sparse_to_dense(storage)
  if (identical(metric, "cosine")) {
    normalized <- .Call(C_cudaverse_cuda_normalize_rows, dense)
    .native_release(dense)
    dense <- normalized
  }
  norms <- tryCatch(
    .Call(C_cudaverse_cuda_row_norms, dense),
    error = function(error) {
      .native_release(dense)
      stop(error)
    }
  )
  list(
    storage = dense,
    norms = norms,
    rows = as.integer(shape[[1L]]),
    columns = as.integer(shape[[2L]])
  )
}

.native_knn_block_compat <- function(storage, values, rows, metric) {
  .native_algorithm_distance(
    values[rows, , drop = FALSE],
    values,
    metric
  )
}

.native_knn_select <- function(storage, values, k, metric, batch_size) {
  on.exit(.native_release(storage$storage), add = TRUE)
  on.exit(.native_release(storage$norms), add = TRUE)
  starts <- seq.int(1L, storage$rows, by = batch_size)
  index <- matrix(NA_integer_, storage$rows, k)
  distance <- matrix(NA_real_, storage$rows, k)
  for (start in starts) {
    count <- min(batch_size, storage$rows - start + 1L)
    block <- .Call(
      C_cudaverse_cuda_knn_block,
      storage$storage,
      storage$norms,
      as.integer(start - 1L),
      as.integer(count),
      as.integer(k),
      as.character(metric)
    )
    rows <- seq.int(start, length.out = count)
    index[rows, ] <- matrix(block$index, nrow = count, ncol = k)
    distance[rows, ] <- matrix(block$distance, nrow = count, ncol = k)
  }
  list(index = index, distance = distance)
}

.native_matmul <- function(x, y) {
  .Call(C_cudaverse_cuda_matmul, x, y)
}

.native_synchronize <- function() {
  invisible(.Call(C_cudaverse_cuda_synchronize))
}

.native_release <- function(storage) {
  invisible(.Call(C_cudaverse_cuda_release, storage))
}

.native_error_translate <- function(error, operation) {
  structure(
    list(
      message = sprintf(
        "Native CUDA backend failed during `%s`: %s",
        operation,
        conditionMessage(error)
      ),
      call = NULL,
      backend = "native",
      operation = operation,
      parent = error
    ),
    class = c(
      "cudaverse_native_error",
      "cudaverse_backend_operation_error",
      "cudaverse_backend_error",
      "error",
      "condition"
    )
  )
}

.native_backend_factory <- function() {
  list(
    name = "native",
    device = "cuda",
    contract = .native_contract,
    diagnostics = .native_diagnostics,
    capabilities = .native_capabilities,
    from_host = .native_from_host,
    to_host = .native_to_host,
    sparse_from_coo = .native_sparse_from_coo,
    sparse_to_host = .native_sparse_to_host,
    sparse_transpose = .native_sparse_transpose,
    sparse_reduce = .native_sparse_reduce,
    sparse_normalize = .native_sparse_normalize,
    sparse_matmul_dense = .native_sparse_matmul_dense,
    sparse_to_dense = .native_sparse_to_dense,
    sparse_share = .native_sparse_share,
    sparse_release = .native_sparse_release,
    cast = .native_cast,
    reshape = .native_reshape,
    subset = .native_subset,
    replace = .native_replace,
    broadcast = .native_broadcast,
    binary = .native_binary,
    transpose = .native_transpose,
    matmul = .native_matmul,
    reduce = .native_reduce,
    algorithm_matrix_validate = .native_algorithm_matrix_validate,
    algorithm_svd = .native_algorithm_svd,
    algorithm_svd_storage = .native_algorithm_svd_storage,
    algorithm_pca = .native_algorithm_pca,
    algorithm_pca_storage = .native_algorithm_pca_storage,
    algorithm_pca_predict = .native_algorithm_pca_predict,
    algorithm_sparse_pca = .native_algorithm_sparse_pca,
    algorithm_distance = .native_algorithm_distance,
    algorithm_distance_batched = .native_algorithm_distance_batched,
    algorithm_kmeans = .native_algorithm_kmeans,
    algorithm_kmeans_batched = .native_algorithm_kmeans,
    algorithm_knn_prepare = .native_knn_prepare,
    algorithm_sparse_knn_prepare = .native_sparse_knn_prepare,
    algorithm_knn_block = .native_knn_block_compat,
    algorithm_knn_select = .native_knn_select,
    memory_info = .native_backend_memory_info,
    synchronize = .native_synchronize,
    release = .native_release,
    test_inject_cuda_error = .native_test_inject_cuda_error,
    error_translate = .native_error_translate
  )
}

.native_memory_info <- function() {
  .Call(C_cudaverse_cuda_memory_info)
}

.native_memory_tracker <- function(reset = FALSE) {
  .Call(C_cudaverse_cuda_memory_tracker, reset)
}

.native_backend_memory_info <- function() {
  physical <- .native_memory_info()
  tracked <- .native_memory_tracker()
  list(
    available = TRUE,
    total_bytes = as.numeric(physical$total),
    free_bytes = as.numeric(physical$free),
    used_bytes = as.numeric(physical$used),
    allocated_bytes = as.numeric(tracked$current),
    allocated_peak_bytes = as.numeric(tracked$peak),
    reserved_bytes = NA_real_,
    reserved_peak_bytes = NA_real_,
    reason = "native_driver_reported"
  )
}

.native_test_inject_cuda_error <- function(bytes = 4096L) {
  .Call(C_cudaverse_cuda_test_inject_error, as.integer(bytes))
}

.native_share <- function(storage) {
  .Call(C_cudaverse_cuda_share, storage)
}

.native_sparse_release <- function(storage) {
  invisible(.Call(C_cudaverse_cuda_sparse_release, storage))
}

.native_sparse_share <- function(storage) {
  .Call(C_cudaverse_cuda_sparse_share, storage)
}
