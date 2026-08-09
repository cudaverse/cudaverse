.cudaverse_backends <- new.env(parent = emptyenv())
.cudaverse_backend_discovery <- new.env(parent = emptyenv())

.backend_condition <- function(message, subclass, backend = NULL,
                               operation = NULL, parent = NULL) {
  structure(
    list(
      message = message,
      call = NULL,
      backend = backend,
      operation = operation,
      parent = parent
    ),
    class = c(subclass, "cudaverse_backend_error", "error", "condition")
  )
}

.backend_stop <- function(message, subclass = "cudaverse_backend_error",
                          backend = NULL, operation = NULL, parent = NULL) {
  stop(.backend_condition(message, subclass, backend, operation, parent))
}

.backend_validate <- function(factory, name = NULL) {
  if (!is.list(factory)) {
    .backend_stop("A backend factory must return a named list.",
                  "cudaverse_backend_contract_error", name)
  }
  required_fields <- c(
    "name", "device", "diagnostics", "capabilities", "from_host",
    "to_host", "matmul", "synchronize", "release", "error_translate"
  )
  missing_fields <- setdiff(required_fields, names(factory))
  if (length(missing_fields)) {
    .backend_stop(
      sprintf(
        "Backend `%s` is missing contract field(s): %s.",
        if (is.null(name)) "<unknown>" else name,
        paste(missing_fields, collapse = ", ")
      ),
      "cudaverse_backend_contract_error",
      name
    )
  }
  if (!is.character(factory$name) || length(factory$name) != 1L ||
      !nzchar(factory$name) || !factory$device %in% c("cpu", "cuda")) {
    .backend_stop("Backend `name` and `device` fields are invalid.",
                  "cudaverse_backend_contract_error", name)
  }
  callable <- setdiff(required_fields, c("name", "device"))
  if (any(!vapply(factory[callable], is.function, logical(1)))) {
    .backend_stop("Backend contract operations must be functions.",
                  "cudaverse_backend_contract_error", factory$name)
  }
  factory
}

.backend_register <- function(factory, replace = FALSE) {
  factory <- .backend_validate(factory)
  name <- factory$name
  if (exists(name, envir = .cudaverse_backends, inherits = FALSE) &&
      !isTRUE(replace)) {
    .backend_stop(sprintf("Backend `%s` is already registered.", name),
                  "cudaverse_backend_contract_error", name)
  }
  assign(name, factory, envir = .cudaverse_backends)
  invisible(factory)
}

.backend_get <- function(name) {
  .backend_register_builtins()
  .backend_discover_native()
  if (!exists(name, envir = .cudaverse_backends, inherits = FALSE)) {
    .backend_stop(sprintf("Backend `%s` is not registered.", name),
                  "cudaverse_backend_unavailable", name)
  }
  get(name, envir = .cudaverse_backends, inherits = FALSE)
}

.backend_call <- function(name, operation, ...) {
  backend <- .backend_get(name)
  method <- backend[[operation]]
  if (!is.function(method)) {
    .backend_stop(
      sprintf("Backend `%s` does not implement `%s`.", name, operation),
      "cudaverse_backend_capability_error",
      name,
      operation
    )
  }
  tryCatch(
    method(...),
    error = function(error) {
      if (inherits(error, "cudaverse_backend_error")) {
        stop(error)
      }
      translated <- backend$error_translate(error, operation)
      if (inherits(translated, "condition")) {
        stop(translated)
      }
      .backend_stop(
        sprintf("Backend `%s` failed during `%s`: %s",
                name, operation, conditionMessage(error)),
        "cudaverse_backend_operation_error",
        name,
        operation,
        error
      )
    }
  )
}

.backend_has_operation <- function(name, operation) {
  backend <- .backend_get(name)
  is.character(operation) && length(operation) == 1L &&
    !is.na(operation) && is.function(backend[[operation]])
}

.backend_default_error_translate <- function(backend) {
  force(backend)
  function(error, operation) {
    .backend_condition(
      sprintf("Backend `%s` failed during `%s`: %s",
              backend, operation, conditionMessage(error)),
      "cudaverse_backend_operation_error",
      backend,
      operation,
      error
    )
  }
}

.base_backend_factory <- function() {
  list(
    name = "base",
    device = "cpu",
    diagnostics = function() list(
      available = TRUE,
      device_count = 0L,
      version = as.character(getRversion()),
      reason = "cpu_available",
      detection_error = NULL
    ),
    capabilities = function() c(
      "transfer", "cast", "arithmetic", "matmul", "reduce",
      "reshape", "broadcast", "transpose", "sparse"
    ),
    from_host = function(x, dtype, shape, dimnames = NULL) {
      values <- switch(
        dtype,
        integer = as.integer(x),
        float32 = as.numeric(x),
        float64 = as.numeric(x)
      )
      array(values, dim = shape, dimnames = dimnames)
    },
    to_host = function(storage) storage,
    cast = function(storage, dtype) {
      switch(
        dtype,
        integer = array(as.integer(storage), dim = dim(storage)),
        float32 = .as_float32(storage),
        float64 = array(as.numeric(storage), dim = dim(storage))
      )
    },
    reshape = function(storage, source_shape, target_shape) {
      array(storage, dim = target_shape)
    },
    matmul = function(x, y) x %*% y,
    reduce = function(storage, dim, keepdim, method) {
      fun <- if (identical(method, "sum")) sum else mean
      shape <- base::dim(storage)
      if (is.null(dim) || length(dim) == length(shape)) {
        value <- fun(storage)
        target_shape <- if (isTRUE(keepdim)) rep(1L, length(shape)) else 1L
      } else {
        margin <- setdiff(seq_along(shape), dim)
        value <- apply(storage, margin, fun)
        target_shape <- if (isTRUE(keepdim)) {
          result <- shape
          result[dim] <- 1L
          result
        } else {
          shape[margin]
        }
      }
      list(storage = array(value, dim = target_shape),
           shape = as.integer(target_shape))
    },
    broadcast = function(storage, source_shape, target_shape) {
      padded <- c(rep(1L, length(target_shape) - length(source_shape)),
                  source_shape)
      coordinates <- arrayInd(seq_len(prod(target_shape)), .dim = target_shape)
      source_coordinates <- coordinates
      source_coordinates[, padded == 1L] <- 1L
      strides <- cumprod(c(1L, utils::head(padded, -1L)))
      source_index <- 1L + rowSums(
        sweep(source_coordinates - 1L, 2L, strides, `*`)
      )
      array(as.vector(storage)[source_index], dim = target_shape)
    },
    binary = function(x, y, operator) {
      switch(operator, "+" = x + y, "-" = x - y, "*" = x * y,
             "/" = x / y, "^" = x^y)
    },
    transpose = function(storage) t(storage),
    sparse_from_coo = function(i, j, values, shape) NULL,
    sparse_matmul_dense = function(storage, i, j, values, shape, dense) {
      Matrix::sparseMatrix(i = i, j = j, x = values, dims = shape) %*% dense
    },
    algorithm_svd = function(x, nu, nv) {
      result <- base::svd(x, nu = nu, nv = nv)
      list(d = result$d, u = result$u, v = result$v)
    },
    algorithm_pca = function(x, n_components, center, scale) {
      fit <- stats::prcomp(
        x, center = center, scale. = scale, rank. = n_components
      )
      list(
        sdev = fit$sdev[seq_len(n_components)],
        rotation = fit$rotation[, seq_len(n_components), drop = FALSE],
        x = fit$x[, seq_len(n_components), drop = FALSE],
        center = fit$center,
        scale = fit$scale
      )
    },
    algorithm_pca_predict = function(values, center, scale, rotation) {
      transformed <- values
      if (is.numeric(center)) transformed <- sweep(transformed, 2L, center, "-")
      if (is.numeric(scale)) transformed <- sweep(transformed, 2L, scale, "/")
      transformed %*% rotation
    },
    algorithm_distance = function(x, y, metric, source_x = x, source_y = y) {
      if (identical(metric, "euclidean")) {
        .euclidean_distance_cpu(x, y)
      } else {
        1 - tcrossprod(x, y)
      }
    },
    algorithm_knn_prepare = function(values, metric = "euclidean",
                                     source_values = values) values,
    algorithm_knn_block = function(storage, values, rows, metric) {
      if (identical(metric, "euclidean")) {
        .euclidean_distance_cpu(values[rows, , drop = FALSE], values)
      } else {
        1 - tcrossprod(values[rows, , drop = FALSE], values)
      }
    },
    synchronize = function() invisible(TRUE),
    release = function(storage) invisible(TRUE),
    error_translate = .backend_default_error_translate("base")
  )
}

.torch_backend_factory <- function() {
  list(
    name = "torch",
    device = "cuda",
    diagnostics = function() {
      installed <- .cuda_torch_installed()
      detection_error <- NULL
      available <- if (installed) {
        tryCatch(isTRUE(.cuda_torch_is_available()), error = function(error) {
          detection_error <<- conditionMessage(error)
          FALSE
        })
      } else {
        FALSE
      }
      count <- if (available) {
        tryCatch(.cuda_torch_device_count(), error = function(error) {
          detection_error <<- conditionMessage(error)
          NA_real_
        })
      } else {
        0L
      }
      if (available && !.valid_cuda_device_count(count)) {
        if (is.null(detection_error)) {
          detection_error <- paste0(
            "`torch::cuda_device_count()` returned an invalid value; ",
            "expected one non-negative whole number."
          )
        }
        available <- FALSE
        count <- NA_integer_
      } else {
        count <- as.integer(count)
      }
      if (!is.na(count) && count < 1L) available <- FALSE
      list(
        installed = installed,
        available = available,
        device_count = count,
        version = if (installed) .cuda_torch_version() else NA_character_,
        reason = .cuda_diagnostic_reason(installed, available, detection_error),
        detection_error = detection_error
      )
    },
    capabilities = function() c(
      "transfer", "cast", "arithmetic", "matmul", "reduce",
      "reshape", "broadcast", "transpose", "svd", "pca", "distance",
      "knn", "sparse"
    ),
    from_host = function(x, dtype, shape, dimnames = NULL) {
      torch::torch_tensor(x, dtype = .torch_dtype(dtype), device = "cuda")
    },
    to_host = function(storage) as.array(storage$to(device = "cpu")),
    cast = function(storage, dtype) storage$to(dtype = .torch_dtype(dtype)),
    reshape = function(storage, source_shape, target_shape) {
      storage$
        permute(rev(seq_along(source_shape)))$
        contiguous()$
        reshape(rev(target_shape))$
        permute(rev(seq_along(target_shape)))
    },
    matmul = function(x, y) x$matmul(y),
    reduce = function(storage, dim, keepdim, method) {
      if (is.null(dim)) {
        result <- storage[[method]]()
        if (isTRUE(keepdim)) result <- result$reshape(rep(1L, length(storage$shape)))
      } else {
        result <- storage[[method]](dim = dim, keepdim = keepdim)
      }
      shape <- as.integer(result$shape)
      if (!length(shape)) shape <- 1L
      list(storage = result, shape = shape)
    },
    broadcast = function(storage, source_shape, target_shape) {
      padded <- c(rep(1L, length(target_shape) - length(source_shape)),
                  source_shape)
      storage$reshape(padded)$expand(target_shape)
    },
    binary = function(x, y, operator) {
      switch(operator, "+" = x + y, "-" = x - y, "*" = x * y,
             "/" = x / y, "^" = x^y)
    },
    transpose = function(storage) storage$t(),
    sparse_from_coo = function(i, j, values, shape) {
      indices <- torch::torch_tensor(rbind(i, j),
                                     dtype = torch::torch_int64(),
                                     device = "cuda")
      torch_values <- torch::torch_tensor(values,
                                          dtype = torch::torch_float64(),
                                          device = "cuda")
      torch::torch_sparse_coo_tensor(
        indices = indices,
        values = torch_values,
        size = shape,
        device = "cuda"
      )$coalesce()
    },
    sparse_matmul_dense = function(storage, i, j, values, shape, dense) {
      dense_gpu <- torch::torch_tensor(dense,
                                       dtype = torch::torch_float64(),
                                       device = "cuda")
      as.array(storage$matmul(dense_gpu)$to(device = "cpu"))
    },
    algorithm_svd = function(x, nu, nv) {
      result <- torch::torch_svd(.torch_matrix(x), some = TRUE)
      list(
        d = as.vector(.torch_array(result[[2]])),
        u = if (nu == 0L) matrix(numeric(), nrow(x), 0L) else
          .torch_array(result[[1]][, seq_len(nu), drop = FALSE]),
        v = if (nv == 0L) matrix(numeric(), ncol(x), 0L) else
          .torch_array(result[[3]][, seq_len(nv), drop = FALSE])
      )
    },
    algorithm_pca = function(x, n_components, center, scale) {
      tensor <- .torch_matrix(x)
      centre_values <- if (center) {
        tensor$mean(dim = 1L, keepdim = TRUE)
      } else {
        torch::torch_zeros(
          c(1L, ncol(x)), dtype = torch::torch_float64(), device = "cuda"
        )
      }
      transformed <- tensor - centre_values
      scale_values <- if (scale) {
        transformed$std(dim = 1L, unbiased = TRUE, keepdim = TRUE)
      } else {
        torch::torch_ones(
          c(1L, ncol(x)), dtype = torch::torch_float64(), device = "cuda"
        )
      }
      transformed <- transformed / scale_values
      decomposition <- torch::torch_svd(transformed, some = TRUE)
      components <- seq_len(n_components)
      scores <- decomposition[[1]][, components, drop = FALSE] *
        decomposition[[2]][components]
      list(
        sdev = as.vector(.torch_array(
          decomposition[[2]][components] / sqrt(nrow(x) - 1)
        )),
        rotation = .torch_array(
          decomposition[[3]][, components, drop = FALSE]
        ),
        x = .torch_array(scores),
        center = if (center) as.vector(.torch_array(centre_values)) else FALSE,
        scale = if (scale) as.vector(.torch_array(scale_values)) else FALSE
      )
    },
    algorithm_pca_predict = function(values, center, scale, rotation) {
      transformed <- .torch_matrix(values)
      if (is.numeric(center)) {
        transformed <- transformed - .torch_matrix(matrix(center, nrow = 1L))
      }
      if (is.numeric(scale)) {
        transformed <- transformed / .torch_matrix(matrix(scale, nrow = 1L))
      }
      matrix(
        .torch_array(transformed$matmul(.torch_matrix(rotation))),
        nrow = nrow(values),
        ncol = ncol(rotation)
      )
    },
    algorithm_distance = function(x, y, metric, source_x = x, source_y = y) {
      x_gpu <- .torch_matrix(x)
      y_gpu <- if (identical(x, y)) x_gpu else .torch_matrix(y)
      result <- if (identical(metric, "euclidean")) {
        torch::torch_cdist(x_gpu, y_gpu, p = 2)
      } else {
        1 - x_gpu$matmul(y_gpu$t())
      }
      .torch_array(result)
    },
    algorithm_knn_prepare = function(values, metric = "euclidean",
                                     source_values = values) {
      .torch_matrix(values)
    },
    algorithm_knn_block = function(storage, values, rows, metric) {
      query <- storage[rows, , drop = FALSE]
      result <- if (identical(metric, "euclidean")) {
        torch::torch_cdist(query, storage, p = 2)
      } else {
        1 - query$matmul(storage$t())
      }
      .torch_array(result)
    },
    synchronize = function() {
      torch::cuda_synchronize()
      invisible(TRUE)
    },
    release = function(storage) invisible(TRUE),
    error_translate = .backend_default_error_translate("torch")
  )
}

.backend_register_builtins <- function() {
  if (!exists("base", envir = .cudaverse_backends, inherits = FALSE)) {
    .backend_register(.base_backend_factory())
  }
  if (!exists("torch", envir = .cudaverse_backends, inherits = FALSE)) {
    .backend_register(.torch_backend_factory())
  }
  invisible(TRUE)
}

.backend_discover_native <- function() {
  if (isTRUE(.cudaverse_backend_discovery$attempted)) return(invisible(NULL))
  .cudaverse_backend_discovery$attempted <- TRUE
  extension <- "cudaverseCUDA"
  if (!requireNamespace(extension, quietly = TRUE)) {
    .cudaverse_backend_discovery$error <- NULL
    return(invisible(NULL))
  }
  factory <- tryCatch(
    getExportedValue(extension, "cudaverse_cuda_backend_factory")(),
    error = identity
  )
  if (inherits(factory, "error")) {
    .cudaverse_backend_discovery$error <- conditionMessage(factory)
    return(invisible(NULL))
  }
  tryCatch(
    .backend_register(factory, replace = TRUE),
    error = function(error) {
      .cudaverse_backend_discovery$error <- conditionMessage(error)
      NULL
    }
  )
  invisible(NULL)
}

.backend_diagnostics <- function(name) {
  tryCatch(
    .backend_call(name, "diagnostics"),
    error = function(error) list(
      available = FALSE,
      device_count = NA_integer_,
      version = NA_character_,
      reason = "backend_error",
      detection_error = conditionMessage(error)
    )
  )
}

.backend_cuda_order <- function() {
  requested <- getOption("cudaverse.cuda_backends", c("torch", "native"))
  intersect(unique(as.character(requested)), c("torch", "native"))
}

.backend_select_cuda <- function(diagnostics = NULL) {
  .backend_register_builtins()
  .backend_discover_native()
  registered <- ls(.cudaverse_backends, all.names = TRUE)
  for (name in .backend_cuda_order()) {
    if (!name %in% registered) next
    details <- if (!is.null(diagnostics) &&
                   !is.null(diagnostics$backend_diagnostics[[name]])) {
      diagnostics$backend_diagnostics[[name]]
    } else {
      .backend_diagnostics(name)
    }
    if (isTRUE(details$available)) return(name)
  }
  NULL
}
