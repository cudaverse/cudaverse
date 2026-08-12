#' Detect a usable CUDA backend
#'
#' Detection uses the built-in lightweight native backend or a CUDA-enabled
#' installation of the optional `torch` package. NVIDIA libraries are loaded
#' only when diagnostics or CUDA selection is requested.
#'
#' @return A single logical value.
#' @export
#' @examples
#' cuda_available()
cuda_available <- function() {
  isTRUE(cuda_diagnostics()$cuda_available)
}

.tensor_dtype <- function(x) {
  if (is.integer(x)) "integer" else "float64"
}

.as_float32 <- function(x) {
  values <- readBin(
    writeBin(as.numeric(x), raw(), size = 4L),
    what = double(),
    n = length(x),
    size = 4L
  )
  array(values, dim = dim(x), dimnames = dimnames(x))
}

.validate_tensor_dimnames <- function(value, shape, argument = "dimnames") {
  if (is.null(value)) {
    return(NULL)
  }
  if (!is.list(value) || length(value) != length(shape)) {
    stop(
      sprintf(
        "`%s` must be NULL or a list with one entry per tensor dimension.",
        argument
      ),
      call. = FALSE
    )
  }

  result <- vector("list", length(shape))
  for (index in seq_along(shape)) {
    labels <- value[[index]]
    if (is.null(labels)) {
      next
    }
    if (!is.character(labels) || !is.null(dim(labels)) ||
        length(labels) != shape[[index]]) {
      stop(
        sprintf(
          "Each non-NULL `%s` entry must be a character vector matching its tensor dimension.",
          argument
        ),
        call. = FALSE
      )
    }
    result[[index]] <- labels
  }
  if (!is.null(names(value))) {
    names(result) <- names(value)
  }
  result
}

.tensor_dimnames <- function(x) {
  .validate_tensor_dimnames(x$dimnames, x$shape)
}

.new_tensor_dimnames <- function(labels, axis_names = NULL) {
  if (!is.null(axis_names)) {
    names(labels) <- axis_names
  }
  if (all(vapply(labels, is.null, logical(1))) &&
      is.null(axis_names)) {
    return(NULL)
  }
  labels
}

.tensor_axis_name <- function(value, index) {
  if (is.null(value) || is.null(names(value))) {
    return(NULL)
  }
  names(value)[[index]]
}

.meaningful_axis_name <- function(value) {
  !is.null(value) && (is.na(value) || nzchar(value))
}

.aligned_tensor_dimnames <- function(x, shape) {
  source <- .tensor_dimnames(x)
  if (is.null(source)) {
    return(NULL)
  }

  offset <- length(shape) - length(x$shape)
  labels <- vector("list", length(shape))
  source_axis_names <- names(source)
  axis_names <- if (is.null(source_axis_names)) {
    NULL
  } else {
    rep("", length(shape))
  }
  for (index in seq_along(x$shape)) {
    target <- offset + index
    if (x$shape[[index]] == shape[[target]]) {
      labels[target] <- list(source[[index]])
    }
    if (!is.null(axis_names)) {
      axis_names[[target]] <- source_axis_names[[index]]
    }
  }
  .new_tensor_dimnames(labels, axis_names)
}

.merge_tensor_dimnames <- function(x, y, context = "arithmetic") {
  x_names <- .tensor_dimnames(x)
  y_names <- .tensor_dimnames(y)
  if (is.null(x_names) && is.null(y_names)) {
    return(NULL)
  }

  labels <- vector("list", length(x$shape))
  axis_names <- rep("", length(x$shape))
  has_axis_names <- FALSE
  for (index in seq_along(labels)) {
    x_labels <- if (is.null(x_names)) NULL else x_names[[index]]
    y_labels <- if (is.null(y_names)) NULL else y_names[[index]]
    if (!is.null(x_labels) && !is.null(y_labels) &&
        !identical(x_labels, y_labels)) {
      stop(
        sprintf(
          "Tensor dimension names are incompatible for %s on dimension %s.",
          context, index
        ),
        call. = FALSE
      )
    }
    labels[index] <- list(
      if (!is.null(x_labels)) x_labels else y_labels
    )

    x_axis <- .tensor_axis_name(x_names, index)
    y_axis <- .tensor_axis_name(y_names, index)
    selected_axis <- if (.meaningful_axis_name(x_axis)) x_axis else y_axis
    if (.meaningful_axis_name(selected_axis)) {
      axis_names[[index]] <- selected_axis
      has_axis_names <- TRUE
    }
  }
  .new_tensor_dimnames(
    labels,
    if (has_axis_names) axis_names else NULL
  )
}

.matmul_tensor_dimnames <- function(x, y) {
  x_names <- .tensor_dimnames(x)
  y_names <- .tensor_dimnames(y)
  x_inner <- if (is.null(x_names)) NULL else x_names[[2L]]
  y_inner <- if (is.null(y_names)) NULL else y_names[[1L]]
  if (!is.null(x_inner) && !is.null(y_inner) &&
      !identical(x_inner, y_inner)) {
    stop(
      "Tensor inner dimension names are incompatible for matrix multiplication.",
      call. = FALSE
    )
  }

  labels <- list(
    if (is.null(x_names)) NULL else x_names[[1L]],
    if (is.null(y_names)) NULL else y_names[[2L]]
  )
  x_axis <- .tensor_axis_name(x_names, 1L)
  y_axis <- .tensor_axis_name(y_names, 2L)
  has_axis_names <- .meaningful_axis_name(x_axis) ||
    .meaningful_axis_name(y_axis)
  axis_names <- if (has_axis_names) {
    c(
      if (.meaningful_axis_name(x_axis)) x_axis else "",
      if (.meaningful_axis_name(y_axis)) y_axis else ""
    )
  } else {
    NULL
  }
  .new_tensor_dimnames(labels, axis_names)
}

.reduced_tensor_dimnames <- function(x, dim, keepdim) {
  source <- .tensor_dimnames(x)
  if (is.null(source)) {
    return(NULL)
  }
  all_dimensions <- seq_along(x$shape)
  reduced <- if (is.null(dim)) all_dimensions else dim
  if (!length(reduced)) {
    return(source)
  }
  if (!isTRUE(keepdim)) {
    retained <- setdiff(all_dimensions, reduced)
    if (!length(retained)) {
      return(NULL)
    }
    result <- source[retained]
    return(.new_tensor_dimnames(result, names(result)))
  }

  result <- source
  for (index in reduced) {
    result[index] <- list(NULL)
  }
  .new_tensor_dimnames(result, names(source))
}

.validate_integer_values <- function(x, argument = "x") {
  values <- as.numeric(x)
  representable <- is.finite(values) &
    values == trunc(values) &
    values >= -.Machine$integer.max &
    values <= .Machine$integer.max
  if (any(!representable)) {
    stop(
      sprintf(
        "`%s` contains values that cannot be represented exactly as integer dtype.",
        argument
      ),
      call. = FALSE
    )
  }
  invisible(x)
}

.new_cudatensor <- function(storage, device, backend, dtype, shape,
                            dimnames = NULL, compute_stages = NULL) {
  shape <- as.integer(shape)
  if (is.null(compute_stages)) {
    compute_stages <- list(
      tensor_operation = cuda_stage(
        requested_device = "inherited",
        device = device,
        backend = backend,
        selection_reason = "inherited_device",
        fallback = FALSE,
        output_device = device
      )
    )
  }
  compute_stages <- .validate_cuda_stages(compute_stages)
  structure(
    list(
      storage = storage,
      device = device,
      backend = backend,
      dtype = dtype,
      shape = shape,
      dimnames = .validate_tensor_dimnames(dimnames, shape),
      provenance_schema = .cudaverse_provenance_schema,
      compute_device = .compute_device_from_stages(compute_stages),
      compute_stages = compute_stages
    ),
    class = "cudatensor"
  )
}

.tensor_stage <- function(device, backend, output_device = device,
                          requested_device = "inherited",
                          reason = "inherited_device") {
  cuda_stage(
    requested_device = requested_device,
    device = device,
    backend = backend,
    selection_reason = reason,
    fallback = FALSE,
    output_device = output_device
  )
}

.with_tensor_stages <- function(x, stages) {
  .check_tensor(x)
  provenance <- cuda_provenance(stages)
  x$provenance_schema <- attr(provenance, "schema", exact = TRUE)
  x$compute_device <- attr(provenance, "compute_device", exact = TRUE)
  x$compute_stages <- attr(provenance, "compute_stages", exact = TRUE)
  x
}

.tensor_result_stage <- function(x, stage, device = x$device,
                                 backend = x$backend,
                                 output_device = device,
                                 requested_device = "inherited",
                                 reason = "inherited_device") {
  stages <- list(.tensor_stage(
    device = device,
    backend = backend,
    output_device = output_device,
    requested_device = requested_device,
    reason = reason
  ))
  names(stages) <- stage
  .with_tensor_stages(x, stages)
}

.torch_dtype <- function(dtype) {
  switch(
    dtype,
    float32 = torch::torch_float32(),
    float64 = torch::torch_float64(),
    integer = torch::torch_int64()
  )
}

.promote_tensor_dtype <- function(x, y, operation = "arithmetic") {
  ranks <- c(integer = 1L, float32 = 2L, float64 = 3L)
  dtype <- if (ranks[[x]] >= ranks[[y]]) x else y
  if (identical(x, "integer") && identical(y, "integer") &&
      operation %in% c("arithmetic", "division", "power", "matmul")) {
    return("float64")
  }
  dtype
}

.cast_tensor <- function(x, dtype) {
  .check_tensor(x)
  dtype <- match.arg(dtype, c("float64", "float32", "integer"))
  if (identical(x$dtype, dtype)) {
    return(x)
  }
  if (identical(dtype, "integer")) {
    .validate_integer_values(to_cpu(x))
  }
  storage <- .backend_call(x$backend, "cast", x$storage, dtype)
  .new_cudatensor(
    storage, x$device, x$backend, dtype, x$shape,
    dimnames = .tensor_dimnames(x),
    compute_stages = list(
      cast = .tensor_stage(x$device, x$backend)
    )
  )
}

#' Create a GPU-aware tensor
#'
#' @param x Numeric vector, matrix, array, or another `cudatensor`.
#' @param device One of `"auto"`, `"cuda"`, or `"cpu"`. Auto selects CUDA
#'   only when [cuda_available()] is true.
#' @param dtype One of `"float64"`, `"float32"`, or `"integer"`.
#'
#' Matrix and array dimnames, including names on a one-dimensional input, are
#' retained as R metadata on both CPU and CUDA tensors.
#' Floating dtypes accept IEEE `Inf`, `-Inf`, `NaN`, and R's floating `NA`;
#' torch backends may normalize `NA` to `NaN`. Integer dtype rejects
#' non-finite or fractional values because they have no exact integer
#' representation.
#'
#' @return A `cudatensor` object.
#' @export
#' @examples
#' x <- cuda_tensor(matrix(1:6, nrow = 2), device = "cpu")
#' x
cuda_tensor <- function(x, device = c("auto", "cuda", "cpu"),
                        dtype = NULL) {
  requested_device <- match.arg(device)
  selection <- cuda_select_device(requested_device)
  device <- selection$device
  if (inherits(x, "cudatensor")) {
    .tensor_dimnames(x)
    if (is.null(dtype)) {
      dtype <- x$dtype
    }
    if (identical(device, x$device) && identical(dtype, x$dtype)) {
      return(x)
    }
    x <- to_cpu(x)
  }

  if (!is.numeric(x) || length(x) == 0L) {
    stop("`x` must be a non-empty numeric object.", call. = FALSE)
  }
  if (is.null(dim(x))) {
    value_names <- names(x)
    dim(x) <- length(x)
    if (!is.null(value_names)) {
      dimnames(x) <- list(value_names)
    }
  }
  if (is.null(dtype)) {
    dtype <- .tensor_dtype(x)
  }
  dtype <- match.arg(dtype, c("float64", "float32", "integer"))
  if (identical(dtype, "integer")) {
    .validate_integer_values(x)
  }
  if (identical(dtype, "float32")) {
    x <- .as_float32(x)
  }
  shape <- dim(x)
  tensor_dimnames <- .validate_tensor_dimnames(dimnames(x), shape)
  backend <- if (is.null(selection$backend)) {
    if (identical(device, "cuda")) "torch" else "base"
  } else {
    selection$backend
  }
  materialization <- list(
    tensor_materialization = cuda_stage(
      requested_device = selection$requested_device,
      device = selection$device,
      backend = backend,
      selection_reason = selection$selection_reason,
      fallback = selection$fallback,
      output_device = selection$device
    )
  )
  storage <- .backend_call(
    backend, "from_host", x, dtype, shape, tensor_dimnames
  )
  .new_cudatensor(
    storage, device, backend, dtype, shape,
    dimnames = tensor_dimnames,
    compute_stages = materialization
  )
}

#' Inspect tensor device and backend
#'
#' @param x A `cudatensor`.
#' @return A named character vector.
#' @export
#' @examples
#' tensor_device(cuda_tensor(1:3, device = "cpu"))
tensor_device <- function(x) {
  .check_tensor(x)
  c(device = x$device, backend = x$backend)
}

#' Inspect tensor shape
#'
#' @param x A `cudatensor`.
#' @return An integer vector.
#' @export
#' @examples
#' tensor_shape(cuda_tensor(matrix(1:6, 2), device = "cpu"))
tensor_shape <- function(x) {
  .check_tensor(x)
  x$shape
}

#' Reshape a tensor without changing its values
#'
#' @param x A `cudatensor`.
#' @param shape Positive whole-number dimensions whose product equals
#'   `length(x)`.
#' @details The native CUDA backend creates an allocation-free metadata view
#'   that shares the source device allocation. The source and reshaped tensor
#'   have independent external-pointer lifetimes, and the allocation is freed
#'   only after the final view is released. Compatibility backends retain their
#'   established reshape behavior.
#' @return A `cudatensor` on the same device with the requested shape.
#' @export
#' @examples
#' x <- cuda_tensor(1:6, device = "cpu")
#' tensor_reshape(x, c(2, 3))
tensor_reshape <- function(x, shape) {
  .check_tensor(x)
  if (!is.numeric(shape) || length(shape) == 0L || anyNA(shape) ||
      any(!is.finite(shape)) || any(shape < 1) ||
      any(shape != as.integer(shape))) {
    stop("`shape` must contain positive whole-number dimensions.",
         call. = FALSE)
  }
  shape <- as.integer(shape)
  if (prod(shape) != length(x)) {
    stop("The requested shape must contain exactly `length(x)` values.",
         call. = FALSE)
  }

  storage <- .backend_call(
    x$backend, "reshape", x$storage, x$shape, shape
  )
  .new_cudatensor(
    storage,
    x$device,
    x$backend,
    x$dtype,
    shape,
    compute_stages = list(
      reshape = .tensor_stage(x$device, x$backend)
    )
  )
}

.check_tensor <- function(x, argument = "x") {
  if (!inherits(x, "cudatensor")) {
    stop(sprintf("`%s` must be a `cudatensor`.", argument), call. = FALSE)
  }
  invisible(x)
}

#' Transfer a tensor to a device
#'
#' @param x A `cudatensor`.
#' @param device `"cpu"` or `"cuda"`.
#' @return A `cudatensor` on the requested device.
#' @export
#' @examples
#' x <- cuda_tensor(1:4, device = "cpu")
#' to_device(x, "cpu")
to_device <- function(x, device = c("cpu", "cuda")) {
  .check_tensor(x)
  device <- match.arg(device)
  if (identical(x$device, device)) {
    return(x)
  }
  result <- cuda_tensor(to_cpu(x), device = device, dtype = x$dtype)
  .tensor_result_stage(
    result,
    "device_transfer",
    requested_device = device,
    reason = "explicit_transfer"
  )
}

#' Transfer tensor data to base R
#'
#' @param x A `cudatensor`.
#' @return A base R vector, matrix, or array with the tensor shape.
#' @export
#' @examples
#' to_cpu(cuda_tensor(matrix(1:4, 2), device = "cpu"))
to_cpu <- function(x) {
  .check_tensor(x)
  result <- .backend_call(x$backend, "to_host", x$storage)
  dim(result) <- x$shape
  dimnames(result) <- .tensor_dimnames(x)
  result
}

#' Matrix multiplication for tensors
#'
#' @param x,y Two-dimensional `cudatensor` objects or numeric matrices.
#' @details Row names come from `x` and column names come from `y`. When both
#'   operands name the contracted dimension, those names must be identical.
#' @return A `cudatensor`.
#' @export
#' @examples
#' x <- cuda_tensor(matrix(1:6, 2, 3), device = "cpu")
#' y <- cuda_tensor(matrix(1:6, 3, 2), device = "cpu")
#' tensor_matmul(x, y)
tensor_matmul <- function(x, y) {
  device <- if (inherits(x, "cudatensor")) {
    x$device
  } else if (inherits(y, "cudatensor")) {
    y$device
  } else {
    "cpu"
  }
  if (!inherits(x, "cudatensor")) {
    x <- cuda_tensor(x, device = device)
  }
  .check_tensor(x)
  if (!inherits(y, "cudatensor")) {
    y <- cuda_tensor(y, device = device)
  }
  .check_tensor(y, "y")
  if (length(x$shape) != 2L || length(y$shape) != 2L) {
    stop("`x` and `y` must both be two-dimensional.", call. = FALSE)
  }
  if (x$shape[[2]] != y$shape[[1]]) {
    stop("Tensor dimensions are not conformable for matrix multiplication.",
         call. = FALSE)
  }
  transfer_needed <- !identical(x$device, y$device)
  if (transfer_needed) {
    y <- to_device(y, x$device)
  }
  if (!identical(x$backend, y$backend)) {
    y_storage <- .backend_call(
      x$backend, "from_host", to_cpu(y), y$dtype, y$shape,
      .tensor_dimnames(y)
    )
    y <- .new_cudatensor(
      y_storage, x$device, x$backend, y$dtype, y$shape,
      dimnames = .tensor_dimnames(y),
      compute_stages = list(
        backend_transfer = .tensor_stage(
          x$device, x$backend,
          requested_device = x$device,
          reason = "backend_transfer"
        )
      )
    )
  }
  result_dtype <- .promote_tensor_dtype(x$dtype, y$dtype, "matmul")
  x <- .cast_tensor(x, result_dtype)
  y <- .cast_tensor(y, result_dtype)
  result_dimnames <- .matmul_tensor_dimnames(x, y)

  storage <- .backend_call(x$backend, "matmul", x$storage, y$storage)
  .new_cudatensor(
    storage, x$device, x$backend, result_dtype,
    c(x$shape[[1]], y$shape[[2]]),
    dimnames = result_dimnames,
    compute_stages = list(
      matrix_multiply = .tensor_stage(x$device, x$backend)
    )
  )
}

.tensor_reduce <- function(x, dim, keepdim, fun, torch_method,
                           result_dtype = x$dtype) {
  .check_tensor(x)
  if (!is.logical(keepdim) || length(keepdim) != 1L || is.na(keepdim)) {
    stop("`keepdim` must be TRUE or FALSE.", call. = FALSE)
  }
  if (!is.null(dim)) {
    if (!is.numeric(dim) || anyNA(dim) || any(dim < 1) ||
        any(dim > length(x$shape)) || any(dim != as.integer(dim))) {
      stop("`dim` must contain valid one-based tensor dimensions.",
           call. = FALSE)
    }
    dim <- unique(as.integer(dim))
    if (!length(dim)) {
      return(.cast_tensor(x, result_dtype))
    }
  }
  result_dimnames <- .reduced_tensor_dimnames(x, dim, keepdim)

  source_storage <- if (!identical(result_dtype, x$dtype)) {
    .backend_call(x$backend, "cast", x$storage, result_dtype)
  } else {
    x$storage
  }
  result <- .backend_call(
    x$backend, "reduce", source_storage, dim, keepdim, torch_method
  )
  .new_cudatensor(
    result$storage, x$device, x$backend, result_dtype, result$shape,
    dimnames = result_dimnames,
    compute_stages = structure(
      list(.tensor_stage(x$device, x$backend)),
      names = torch_method
    )
  )
}

#' Tensor reductions
#'
#' @param x A `cudatensor`.
#' @param dim Optional one-based dimensions to reduce.
#' @param keepdim Whether reduced dimensions should be retained with size one.
#' @details Labels on dimensions that are not reduced are retained. A reduced
#'   dimension kept with size one retains its axis name but not its individual
#'   labels. Supplying `integer(0)` performs no reduction and returns the
#'   tensor values, shape, device, and dimnames unchanged (with the documented
#'   reduction dtype promotion).
#' @return A `cudatensor`.
#' @export
#' @examples
#' x <- cuda_tensor(matrix(1:6, 2), device = "cpu")
#' tensor_sum(x)
#' tensor_mean(x, dim = 1)
tensor_sum <- function(x, dim = NULL, keepdim = FALSE) {
  result_dtype <- if (identical(x$dtype, "integer")) "float64" else x$dtype
  .tensor_reduce(x, dim, keepdim, sum, "sum", result_dtype)
}

#' @rdname tensor_sum
#' @export
tensor_mean <- function(x, dim = NULL, keepdim = FALSE) {
  result_dtype <- if (identical(x$dtype, "integer")) "float64" else x$dtype
  .tensor_reduce(x, dim, keepdim, mean, "mean", result_dtype)
}

#' Broadcast a tensor to a compatible shape
#'
#' @param x A `cudatensor`.
#' @param shape Target dimensions. Existing dimensions are aligned from the
#'   right and must either match or equal one.
#' @details Labels are retained on dimensions whose sizes do not change.
#'   Labels are dropped from singleton dimensions that are expanded because a
#'   single input label cannot identify multiple output positions.
#' @return A `cudatensor`.
#' @export
#' @examples
#' x <- cuda_tensor(1:3, device = "cpu")
#' tensor_broadcast_to(x, c(2, 3))
tensor_broadcast_to <- function(x, shape) {
  .check_tensor(x)
  if (!is.numeric(shape) || length(shape) == 0L || anyNA(shape) ||
      any(shape < 1) || any(shape != as.integer(shape))) {
    stop("`shape` must contain positive whole-number dimensions.",
         call. = FALSE)
  }
  shape <- as.integer(shape)
  if (length(shape) < length(x$shape)) {
    stop("Target shape cannot have fewer dimensions than the tensor.",
         call. = FALSE)
  }
  padded <- c(rep(1L, length(shape) - length(x$shape)), x$shape)
  if (any(padded != 1L & padded != shape)) {
    stop("Tensor shape is not compatible with the target shape.",
         call. = FALSE)
  }
  result_dimnames <- .aligned_tensor_dimnames(x, shape)

  storage <- .backend_call(
    x$backend, "broadcast", x$storage, x$shape, shape
  )
  .new_cudatensor(
    storage, x$device, x$backend, x$dtype, shape,
    dimnames = result_dimnames,
    compute_stages = list(
      broadcast = .tensor_stage(x$device, x$backend)
    )
  )
}

.broadcast_shape <- function(x, y) {
  size <- max(length(x), length(y))
  x <- c(rep(1L, size - length(x)), x)
  y <- c(rep(1L, size - length(y)), y)
  if (any(x != y & x != 1L & y != 1L)) {
    stop("Tensor shapes are not compatible for broadcasting.", call. = FALSE)
  }
  as.integer(pmax(x, y))
}

.tensor_binary <- function(e1, e2, operator) {
  device <- if (inherits(e1, "cudatensor")) {
    e1$device
  } else if (inherits(e2, "cudatensor")) {
    e2$device
  } else {
    "cpu"
  }
  if (!inherits(e1, "cudatensor")) {
    e1 <- cuda_tensor(e1, device = device)
  }
  if (!inherits(e2, "cudatensor")) {
    e2 <- cuda_tensor(e2, device = device)
  }
  .check_tensor(e1, "e1")
  .check_tensor(e2, "e2")
  if (!identical(e1$device, e2$device)) {
    e2 <- to_device(e2, e1$device)
  }
  operation <- switch(
    operator,
    "/" = "division",
    "^" = "power",
    "arithmetic"
  )
  dtype <- .promote_tensor_dtype(e1$dtype, e2$dtype, operation)
  e1 <- .cast_tensor(e1, dtype)
  e2 <- .cast_tensor(e2, dtype)
  shape <- .broadcast_shape(e1$shape, e2$shape)
  if (!identical(e1$shape, shape)) {
    e1 <- tensor_broadcast_to(e1, shape)
  }
  if (!identical(e2$shape, shape)) {
    e2 <- tensor_broadcast_to(e2, shape)
  }
  result_dimnames <- .merge_tensor_dimnames(e1, e2)

  if (!identical(e1$backend, e2$backend)) {
    e2_storage <- .backend_call(
      e1$backend, "from_host", to_cpu(e2), e2$dtype, e2$shape,
      .tensor_dimnames(e2)
    )
    e2 <- .new_cudatensor(
      e2_storage, e1$device, e1$backend, e2$dtype, e2$shape,
      dimnames = .tensor_dimnames(e2)
    )
  }
  storage <- .backend_call(
    e1$backend, "binary", e1$storage, e2$storage, operator
  )
  .new_cudatensor(
    storage, e1$device, e1$backend, dtype, shape,
    dimnames = result_dimnames,
    compute_stages = list(
      arithmetic = .tensor_stage(e1$device, e1$backend)
    )
  )
}

#' Arithmetic operators for GPU-aware tensors
#'
#' `cudatensor` objects support element-wise `+`, `-`, `*`, `/`, and `^`.
#' Operands follow trailing-dimension broadcasting. Mixed dtypes are promoted
#' without silently truncating fractional values; integer arithmetic is
#' promoted to `float64` to avoid R integer overflow.
#' Compatible dimension labels are retained. When both operands label the same
#' non-broadcast dimension, their labels must be identical.
#'
#' Use `%*%` for matrix multiplication.
#'
#' @param e1,e2 A `cudatensor` or numeric object for element-wise
#'   arithmetic.
#' @param x,y A `cudatensor` or numeric matrix for matrix
#'   multiplication.
#' @return A `cudatensor` on the device of the tensor operand on the left (or
#'   the tensor operand on the right when the left operand is a base object).
#' @name cudatensor-operators
#' @examples
#' x <- cuda_tensor(matrix(1:6, 2, 3), device = "cpu")
#' to_cpu(x + c(0.5, 1, 1.5))
#'
#' y <- cuda_tensor(matrix(1:6, 3, 2), device = "cpu")
#' to_cpu(x %*% y)
NULL

#' @rdname cudatensor-operators
#' @export
Ops.cudatensor <- function(e1, e2) {
  operator <- .Generic
  supported <- c("+", "-", "*", "/", "^")
  if (!operator %in% supported) {
    stop(
      sprintf(
        "Operator `%s` is not supported for `cudatensor` objects.",
        operator
      ),
      call. = FALSE
    )
  }
  if (missing(e2)) {
    if (identical(operator, "+")) {
      return(e1)
    }
    if (identical(operator, "-")) {
      return(.tensor_binary(0, e1, "-"))
    }
    stop(
      sprintf(
        "Unary operator `%s` is not supported for `cudatensor` objects.",
        operator
      ),
      call. = FALSE
    )
  }
  .tensor_binary(e1, e2, operator)
}

#' @rdname cudatensor-operators
#' @export
`%*%.cudatensor` <- function(x, y) {
  tensor_matmul(x, y)
}

#' Subset and replace tensor values
#'
#' Tensor indices follow ordinary one-based R array semantics. Subsetting
#' returns a `cudatensor`, including when a single value is selected.
#' Replacement preserves the tensor dtype; fractional values therefore cannot
#' be assigned to an integer tensor.
#'
#' Backends may implement value gathering and replacement directly. The native
#' CUDA backend evaluates only R index metadata on the host and keeps tensor
#' values on the device. Compatibility backends without indexing operations
#' use a recorded CPU round trip. Subscripts containing `NA` currently use the
#' compatibility path. A replacement tensor on the same device and backend is
#' cast to the target floating dtype on that device before replacement. Integer
#' targets retain exact host validation for non-integer replacement values.
#'
#' @param x A `cudatensor`.
#' @param ... One-based R array indices.
#' @param drop Whether dimensions of length one are dropped.
#' @param value Numeric replacement values or another `cudatensor`.
#' @return A `cudatensor` on the same device as `x`.
#' @name cudatensor-subset
NULL

.tensor_subscripts <- function(...) {
  as.list(substitute(list(...)))[-1L]
}

.host_tensor_metadata <- function(x) {
  shape <- dim(x)
  labels <- dimnames(x)
  if (is.null(shape)) {
    shape <- length(x)
    value_names <- names(x)
    labels <- if (is.null(value_names)) NULL else list(value_names)
  }
  list(shape = as.integer(shape), dimnames = labels)
}

.tensor_index_plan <- function(x, subscripts, caller, drop = FALSE) {
  elements <- prod(as.double(x$shape))
  if (elements > .Machine$integer.max) {
    return(NULL)
  }
  proxy <- array(
    seq_len(as.integer(elements)),
    dim = x$shape,
    dimnames = .tensor_dimnames(x)
  )
  selected <- .tensor_eval_index(
    "[", proxy, subscripts, caller, drop = drop
  )
  metadata <- .host_tensor_metadata(selected)
  list(
    indices = as.integer(selected),
    shape = metadata$shape,
    dimnames = metadata$dimnames
  )
}

.tensor_subset_host <- function(x, subscripts, caller, drop) {
  result <- .tensor_eval_index(
    "[", to_cpu(x), subscripts, caller, drop = drop
  )
  if (!length(result)) {
    stop("Subsetting produced an empty tensor, which is not supported.",
         call. = FALSE)
  }
  metadata <- .host_tensor_metadata(result)
  output_storage <- .backend_call(
    x$backend, "from_host", result, x$dtype,
    metadata$shape, metadata$dimnames
  )
  output <- .new_cudatensor(
    output_storage, x$device, x$backend, x$dtype, metadata$shape,
    dimnames = metadata$dimnames
  )
  if (!identical(x$device, "cuda")) {
    return(.tensor_result_stage(output, "subset"))
  }
  .with_tensor_stages(
    output,
    list(
      materialization = .tensor_stage(
        "cpu", "base", output_device = "cpu", reason = "input_transfer"
      ),
      subset = .tensor_stage("cpu", "base"),
      upload = .tensor_stage(
        "cuda", x$backend, output_device = "cuda",
        requested_device = "cuda", reason = "output_transfer"
      )
    )
  )
}

.tensor_replacement_host <- function(x, subscripts, caller, value) {
  result <- .tensor_eval_index(
    "[<-", to_cpu(x), subscripts, caller, replacement = value
  )
  metadata <- .host_tensor_metadata(result)
  output_storage <- .backend_call(
    x$backend, "from_host", result, x$dtype,
    metadata$shape, metadata$dimnames
  )
  output <- .new_cudatensor(
    output_storage, x$device, x$backend, x$dtype, metadata$shape,
    dimnames = metadata$dimnames
  )
  if (!identical(x$device, "cuda")) {
    return(.tensor_result_stage(output, "replacement"))
  }
  .with_tensor_stages(
    output,
    list(
      materialization = .tensor_stage(
        "cpu", "base", output_device = "cpu", reason = "input_transfer"
      ),
      replacement = .tensor_stage("cpu", "base"),
      upload = .tensor_stage(
        "cuda", x$backend, output_device = "cuda",
        requested_device = "cuda", reason = "output_transfer"
      )
    )
  )
}

.tensor_eval_index <- function(operator, values, subscripts, caller,
                               drop = NULL, replacement = NULL) {
  arguments <- c(list(as.name(operator), quote(.tensor_values)), subscripts)
  if (!is.null(drop)) {
    arguments <- c(arguments, list(drop = drop))
  }
  if (identical(operator, "[<-")) {
    arguments <- c(arguments, list(value = quote(.tensor_replacement)))
  }
  evaluation <- new.env(parent = caller)
  evaluation$.tensor_values <- values
  evaluation$.tensor_replacement <- replacement
  eval(as.call(arguments), envir = evaluation)
}

#' @rdname cudatensor-subset
#' @export
`[.cudatensor` <- function(x, ..., drop = TRUE) {
  .check_tensor(x)
  if (!is.logical(drop) || length(drop) != 1L || is.na(drop)) {
    stop("`drop` must be TRUE or FALSE.", call. = FALSE)
  }
  subscripts <- .tensor_subscripts(...)
  caller <- parent.frame()
  plan <- .tensor_index_plan(x, subscripts, caller, drop = drop)
  direct <- !is.null(plan) && length(plan$indices) &&
    !anyNA(plan$indices) && .backend_has_operation(x$backend, "subset")
  if (!direct) {
    return(.tensor_subset_host(x, subscripts, caller, drop))
  }
  output_storage <- .backend_call(
    x$backend, "subset", x$storage, plan$indices, plan$shape
  )
  output <- .new_cudatensor(
    output_storage, x$device, x$backend, x$dtype, plan$shape,
    dimnames = plan$dimnames
  )
  .tensor_result_stage(output, "subset")
}

#' @rdname cudatensor-subset
#' @export
`[<-.cudatensor` <- function(x, ..., value) {
  .check_tensor(x)
  tensor_value <- inherits(value, "cudatensor")
  if (!tensor_value && (!is.numeric(value) || !length(value))) {
    stop("`value` must be a non-empty numeric object.", call. = FALSE)
  }
  if (tensor_value) {
    .check_tensor(value)
  }
  if (identical(x$dtype, "integer") &&
      (!tensor_value || !identical(value$dtype, "integer"))) {
    if (tensor_value) value <- to_cpu(value)
    tensor_value <- FALSE
    .validate_integer_values(value, "value")
  }
  subscripts <- .tensor_subscripts(...)
  caller <- parent.frame()
  plan <- .tensor_index_plan(x, subscripts, caller, drop = FALSE)
  can_device_replace <- !is.null(plan) && !anyNA(plan$indices) &&
    .backend_has_operation(x$backend, "replace")
  same_backend_device <- tensor_value && identical(value$backend, x$backend) &&
    identical(value$device, x$device)
  owned_cast <- FALSE
  if (can_device_replace && same_backend_device &&
      !identical(value$dtype, x$dtype)) {
    value <- .cast_tensor(value, x$dtype)
    owned_cast <- TRUE
    on.exit(.backend_call(x$backend, "release", value$storage), add = TRUE)
  }
  compatible_tensor <- tensor_value && identical(value$backend, x$backend) &&
    identical(value$device, x$device) && identical(value$dtype, x$dtype)
  direct <- can_device_replace &&
    (!tensor_value || compatible_tensor)
  if (!direct) {
    if (tensor_value) value <- to_cpu(value)
    return(.tensor_replacement_host(x, subscripts, caller, value))
  }

  value_length <- if (tensor_value) prod(value$shape) else length(value)
  if (length(plan$indices) %% value_length != 0L) {
    stop(
      "number of items to replace is not a multiple of replacement length",
      call. = FALSE
    )
  }
  if (!length(plan$indices)) {
    return(.tensor_result_stage(x, "replacement"))
  }
  replacement_positions <-
    (seq_along(plan$indices) - 1L) %% value_length + 1L
  keep <- !duplicated(plan$indices, fromLast = TRUE)
  indices <- plan$indices[keep]
  replacement_positions <- replacement_positions[keep]

  owned_replacement <- !tensor_value
  replacement_storage <- if (tensor_value) {
    value$storage
  } else {
    .backend_call(
      x$backend, "from_host", value, x$dtype,
      as.integer(length(value)), NULL
    )
  }
  if (owned_replacement) {
    on.exit(.backend_call(x$backend, "release", replacement_storage),
            add = TRUE)
  }
  output_storage <- .backend_call(
    x$backend, "replace", x$storage, indices, replacement_storage,
    replacement_positions
  )
  output <- .new_cudatensor(
    output_storage, x$device, x$backend, x$dtype, x$shape,
    dimnames = .tensor_dimnames(x)
  )
  if (!owned_cast) {
    return(.tensor_result_stage(output, "replacement"))
  }
  .with_tensor_stages(
    output,
    list(
      replacement_cast = .tensor_stage(
        x$device, x$backend, reason = "replacement_dtype_conversion"
      ),
      replacement = .tensor_stage(x$device, x$backend)
    )
  )
}

#' @export
dim.cudatensor <- function(x) {
  x$shape
}

#' Inspect tensor dimension labels
#'
#' @param x A `cudatensor`.
#' @return `NULL` for an unnamed tensor, otherwise one character vector (or
#'   `NULL`) per tensor dimension, following base R `dimnames()` semantics.
#' @export
dimnames.cudatensor <- function(x) {
  .check_tensor(x)
  .tensor_dimnames(x)
}

#' @export
length.cudatensor <- function(x) {
  prod(x$shape)
}

#' @export
as.array.cudatensor <- function(x, ...) {
  to_cpu(x)
}

#' @export
as.matrix.cudatensor <- function(x, rownames.force = NA, ...) {
  .check_tensor(x)
  if (length(x$shape) > 2L) {
    stop("Only one- or two-dimensional tensors can be converted to a matrix.",
         call. = FALSE)
  }
  values <- to_cpu(x)
  if (length(x$shape) == 1L) {
    result <- matrix(as.vector(values), ncol = 1L)
    value_dimnames <- .tensor_dimnames(x)
    if (!is.null(value_dimnames)) {
      rownames(result) <- value_dimnames[[1L]]
    }
    return(result)
  }
  as.matrix(values, rownames.force = rownames.force, ...)
}

#' @export
t.cudatensor <- function(x) {
  .check_tensor(x)
  if (length(x$shape) != 2L) {
    stop("`t()` requires a two-dimensional tensor.", call. = FALSE)
  }
  shape <- rev(x$shape)
  tensor_dimnames <- .tensor_dimnames(x)
  result_dimnames <- if (is.null(tensor_dimnames)) {
    NULL
  } else {
    rev(tensor_dimnames)
  }
  storage <- .backend_call(x$backend, "transpose", x$storage)
  .new_cudatensor(
    storage,
    x$device,
    x$backend,
    x$dtype,
    shape,
    dimnames = result_dimnames,
    compute_stages = list(
      transpose = .tensor_stage(x$device, x$backend)
    )
  )
}

#' @export
print.cudatensor <- function(x, ...) {
  cat(
    sprintf(
      "<cudatensor[%s] device=%s backend=%s dtype=%s>\n",
      paste(x$shape, collapse = "x"),
      x$device,
      x$backend,
      x$dtype
    )
  )
  max_values <- getOption("cudaverse.max_print", 100L)
  if (!is.numeric(max_values) || length(max_values) != 1L ||
      is.na(max_values) || !is.finite(max_values) || max_values < 0) {
    max_values <- 100L
  }
  if (length(x) <= max_values) {
    print(to_cpu(x), ...)
  } else {
    cat(
      sprintf(
        "<%s values omitted; use `to_cpu()` to materialize>\n",
        format(length(x), big.mark = ",", scientific = FALSE)
      )
    )
  }
  invisible(x)
}
