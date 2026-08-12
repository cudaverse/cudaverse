.check_sparse <- function(x, argument = "x") {
  if (!inherits(x, "cudasparse")) {
    stop(sprintf("`%s` must be a `cudasparse` matrix.", argument),
         call. = FALSE)
  }
  invisible(x)
}

.sparse_provenance <- function(stages) {
  provenance <- cuda_provenance(stages)
  list(
    provenance_schema = attr(provenance, "schema", exact = TRUE),
    compute_device = attr(provenance, "compute_device", exact = TRUE),
    compute_stages = attr(provenance, "compute_stages", exact = TRUE)
  )
}

.with_sparse_provenance <- function(x, stages) {
  metadata <- .sparse_provenance(stages)
  if (is.list(x) && !methods::is(x, "Matrix")) {
    x$provenance_schema <- metadata$provenance_schema
    x$compute_device <- metadata$compute_device
    x$compute_stages <- metadata$compute_stages
    return(x)
  }
  attr(x, "provenance_schema") <- metadata$provenance_schema
  attr(x, "compute_device") <- metadata$compute_device
  attr(x, "compute_stages") <- metadata$compute_stages
  x
}

.sparse_inherited_stage <- function(device, backend, output_device = device,
                                    reason = "inherited_device") {
  cuda_stage(
    requested_device = "inherited",
    device = device,
    backend = backend,
    selection_reason = reason,
    fallback = FALSE,
    output_device = output_device
  )
}

.validate_sparse_dimnames <- function(value, shape, argument = "dimnames") {
  if (is.null(value)) {
    return(NULL)
  }
  if (!is.list(value) || length(value) != length(shape)) {
    stop(
      sprintf(
        "`%s` must be NULL or a list with one entry per sparse dimension.",
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
          "Each non-NULL `%s` entry must be a character vector matching its sparse dimension.",
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

.sparse_dimnames <- function(x) {
  .validate_sparse_dimnames(x$dimnames, x$shape)
}

.sparse_axis_name <- function(value, index) {
  if (is.null(value) || is.null(names(value))) {
    return(NULL)
  }
  names(value)[[index]]
}

.meaningful_sparse_axis_name <- function(value) {
  !is.null(value) && (is.na(value) || nzchar(value))
}

.sparse_product_dimnames <- function(x, y_dimnames) {
  x_dimnames <- .sparse_dimnames(x)
  x_inner <- if (is.null(x_dimnames)) NULL else x_dimnames[[2L]]
  y_inner <- if (is.null(y_dimnames)) NULL else y_dimnames[[1L]]
  if (!is.null(x_inner) && !is.null(y_inner) &&
      !identical(x_inner, y_inner)) {
    stop(
      "Sparse and dense inner dimension names are incompatible.",
      call. = FALSE
    )
  }

  labels <- list(
    if (is.null(x_dimnames)) NULL else x_dimnames[[1L]],
    if (is.null(y_dimnames)) NULL else y_dimnames[[2L]]
  )
  x_axis <- .sparse_axis_name(x_dimnames, 1L)
  y_axis <- .sparse_axis_name(y_dimnames, 2L)
  has_axis_names <- .meaningful_sparse_axis_name(x_axis) ||
    .meaningful_sparse_axis_name(y_axis)
  if (has_axis_names) {
    names(labels) <- c(
      if (.meaningful_sparse_axis_name(x_axis)) x_axis else "",
      if (.meaningful_sparse_axis_name(y_axis)) y_axis else ""
    )
  }
  if (all(vapply(labels, is.null, logical(1))) && !has_axis_names) {
    return(NULL)
  }
  labels
}

.triplet_matrix <- function(x) {
  Matrix::sparseMatrix(
    i = x$i,
    j = x$j,
    x = x$values,
    dims = x$shape,
    dimnames = .sparse_dimnames(x),
    giveCsparse = TRUE
  )
}

.sparse_margin_sums <- function(x, margin) {
  groups <- if (margin == 0L) x$i else x$j
  group_count <- x$shape[[margin + 1L]]
  result <- numeric(group_count)
  if (!length(x$values)) return(result)

  grouped <- rowsum(
    matrix(x$values, ncol = 1L),
    group = groups,
    reorder = FALSE
  )
  result[as.integer(rownames(grouped))] <- grouped[, 1L]
  result
}

#' Create a GPU-aware sparse matrix
#'
#' @param x A numeric matrix, a sparse matrix from the `Matrix` package, or a
#'   `cudasparse` object.
#' @param format Logical storage format, `"csr"` or `"coo"`.
#' @param device One of `"auto"`, `"cuda"`, or `"cpu"`.
#' @param drop_zeros Whether to remove explicitly stored zeros.
#' @details Existing `cudasparse` inputs use their stable sorted COO mirror
#'   directly. Same-device format changes share backend storage; transfers and
#'   zero filtering do not construct an intermediate Matrix object.
#'
#' @return A `cudasparse` list. Stable public metadata include one-based COO
#'   `i` and `j`, numeric `values`, zero-based CSR `row_ptr` and `col_index`,
#'   integer `shape`, matrix `dimnames`, logical `format`, actual `device`, and
#'   `backend`. `storage` is backend-internal and should not be accessed
#'   directly.
#' @export
#' @examples
#' library(Matrix)
#' x <- rsparsematrix(5, 4, density = 0.25)
#' cuda_sparse(x, device = "cpu")
cuda_sparse <- function(x, format = c("csr", "coo"),
                        device = c("auto", "cuda", "cpu"),
                        drop_zeros = TRUE) {
  format <- match.arg(format)
  requested_device <- match.arg(device)
  selection <- cuda_select_device(requested_device)
  device <- selection$device
  if (!is.logical(drop_zeros) || length(drop_zeros) != 1L ||
      is.na(drop_zeros)) {
    stop("`drop_zeros` must be TRUE or FALSE.", call. = FALSE)
  }
  sparse_source <- inherits(x, "cudasparse")
  if (inherits(x, "cudasparse")) {
    .check_sparse(x)
    input_dimnames <- .sparse_dimnames(x)
    can_reuse <- identical(x$device, device) &&
      (!isTRUE(drop_zeros) || !any(x$values == 0))
    if (can_reuse) {
      return(.sparse_reformat(x, format))
    }
    shape <- as.integer(x$shape)
    i <- as.integer(x$i)
    j <- as.integer(x$j)
    values <- as.numeric(x$values)
    if (isTRUE(drop_zeros) && any(values == 0)) {
      retained <- values != 0
      i <- i[retained]
      j <- j[retained]
      values <- values[retained]
    }
  } else {
    input_dimnames <- NULL
  }
  if (!sparse_source && !(is.matrix(x) || methods::is(x, "Matrix"))) {
    stop("`x` must be a numeric matrix or a `Matrix` sparse matrix.",
         call. = FALSE)
  }
  if (!sparse_source && is.matrix(x) && !is.numeric(x)) {
    stop("`x` must contain numeric values.", call. = FALSE)
  }
  if (!sparse_source && is.null(input_dimnames)) {
    input_dimnames <- .validate_sparse_dimnames(
      dimnames(x),
      dim(x),
      "dimnames(x)"
    )
  }
  if (!sparse_source) {
    sparse <- if (methods::is(x, "Matrix")) {
      converted <- tryCatch(
        methods::as(x, "dMatrix"),
        error = function(...) NULL
      )
      if (is.null(converted)) {
        stop("`x` must contain numeric values.", call. = FALSE)
      }
      methods::as(converted, "generalMatrix")
    } else {
      Matrix::Matrix(x, sparse = TRUE)
    }
    sparse <- methods::as(methods::as(sparse, "dMatrix"), "generalMatrix")
    if (isTRUE(drop_zeros)) {
      sparse <- Matrix::drop0(sparse)
    }
    entries <- Matrix::summary(sparse)
    if (NROW(entries) == 0L) {
      i <- j <- integer()
      values <- numeric()
    } else {
      order_index <- order(entries$i, entries$j)
      i <- as.integer(entries$i[order_index])
      j <- as.integer(entries$j[order_index])
      values <- as.numeric(entries$x[order_index])
    }
    shape <- as.integer(dim(sparse))
  }
  if (anyNA(values) || any(!is.finite(values))) {
    stop("`x` must contain finite, non-missing values.", call. = FALSE)
  }
  input_dimnames <- .validate_sparse_dimnames(
    input_dimnames,
    shape,
    "dimnames(x)"
  )
  row_ptr <- c(0L, cumsum(tabulate(i, nbins = shape[[1]])))
  col_index <- j - 1L

  backend_id <- if (is.null(selection$backend)) {
    if (identical(device, "cuda")) "torch" else "base"
  } else {
    selection$backend
  }
  storage <- .backend_call(
    backend_id, "sparse_from_coo", i, j, values, shape, format
  )
  backend <- if (device == "cuda") {
    if (identical(backend_id, "torch")) "torch-coo" else backend_id
  } else {
    "Matrix"
  }

  result <- structure(
    list(
      i = i,
      j = j,
      values = values,
      row_ptr = as.integer(row_ptr),
      col_index = as.integer(col_index),
      shape = shape,
      dimnames = input_dimnames,
      format = format,
      device = device,
      backend = backend,
      backend_id = backend_id,
      storage = storage
    ),
    class = "cudasparse"
  )
  .with_sparse_provenance(
    result,
    list(
      sparse_materialization = cuda_stage(
        requested_device = selection$requested_device,
        device = selection$device,
        backend = backend,
        selection_reason = selection$selection_reason,
        fallback = selection$fallback,
        output_device = selection$device
      )
    )
  )
}

#' Inspect sparse matrix metadata
#'
#' @param x A `cudasparse` matrix.
#' @return A named list containing `shape`, `nnz`, `density`, `format`,
#'   actual `device`, and `backend`.
#' @export
#' @examples
#' sparse_info(cuda_sparse(diag(3), device = "cpu"))
sparse_info <- function(x) {
  .check_sparse(x)
  list(
    shape = x$shape,
    nnz = length(x$values),
    density = length(x$values) / prod(x$shape),
    format = x$format,
    device = x$device,
    backend = x$backend,
    provenance_schema = x$provenance_schema,
    compute_device = x$compute_device
  )
}

#' Inspect sparse matrix dimension labels
#'
#' @param x A `cudasparse` matrix.
#' @return `NULL` for an unnamed matrix, otherwise its row and column names,
#'   following base R `dimnames()` semantics.
#' @export
dimnames.cudasparse <- function(x) {
  .check_sparse(x)
  .sparse_dimnames(x)
}

.sparse_reformat <- function(x, format) {
  .check_sparse(x)
  if (identical(x$format, format)) return(x)
  backend_id <- if (is.null(x$backend_id)) "base" else x$backend_id
  if (typeof(x$storage) == "externalptr" &&
      .backend_has_operation(backend_id, "sparse_share")) {
    x$storage <- .backend_call(backend_id, "sparse_share", x$storage)
  }
  x$format <- format
  x
}

#' Convert sparse storage format
#'
#' @param x A `cudasparse` matrix.
#' @return A `cudasparse` matrix.
#' @export
#' @examples
#' x <- cuda_sparse(diag(3), device = "cpu")
#' as_coo(x)
#' as_csr(x)
as_coo <- function(x) {
  .sparse_reformat(x, "coo")
}

#' @rdname as_coo
#' @export
as_csr <- function(x) {
  .sparse_reformat(x, "csr")
}

#' Transpose a GPU-aware sparse matrix
#'
#' Swaps sparse rows and columns while preserving stored values, logical
#' format, dimension labels, and the actual device. The native CUDA backend
#' transposes its CSR backing storage on the device. Compatibility backends
#' rebuild same-device storage from the stable public COO metadata.
#'
#' @param x A `cudasparse` matrix.
#' @return A transposed `cudasparse` matrix on the same device as `x`.
#' @method t cudasparse
#' @export
#' @examples
#' x <- cuda_sparse(matrix(c(1, 0, 2, 0, 3, 0), 2), device = "cpu")
#' t(x)
t.cudasparse <- function(x) {
  .check_sparse(x)
  order_index <- order(x$j, x$i)
  transposed_i <- as.integer(x$j[order_index])
  transposed_j <- as.integer(x$i[order_index])
  transposed_values <- as.numeric(x$values[order_index])
  transposed_shape <- as.integer(rev(x$shape))
  transposed_row_ptr <- as.integer(c(
    0L,
    cumsum(tabulate(transposed_i, nbins = transposed_shape[[1L]]))
  ))
  backend_id <- if (is.null(x$backend_id)) "base" else x$backend_id
  storage <- if (.backend_has_operation(backend_id, "sparse_transpose")) {
    .backend_call(backend_id, "sparse_transpose", x$storage)
  } else {
    .backend_call(
      backend_id,
      "sparse_from_coo",
      transposed_i,
      transposed_j,
      transposed_values,
      transposed_shape,
      x$format
    )
  }
  source_dimnames <- .sparse_dimnames(x)
  output <- x
  output$i <- transposed_i
  output$j <- transposed_j
  output$values <- transposed_values
  output$row_ptr <- transposed_row_ptr
  output$col_index <- transposed_j - 1L
  output$shape <- transposed_shape
  output$dimnames <- if (is.null(source_dimnames)) {
    NULL
  } else {
    rev(source_dimnames)
  }
  output$storage <- storage
  .with_sparse_provenance(
    output,
    list(
      sparse_transpose = .sparse_inherited_stage(
        device = x$device,
        backend = x$backend,
        output_device = x$device
      )
    )
  )
}

#' Convert to an R sparse matrix
#'
#' @param x A `cudasparse` matrix.
#' @return A `Matrix::dgCMatrix`.
#' @export
#' @examples
#' to_dgCMatrix(cuda_sparse(diag(3), device = "cpu"))
to_dgCMatrix <- function(x) {
  .check_sparse(x)
  backend_id <- if (is.null(x$backend_id)) "base" else x$backend_id
  host <- if (typeof(x$storage) == "externalptr" &&
              .backend_has_operation(backend_id, "sparse_to_host")) {
    .backend_call(backend_id, "sparse_to_host", x$storage)
  } else {
    list(i = x$i, j = x$j, values = x$values, shape = x$shape)
  }
  result <- Matrix::sparseMatrix(
    i = host$i,
    j = host$j,
    x = host$values,
    dims = host$shape,
    dimnames = .sparse_dimnames(x),
    giveCsparse = TRUE
  )
  result <- methods::as(result, "dgCMatrix")
  .with_sparse_provenance(
    result,
    list(
      sparse_materialization = .sparse_inherited_stage(
        device = "cpu",
        backend = "Matrix",
        output_device = "cpu",
        reason = "explicit_materialization"
      )
    )
  )
}

#' Sparse matrix by dense matrix multiplication
#'
#' @param x A `cudasparse` matrix.
#' @param y A numeric matrix or `cudatensor`.
#' @return A dense `cudatensor`. The native backend keeps the result on CUDA;
#'   compatibility backends retain their existing portable CPU result.
#' @export
#' @examples
#' x <- cuda_sparse(diag(3), device = "cpu")
#' sparse_matmul_dense(x, matrix(1:6, 3, 2))
sparse_matmul_dense <- function(x, y) {
  .check_sparse(x)
  y_device <- "cpu"
  y_storage <- NULL
  if (inherits(y, "cudatensor")) {
    .check_tensor(y, "y")
    y_device <- y$device
    y_shape <- tensor_shape(y)
    y_dimnames <- .tensor_dimnames(y)
    backend_id <- if (is.null(x$backend_id)) {
      if (identical(x$backend, "torch-coo")) "torch" else "base"
    } else {
      x$backend_id
    }
    resident_native <- identical(backend_id, "native") &&
      identical(y$backend, "native") && identical(y$device, "cuda")
    if (resident_native) {
      y_native <- if (identical(y$dtype, "float64")) {
        y
      } else {
        .cast_tensor(y, "float64")
      }
      y_storage <- y_native$storage
      y_cpu <- NULL
    } else {
      y_cpu <- to_cpu(y)
    }
  } else {
    y_cpu <- y
    y_shape <- dim(y)
    y_dimnames <- dimnames(y)
  }
  valid_host <- is.null(y_cpu) ||
    (is.numeric(y_cpu) && !anyNA(y_cpu) && all(is.finite(y_cpu)))
  if (!valid_host || length(y_shape) != 2L) {
    stop("`y` must be a finite numeric matrix or two-dimensional tensor.",
         call. = FALSE)
  }
  if (x$shape[[2]] != y_shape[[1]]) {
    stop("Sparse and dense dimensions are not conformable.",
         call. = FALSE)
  }
  y_dimnames <- .validate_sparse_dimnames(
    y_dimnames,
    y_shape,
    "dimnames(y)"
  )
  result_dimnames <- .sparse_product_dimnames(x, y_dimnames)

  backend_id <- if (!is.null(x$backend_id)) {
    x$backend_id
  } else if (identical(x$backend, "torch-coo")) {
    "torch"
  } else {
    "base"
  }
  result <- .backend_call(
    backend_id,
    "sparse_matmul_dense",
    x$storage,
    x$i,
    x$j,
    x$values,
    x$shape,
    y_cpu,
    y_storage,
    y_shape
  )
  device_resident <- is.list(result) &&
    isTRUE(result$device_resident) &&
    typeof(result$storage) == "externalptr"
  output <- if (device_resident) {
    .new_cudatensor(
      result$storage,
      device = "cuda",
      backend = backend_id,
      dtype = result$dtype,
      shape = result$shape,
      dimnames = result_dimnames,
      compute_stages = list(
        sparse_multiply = .tensor_stage("cuda", backend_id)
      )
    )
  } else {
    result <- as.matrix(result)
    dim(result) <- c(x$shape[[1]], y_shape[[2]])
    dimnames(result) <- result_dimnames
    cuda_tensor(result, device = "cpu", dtype = "float64")
  }
  stages <- list()
  if (identical(y_device, "cuda") && !device_resident) {
    stages$dense_input_materialization <- .sparse_inherited_stage(
      device = "cpu",
      backend = "base",
      output_device = "cpu",
      reason = "input_transfer"
    )
  }
  stages$sparse_multiply <- .sparse_inherited_stage(
    device = x$device,
    backend = x$backend,
    output_device = if (device_resident) "cuda" else "cpu"
  )
  if (identical(x$device, "cuda") && !device_resident) {
    stages$result_materialization <- .sparse_inherited_stage(
      device = "cpu",
      backend = "base",
      output_device = "cpu",
      reason = "output_transfer"
    )
  }
  .with_sparse_provenance(output, stages)
}

#' Sparse matrix-vector multiplication
#'
#' @param x A `cudasparse` matrix.
#' @param y A numeric vector.
#' @return A numeric vector.
#' @export
#' @examples
#' sparse_matvec(cuda_sparse(diag(3), device = "cpu"), 1:3)
sparse_matvec <- function(x, y) {
  .check_sparse(x)
  if (!is.numeric(y) || is.matrix(y) || length(y) != x$shape[[2]] ||
      anyNA(y) || any(!is.finite(y))) {
    stop("`y` must be a finite numeric vector with one value per column.",
         call. = FALSE)
  }
  dense <- matrix(
    as.numeric(y),
    ncol = 1L,
    dimnames = list(names(y), NULL)
  )
  product <- sparse_matmul_dense(x, dense)
  result <- as.vector(to_cpu(product))
  product_stages <- attr(
    cuda_provenance(product),
    "compute_stages",
    exact = TRUE
  )
  sparse_dimnames <- .sparse_dimnames(x)
  if (!is.null(sparse_dimnames) && !is.null(sparse_dimnames[[1L]])) {
    names(result) <- sparse_dimnames[[1L]]
  }
  .with_sparse_provenance(result, product_stages)
}

#' Sparse row and column reductions
#'
#' @param x A `cudasparse` matrix.
#' @return A numeric vector.
#' @export
#' @examples
#' x <- cuda_sparse(matrix(1:6, 2), device = "cpu")
#' sparse_row_sums(x)
#' sparse_col_sums(x)
sparse_row_sums <- function(x) {
  .sparse_reduce_margin(x, 0L, "row_reduction")
}

#' @rdname sparse_row_sums
#' @export
sparse_col_sums <- function(x) {
  .sparse_reduce_margin(x, 1L, "column_reduction")
}

.sparse_reduce_margin <- function(x, margin, stage_name) {
  .check_sparse(x)
  backend_id <- if (is.null(x$backend_id)) "base" else x$backend_id
  native <- identical(x$device, "cuda") &&
    .backend_has_operation(backend_id, "sparse_reduce")
  result <- if (native) {
    as.numeric(.backend_call(backend_id, "sparse_reduce", x$storage, margin))
  } else {
    .sparse_margin_sums(x, margin)
  }
  sparse_dimnames <- .sparse_dimnames(x)
  axis <- margin + 1L
  if (!is.null(sparse_dimnames) && !is.null(sparse_dimnames[[axis]])) {
    names(result) <- sparse_dimnames[[axis]]
  }
  stages <- list()
  if (identical(x$device, "cuda") && !native) {
    stages$source_materialization <- .sparse_inherited_stage(
      device = "cpu",
      backend = "Matrix",
      output_device = "cpu",
      reason = "metadata_materialization"
    )
  }
  stages[[stage_name]] <- cuda_stage(
    requested_device = if (native) "inherited" else "fixed-cpu",
    device = if (native) "cuda" else "cpu",
    backend = if (native) backend_id else "Matrix",
    selection_reason = if (native) "inherited_device" else "algorithm_cpu_only",
    output_device = "cpu"
  )
  .with_sparse_provenance(result, stages)
}

#' Normalize sparse rows or columns without densifying
#'
#' Each selected row or column is divided by its sum and multiplied by
#' `scale_factor`. Optionally, `log1p()` is applied to stored non-zero values.
#' The operation preserves sparse structure and dimension labels.
#' The native CUDA backend retains normalized storage on the device and updates
#' the public host COO mirror from metadata already held by the object. It does
#' not download the normalized values or the complete margin-sum vector; only
#' a small device-validation flag crosses back before the result is returned.
#' Native results share immutable sparse index storage with their source while
#' retaining independent value storage and release-safe ownership.
#'
#' @param x A non-negative `cudasparse` matrix.
#' @param margin Normalize `"rows"` or `"columns"`.
#' @param scale_factor Positive target sum before the optional log transform.
#' @param log1p Whether to apply `log1p()` to normalized stored values.
#' @return A `cudasparse` matrix on the same device as `x`.
#' @export
#' @examples
#' x <- cuda_sparse(matrix(c(1, 0, 3, 2), 2), device = "cpu")
#' sparse_normalize(x, margin = "rows", scale_factor = 1)
sparse_normalize <- function(x, margin = c("rows", "columns"),
                             scale_factor = 1, log1p = FALSE) {
  .check_sparse(x)
  margin <- match.arg(margin)
  margin_index <- if (identical(margin, "rows")) 0L else 1L
  if (!is.numeric(scale_factor) || length(scale_factor) != 1L ||
      is.na(scale_factor) || !is.finite(scale_factor) || scale_factor <= 0) {
    stop("`scale_factor` must be one positive finite number.", call. = FALSE)
  }
  if (!is.logical(log1p) || length(log1p) != 1L || is.na(log1p)) {
    stop("`log1p` must be TRUE or FALSE.", call. = FALSE)
  }
  if (any(x$values < 0)) {
    stop("Sparse normalization requires non-negative values.", call. = FALSE)
  }
  sums <- .sparse_margin_sums(x, margin_index)
  if (any(!is.finite(sums)) || any(sums <= 0)) {
    stop("Every normalized sparse margin must have a positive finite sum.",
         call. = FALSE)
  }
  groups <- if (margin_index == 0L) x$i else x$j
  values <- x$values * scale_factor / sums[groups]
  if (isTRUE(log1p)) values <- base::log1p(values)

  backend_id <- if (is.null(x$backend_id)) "base" else x$backend_id
  native <- identical(x$device, "cuda") &&
    .backend_has_operation(backend_id, "sparse_normalize")
  if (native) {
    normalized <- .backend_call(
      backend_id,
      "sparse_normalize",
      x$storage,
      margin_index,
      scale_factor,
      log1p
    )
    output <- x
    output$storage <- normalized$storage
    output$values <- values
  } else {
    output <- x
    output$values <- values
    output$storage <- .backend_call(
      backend_id,
      "sparse_from_coo",
      output$i,
      output$j,
      output$values,
      output$shape,
      output$format
    )
  }
  stages <- list(
    normalization = cuda_stage(
      requested_device = "inherited",
      device = if (native) "cuda" else "cpu",
      backend = if (native) backend_id else "Matrix",
      selection_reason = if (native) {
        "inherited_device"
      } else {
        "algorithm_cpu_only"
      },
      fallback = FALSE,
      output_device = if (native) "cuda" else "cpu"
    )
  )
  if (identical(x$device, "cuda") && !native) {
    stages$normalization_upload <- cuda_stage(
      requested_device = "inherited",
      device = "cuda",
      backend = backend_id,
      selection_reason = "sparse_result_upload",
      fallback = FALSE,
      output_device = "cuda"
    )
  }
  .with_sparse_provenance(output, stages)
}

#' @export
dim.cudasparse <- function(x) {
  x$shape
}

#' @export
print.cudasparse <- function(x, ...) {
  info <- sparse_info(x)
  cat(sprintf(
    "<cudasparse[%sx%s] nnz=%s format=%s device=%s backend=%s>\n",
    info$shape[[1]], info$shape[[2]], info$nnz,
    info$format, info$device, info$backend
  ))
  max_values <- getOption("cudaverse.max_print", 100L)
  if (!is.numeric(max_values) || length(max_values) != 1L ||
      is.na(max_values) || !is.finite(max_values) || max_values < 0) {
    max_values <- 100L
  }
  if (info$nnz <= max_values) {
    print(to_dgCMatrix(x), ...)
  } else {
    cat(
      sprintf(
        "<%s stored values omitted; use `to_dgCMatrix()` to materialize>\n",
        format(info$nnz, big.mark = ",", scientific = FALSE)
      )
    )
  }
  invisible(x)
}
