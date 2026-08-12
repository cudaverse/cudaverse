.learn_matrix <- function(x, argument = "x", min_rows = 2L,
                          min_cols = 1L) {
  if (inherits(x, "cudatensor")) {
    x <- to_cpu(x)
  }
  if (!is.matrix(x) || !is.numeric(x) || nrow(x) < min_rows ||
      ncol(x) < min_cols || anyNA(x) || any(!is.finite(x))) {
    stop(
      sprintf(
        "`%s` must be a finite numeric matrix with at least %s rows and %s columns.",
        argument, min_rows, min_cols
      ),
      call. = FALSE
    )
  }
  x
}

.learn_sparse_matrix <- function(x, argument = "x", min_rows = 2L,
                                 min_cols = 1L) {
  if (!inherits(x, "cudasparse")) {
    stop(sprintf("`%s` must be a `cudasparse` matrix.", argument),
         call. = FALSE)
  }
  .check_sparse(x, argument)
  if (x$shape[[1L]] < min_rows || x$shape[[2L]] < min_cols ||
      anyNA(x$values) || any(!is.finite(x$values))) {
    stop(
      sprintf(
        "`%s` must be a finite sparse matrix with at least %s rows and %s columns.",
        argument, min_rows, min_cols
      ),
      call. = FALSE
    )
  }
  x
}

.learn_sparse_dense <- function(x) {
  as.matrix(.triplet_matrix(x))
}

.learn_sparse_constant_columns <- function(x) {
  sparse <- .triplet_matrix(x)
  rows <- x$shape[[1L]]
  sums <- as.numeric(Matrix::colSums(sparse))
  sum_squares <- as.numeric(Matrix::colSums(sparse * sparse))
  centered <- pmax(sum_squares - sums * sums / rows, 0)
  centered <= .Machine$double.eps * pmax(sum_squares, 1)
}

.learn_device <- function(device) {
  cuda_select_device(
    match.arg(device, c("auto", "cuda", "cpu"))
  )
}

.learn_selection_backend <- function(selection, cpu_backend = "base") {
  if (identical(selection$device, "cuda")) {
    if (is.null(selection$backend)) "torch" else selection$backend
  } else {
    cpu_backend
  }
}

.learn_stage <- function(selection, backend, output_device = selection$device,
                         reason = selection$selection_reason) {
  cuda_stage(
    requested_device = selection$requested_device,
    device = selection$device,
    backend = backend,
    selection_reason = reason,
    fallback = selection$fallback,
    output_device = output_device
  )
}

.learn_cpu_stage <- function(backend = "base",
                             reason = "algorithm_cpu_only") {
  cuda_stage(
    requested_device = "fixed-cpu",
    device = "cpu",
    backend = backend,
    selection_reason = reason,
    fallback = FALSE,
    output_device = "cpu"
  )
}

.learn_input_stage <- function(x) {
  resident <- attr(x, "cudaverse_native_state", exact = TRUE)
  if (is.list(resident) && identical(resident$backend, "native") &&
      typeof(resident$storage) == "externalptr") {
    return(cuda_stage(
      requested_device = "inherited",
      device = "cuda",
      backend = "native",
      selection_reason = "device_resident_input",
      fallback = FALSE,
      output_device = "cuda"
    ))
  }
  if (inherits(x, "cudasparse") && identical(x$device, "cuda")) {
    backend <- if (is.null(x$backend_id)) x$backend else x$backend_id
    return(cuda_stage(
      requested_device = "inherited",
      device = "cuda",
      backend = backend,
      selection_reason = "device_resident_input",
      fallback = FALSE,
      output_device = "cuda"
    ))
  }
  if (!inherits(x, "cudatensor") || !identical(x$device, "cuda")) {
    return(NULL)
  }
  cuda_stage(
    requested_device = "inherited",
    device = "cpu",
    backend = "base",
    selection_reason = "input_transfer",
    fallback = FALSE,
    output_device = "cpu"
  )
}

.learn_add_stage <- function(stages, name, stage) {
  if (!is.null(stage)) {
    stages[[name]] <- stage
  }
  stages
}

.with_learning_provenance <- function(x, stages, requested_device = NULL,
                                      backend = NULL, parameters = NULL,
                                      source_device = NULL,
                                      source_class = NULL) {
  provenance <- cuda_provenance(stages)
  schema <- attr(provenance, "schema", exact = TRUE)
  compute_device <- attr(provenance, "compute_device", exact = TRUE)
  stages <- attr(provenance, "compute_stages", exact = TRUE)
  if (is.list(x) && is.null(dim(x))) {
    x$provenance_schema <- schema
    if (!is.null(requested_device)) {
      x$requested_device <- requested_device
    }
    x$compute_device <- compute_device
    x$compute_stages <- stages
    if (!is.null(backend)) {
      x$backend <- backend
    }
    if (!is.null(parameters)) {
      x$parameters <- parameters
    }
    if (!is.null(source_device)) {
      x$source_device <- source_device
    }
    if (!is.null(source_class)) {
      x$source_class <- source_class
    }
    return(x)
  }
  attr(x, "provenance_schema") <- schema
  if (!is.null(requested_device)) {
    attr(x, "requested_device") <- requested_device
  }
  attr(x, "compute_device") <- compute_device
  attr(x, "compute_stages") <- stages
  if (!is.null(backend)) {
    attr(x, "backend") <- backend
  }
  if (!is.null(parameters)) {
    attr(x, "parameters") <- parameters
  }
  if (!is.null(source_device)) {
    attr(x, "source_device") <- source_device
  }
  if (!is.null(source_class)) {
    attr(x, "source_class") <- source_class
  }
  x
}

.learn_source_device <- function(x) {
  resident <- attr(x, "cudaverse_native_state", exact = TRUE)
  if (is.list(resident) && identical(resident$backend, "native") &&
      typeof(resident$storage) == "externalptr") {
    return("cuda")
  }
  if (inherits(x, "cudatensor") || inherits(x, "cudasparse")) {
    x$device
  } else {
    "cpu"
  }
}

.learn_flag <- function(value, argument) {
  if (!is.logical(value) || length(value) != 1L || is.na(value)) {
    stop(sprintf("`%s` must be TRUE or FALSE.", argument), call. = FALSE)
  }
  value
}

.learn_component_names <- function(prefix, n) {
  paste0(prefix, seq_len(n))
}

.named_pca_result <- function(result, observation_names, feature_names) {
  component_names <- .learn_component_names("PC", ncol(result$rotation))
  dimnames(result$rotation) <- list(feature_names, component_names)
  dimnames(result$x) <- list(observation_names, component_names)
  names(result$sdev) <- component_names
  if (is.numeric(result$center)) {
    names(result$center) <- feature_names
  }
  if (is.numeric(result$scale)) {
    names(result$scale) <- feature_names
  }
  structure(result, class = "cuda_pca")
}

.learn_prediction_selection <- function(device, object_device) {
  device <- match.arg(device, c("model", "auto", "cuda", "cpu"))
  if (!identical(device, "model")) {
    return(.learn_device(device))
  }
  if (!is.character(object_device) || length(object_device) != 1L ||
      is.na(object_device) || !object_device %in% c("cuda", "cpu")) {
    stop(
      "The fitted object does not contain a valid `$device` value.",
      call. = FALSE
    )
  }
  selection <- .learn_device(object_device)
  selection$requested_device <- "inherited"
  selection$selection_reason <- "model_device"
  selection$fallback <- FALSE
  selection
}

.learn_feature_summary <- function(x, limit = 5L) {
  shown <- utils::head(x, limit)
  suffix <- if (length(x) > limit) {
    sprintf(" ... (%s more)", length(x) - limit)
  } else {
    ""
  }
  paste0(paste(shown, collapse = ", "), suffix)
}

.learn_prediction_matrix <- function(newdata, feature_names, n_features) {
  values <- if (inherits(newdata, "cudatensor")) {
    newdata
  } else {
    as.matrix(newdata)
  }
  values <- .learn_matrix(
    values,
    argument = "newdata",
    min_rows = 1L,
    min_cols = 1L
  )

  if (ncol(values) != n_features) {
    stop(
      sprintf(
        "`newdata` must contain exactly %s model feature%s; it has %s.",
        n_features,
        if (n_features == 1L) "" else "s",
        ncol(values)
      ),
      call. = FALSE
    )
  }
  if (is.null(feature_names)) {
    return(values)
  }
  if (anyNA(feature_names) || length(feature_names) != n_features) {
    stop("The fitted object contains invalid feature names.", call. = FALSE)
  }

  new_names <- colnames(values)
  if (is.null(new_names)) {
    stop(
      paste0(
        "`newdata` must have column names because the model was fitted ",
        "with named features."
      ),
      call. = FALSE
    )
  }
  if (anyNA(new_names)) {
    stop("`newdata` contains invalid column names.", call. = FALSE)
  }
  if (identical(new_names, feature_names)) {
    return(values)
  }
  if (anyDuplicated(feature_names) || anyDuplicated(new_names)) {
    stop(
      paste0(
        "Duplicated feature names must match the fitted model exactly and ",
        "in the same order."
      ),
      call. = FALSE
    )
  }

  missing_features <- setdiff(feature_names, new_names)
  extra_features <- setdiff(new_names, feature_names)
  if (length(missing_features) || length(extra_features)) {
    details <- c(
      if (length(missing_features)) {
        paste0("missing: ", .learn_feature_summary(missing_features))
      },
      if (length(extra_features)) {
        paste0("unexpected: ", .learn_feature_summary(extra_features))
      }
    )
    stop(
      paste0(
        "`newdata` feature names do not match the fitted model (",
        paste(details, collapse = "; "),
        ")."
      ),
      call. = FALSE
    )
  }

  values[, match(feature_names, new_names), drop = FALSE]
}

.learn_check_prediction_dots <- function(...) {
  dots <- list(...)
  if (!length(dots)) {
    return(invisible(NULL))
  }
  dot_names <- names(dots)
  if (is.null(dot_names)) {
    dot_names <- rep("", length(dots))
  }
  dot_names[!nzchar(dot_names)] <- "<unnamed>"
  stop(
    paste0("Unused argument", if (length(dots) == 1L) "" else "s",
           " in `...`: ", paste(dot_names, collapse = ", "), "."),
    call. = FALSE
  )
}

.learn_validate_model_device <- function(object) {
  if (!is.character(object$device) || length(object$device) != 1L ||
      is.na(object$device) || !object$device %in% c("cuda", "cpu")) {
    stop(
      "The fitted object does not contain a valid `$device` value.",
      call. = FALSE
    )
  }
  invisible(object$device)
}

.learn_validate_pca <- function(object) {
  .learn_validate_model_device(object)
  rotation <- object$rotation
  if (!is.matrix(rotation) || !is.numeric(rotation) ||
      nrow(rotation) < 1L || ncol(rotation) < 1L ||
      anyNA(rotation) || any(!is.finite(rotation))) {
    stop("The fitted PCA object contains invalid loadings.", call. = FALSE)
  }
  feature_names <- rownames(rotation)
  if (!is.null(feature_names) && anyNA(feature_names)) {
    stop(
      "The fitted PCA object contains invalid feature names.",
      call. = FALSE
    )
  }
  component_names <- colnames(rotation)
  if (is.null(component_names) || anyNA(component_names) ||
      anyDuplicated(component_names)) {
    stop(
      "The fitted PCA object contains invalid component names.",
      call. = FALSE
    )
  }
  for (field in c("center", "scale")) {
    value <- object[[field]]
    valid <- identical(value, FALSE) ||
      (is.numeric(value) && length(value) == nrow(rotation) &&
       !anyNA(value) && all(is.finite(value)))
    if (!valid || (identical(field, "scale") && is.numeric(value) &&
                   any(value <= 0))) {
      stop(
        sprintf("The fitted PCA object contains an invalid `$%s` value.", field),
        call. = FALSE
      )
    }
    if (is.numeric(value) && !identical(names(value), feature_names)) {
      stop(
        sprintf(
          "The fitted PCA object's `$%s` names do not match its features.",
          field
        ),
        call. = FALSE
      )
    }
  }
  invisible(rotation)
}

.learn_validate_pca_scores <- function(object) {
  scores <- object$x
  component_names <- colnames(object$rotation)
  if (!is.matrix(scores) || !is.numeric(scores) ||
      nrow(scores) < 2L || ncol(scores) != ncol(object$rotation) ||
      anyNA(scores) || any(!is.finite(scores)) ||
      !identical(colnames(scores), component_names)) {
    stop(
      "The fitted PCA object contains invalid stored training scores.",
      call. = FALSE
    )
  }
  invisible(scores)
}

.learn_validate_kmeans <- function(object) {
  .learn_validate_model_device(object)
  centers <- object$centers
  if (!is.matrix(centers) || !is.numeric(centers) ||
      nrow(centers) < 1L || ncol(centers) < 1L ||
      anyNA(centers) || any(!is.finite(centers))) {
    stop("The fitted k-means object contains invalid centres.", call. = FALSE)
  }
  feature_names <- colnames(centers)
  if (!is.null(feature_names) && anyNA(feature_names)) {
    stop(
      "The fitted k-means object contains invalid feature names.",
      call. = FALSE
    )
  }
  center_names <- rownames(centers)
  if (!is.null(center_names) &&
      (anyNA(center_names) || anyDuplicated(center_names))) {
    stop(
      "The fitted k-means object contains invalid centre names.",
      call. = FALSE
    )
  }
  invisible(centers)
}

.learn_validate_kmeans_clusters <- function(object) {
  cluster <- object$cluster
  number_of_centers <- nrow(object$centers)
  valid <- is.integer(cluster) && length(cluster) >= 2L &&
    !anyNA(cluster) &&
    all(cluster >= 1L & cluster <= number_of_centers)
  if (!valid) {
    stop(
      "The fitted k-means object contains invalid stored assignments.",
      call. = FALSE
    )
  }
  invisible(cluster)
}

.with_preserved_seed <- function(seed, code) {
  if (is.null(seed)) {
    return(force(code))
  }
  if (!is.numeric(seed) || length(seed) != 1L || is.na(seed) ||
      !is.finite(seed)) {
    stop("`seed` must be NULL or one finite whole number.", call. = FALSE)
  }
  integer_seed <- suppressWarnings(as.integer(seed))
  if (is.na(integer_seed) || seed != integer_seed) {
    stop("`seed` must be NULL or one finite whole number.", call. = FALSE)
  }

  had_seed <- exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  if (had_seed) {
    old_seed <- get(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  }
  on.exit({
    if (had_seed) {
      assign(".Random.seed", old_seed, envir = .GlobalEnv)
    } else if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
      rm(".Random.seed", envir = .GlobalEnv)
    }
  }, add = TRUE)

  set.seed(integer_seed)
  force(code)
}

.torch_matrix <- function(x) {
  torch::torch_tensor(
    x,
    dtype = torch::torch_float64(),
    device = "cuda"
  )
}

.torch_array <- function(x) {
  as.array(x$to(device = "cpu"))
}

#' GPU-aware singular value decomposition
#'
#' @param x A finite numeric matrix or `cudatensor`.
#' @param nu,nv Number of left and right singular vectors to return.
#' @param device One of `"auto"`, `"cuda"`, or `"cpu"`.
#' @return A list with `d`, `u`, `v`, and the actual `device`. Matrix row and
#'   column names are retained on the corresponding singular vectors.
#' @export
#' @examples
#' cuda_svd(matrix(rnorm(30), 10, 3), device = "cpu")
cuda_svd <- function(x, nu = min(nrow(x), ncol(x)),
                     nv = min(nrow(x), ncol(x)),
                     device = c("auto", "cuda", "cpu")) {
  source_device <- .learn_source_device(x)
  source_class <- class(x)[[1L]]
  input_stage <- .learn_input_stage(x)
  x <- .learn_matrix(x)
  observation_names <- rownames(x)
  feature_names <- colnames(x)
  selection <- .learn_device(device)
  device <- selection$device
  rank <- min(dim(x))
  for (value in list(nu = nu, nv = nv)) {
    if (!is.numeric(value) || length(value) != 1L || is.na(value) ||
        value < 0 || value > rank || value != as.integer(value)) {
      stop("`nu` and `nv` must be whole numbers between zero and matrix rank.",
           call. = FALSE)
    }
  }
  nu <- as.integer(nu)
  nv <- as.integer(nv)

  backend <- .learn_selection_backend(selection)
  result <- .backend_call(backend, "algorithm_svd", x, nu, nv)
  singular_values <- result$d
  u <- result$u
  v <- result$v

  component_names <- .learn_component_names("SVD", length(singular_values))
  names(singular_values) <- component_names
  dimnames(u) <- list(observation_names, utils::head(component_names, nu))
  dimnames(v) <- list(feature_names, utils::head(component_names, nv))
  output <- structure(
    list(
      d = singular_values,
      u = u,
      v = v,
      device = device
    ),
    class = "cuda_svd"
  )
  stages <- .learn_add_stage(list(), "input_materialization", input_stage)
  stages$decomposition <- .learn_stage(
    selection,
    backend = backend,
    output_device = "cpu"
  )
  .with_learning_provenance(
    output,
    stages,
    requested_device = selection$requested_device,
    backend = backend,
    parameters = list(nu = nu, nv = nv),
    source_device = source_device,
    source_class = source_class
  )
}

#' GPU-aware principal component analysis
#'
#' @param x A matrix or `cudasparse` object with observations in rows and
#'   features in columns.
#' @param n_components Number of components to return.
#' @param center Whether to centre features.
#' @param scale. Whether to scale features to unit variance.
#' @param device One of `"auto"`, `"cuda"`, or `"cpu"`.
#' @return A `cuda_pca` object with scores in `x`, loadings in `rotation`,
#'   standard deviations, centring/scaling values, and actual device.
#'   Observation names, feature names, and stable `PC1`, `PC2`, ... component
#'   names are preserved on every backend.
#' @export
#' @examples
#' fit <- cuda_pca(iris[, 1:4], n_components = 2, device = "cpu")
#' fit
cuda_pca <- function(x, n_components = 2L, center = TRUE, scale. = FALSE,
                     device = c("auto", "cuda", "cpu")) {
  sparse_input <- inherits(x, "cudasparse")
  source_device <- .learn_source_device(x)
  source_class <- class(x)[[1L]]
  input_stage <- .learn_input_stage(x)
  if (sparse_input) {
    x <- .learn_sparse_matrix(x, min_cols = 2L)
    sparse_names <- .sparse_dimnames(x)
    observation_names <- if (is.null(sparse_names)) NULL else sparse_names[[1L]]
    feature_names <- if (is.null(sparse_names)) NULL else sparse_names[[2L]]
    input_shape <- x$shape
  } else {
    x <- .learn_matrix(as.matrix(x), min_cols = 2L)
    observation_names <- rownames(x)
    feature_names <- colnames(x)
    input_shape <- dim(x)
  }
  center <- .learn_flag(center, "center")
  scale. <- .learn_flag(scale., "scale.")
  selection <- .learn_device(device)
  device <- selection$device
  max_components <- min(input_shape[[1L]] - 1L, input_shape[[2L]])
  if (!is.numeric(n_components) || length(n_components) != 1L ||
      is.na(n_components) || n_components < 1 ||
      n_components > max_components ||
      n_components != as.integer(n_components)) {
    stop(
      sprintf("`n_components` must be between 1 and %s.", max_components),
      call. = FALSE
    )
  }
  constant_features <- if (sparse_input) {
    .learn_sparse_constant_columns(x)
  } else {
    apply(x, 2L, stats::sd) == 0
  }
  if (scale. && any(constant_features)) {
    stop("Cannot scale constant features.", call. = FALSE)
  }
  n_components <- as.integer(n_components)

  compute_backend <- .learn_selection_backend(selection, "base")
  backend_sparse <- sparse_input &&
    .backend_has_operation(compute_backend, "algorithm_sparse_pca")
  sparse_transferred <- FALSE
  if (backend_sparse) {
    sparse_compute <- x
    source_backend <- if (is.null(x$backend_id)) "base" else x$backend_id
    if (!identical(x$device, "cuda") ||
        !identical(source_backend, compute_backend)) {
      sparse_compute <- cuda_sparse(
        .triplet_matrix(x),
        format = x$format,
        device = "cuda"
      )
      sparse_transferred <- TRUE
    }
    fit <- .backend_call(
      compute_backend,
      "algorithm_sparse_pca",
      sparse_compute$storage,
      sparse_compute$shape,
      n_components,
      center,
      scale.
    )
  } else {
    dense_input <- if (sparse_input) .learn_sparse_dense(x) else x
    fit <- .backend_call(
      compute_backend,
      "algorithm_pca",
      dense_input,
      n_components,
      center,
      scale.
    )
  }
  output <- .named_pca_result(
    c(fit, list(device = device)),
    observation_names = observation_names,
    feature_names = feature_names
  )

  stages <- if (sparse_input && length(x$compute_stages)) {
    x$compute_stages
  } else {
    .learn_add_stage(list(), "input_materialization", input_stage)
  }
  backend <- if (identical(device, "cuda")) compute_backend else "stats"
  if (backend_sparse) {
    if (sparse_transferred) {
      stages$sparse_transfer <- .learn_stage(
        selection,
        backend = backend,
        output_device = "cuda",
        reason = "sparse_input_transfer"
      )
    }
    stages$sparse_to_dense <- .learn_stage(
      selection,
      backend = backend,
      output_device = "cuda",
      reason = "device_resident_conversion"
    )
  } else if (sparse_input) {
    stages$sparse_to_dense <- .learn_cpu_stage(
      backend = "Matrix",
      reason = "algorithm_materialization"
    )
  }
  stages$preprocessing <- .learn_stage(
    selection,
    backend = backend,
    output_device = device
  )
  stages$decomposition <- .learn_stage(
    selection,
    backend = backend,
    output_device = "cpu"
  )
  resident_scores <- attr(output$x, "cudaverse_native_state", exact = TRUE)
  if (is.list(resident_scores) && identical(resident_scores$backend, "native")) {
    stages$scores_resident <- .learn_stage(
      selection,
      backend = backend,
      output_device = "cuda",
      reason = "device_resident_output"
    )
  }
  .with_learning_provenance(
    output,
    stages,
    requested_device = selection$requested_device,
    backend = backend,
    parameters = list(
      n_components = n_components,
      center = center,
      scale = scale.
    ),
    source_device = source_device,
    source_class = source_class
  )
}

#' Project observations with a fitted CUDA-aware PCA model
#'
#' `predict.cuda_pca()` applies the fitted centring, scaling, and loadings to
#' new observations. Named features may be supplied in any order and are
#' aligned safely before projection. If the fitted model has feature names,
#' unnamed or mismatched columns are rejected instead of being used in the
#' wrong order.
#'
#' @param object A fitted `cuda_pca` object.
#' @param newdata A finite numeric matrix or data frame with observations in
#'   rows and the model features in columns. When omitted, the training scores
#'   in `object$x` are returned.
#' @param device Where to compute the projection. `"model"` reuses the actual
#'   device of the fitted model; `"auto"`, `"cuda"`, and `"cpu"` follow the
#'   usual cudaverse device-selection rules.
#' @param ... Must be empty.
#' @return A numeric matrix of component scores. New observation names and
#'   stable component names are retained. A recomputed prediction includes
#'   stage-level provenance and is materialized as an R matrix on the CPU.
#'   Omitting `newdata` returns the validated stored training scores unchanged;
#'   that retrieval does not create a prediction stage.
#' @seealso [cuda_pca()]
#' @method predict cuda_pca
#' @export
#' @examples
#' train <- as.matrix(iris[1:100, 1:4])
#' fit <- cuda_pca(train, n_components = 2, device = "cpu")
#' predict(fit, as.matrix(iris[101:105, 1:4]), device = "cpu")
predict.cuda_pca <- function(object, newdata, device = c(
                               "model", "auto", "cuda", "cpu"
                             ), ...) {
  .learn_check_prediction_dots(...)
  .learn_validate_pca(object)
  if (missing(newdata)) {
    .learn_validate_pca_scores(object)
    return(object$x)
  }

  source_device <- .learn_source_device(newdata)
  source_class <- class(newdata)[[1L]]
  input_stage <- .learn_input_stage(newdata)
  rotation <- object$rotation
  values <- .learn_prediction_matrix(
    newdata,
    feature_names = rownames(rotation),
    n_features = nrow(rotation)
  )
  selection <- .learn_prediction_selection(device, object$device)

  backend <- .learn_selection_backend(selection)
  scores <- .backend_call(
    backend,
    "algorithm_pca_predict",
    values,
    object$center,
    object$scale,
    rotation
  )

  dimnames(scores) <- list(rownames(values), colnames(rotation))
  attr(scores, "device") <- selection$device
  stages <- .learn_add_stage(list(), "input_materialization", input_stage)
  stages$projection <- .learn_stage(
    selection,
    backend = backend,
    output_device = "cpu"
  )
  .with_learning_provenance(
    scores,
    stages,
    requested_device = selection$requested_device,
    backend = backend,
    parameters = list(n_components = ncol(rotation)),
    source_device = source_device,
    source_class = source_class
  )
}

.stable_row_norm_cpu <- function(x) {
  result <- numeric(nrow(x))
  row_scale <- apply(abs(x), 1L, max)
  infinite <- is.infinite(row_scale)
  result[infinite] <- Inf
  finite_nonzero <- is.finite(row_scale) & row_scale > 0
  if (any(finite_nonzero)) {
    scaled <- x[finite_nonzero, , drop = FALSE] /
      row_scale[finite_nonzero]
    result[finite_nonzero] <- row_scale[finite_nonzero] *
      sqrt(rowSums(scaled^2))
  }
  result
}

.recompute_euclidean_pairs_cpu <- function(distance, x, y, risk) {
  pairs <- which(risk, arr.ind = TRUE)
  if (!nrow(pairs)) {
    return(distance)
  }

  # Bound the temporary direct-difference matrix to roughly one million
  # doubles. This keeps targeted recomputation predictable even when an
  # adversarial input makes many pairs numerically risky.
  pairs_per_chunk <- max(1L, floor(1e6 / ncol(x)))
  starts <- seq.int(1L, nrow(pairs), by = pairs_per_chunk)
  for (start in starts) {
    rows <- seq.int(
      start,
      length.out = min(pairs_per_chunk, nrow(pairs) - start + 1L)
    )
    selected <- pairs[rows, , drop = FALSE]
    differences <- x[selected[, 1L], , drop = FALSE] -
      y[selected[, 2L], , drop = FALSE]
    distance[selected] <- .stable_row_norm_cpu(differences)
  }
  distance
}

.euclidean_distance_cpu <- function(x, y) {
  # Euclidean distance is invariant to a common translation. Translating by
  # one observation removes large shared offsets before the fast squared-norm
  # identity is evaluated by BLAS.
  anchor <- x[1L, , drop = TRUE]
  x_translated <- sweep(x, 2L, anchor, "-")
  y_translated <- sweep(y, 2L, anchor, "-")
  input_scale <- 1
  pre_scaled <- any(!is.finite(x_translated)) ||
    any(!is.finite(y_translated))

  if (pre_scaled) {
    # Opposite extreme finite values can overflow during translation. Scaling
    # the original inputs first bounds that subtraction without changing the
    # final distance. Numerically collapsed close pairs are caught below and
    # recomputed from their direct differences.
    input_scale <- max(abs(x), abs(y))
    x_scaled <- x / input_scale
    y_scaled <- y / input_scale
    anchor <- x_scaled[1L, , drop = TRUE]
    x_translated <- sweep(x_scaled, 2L, anchor, "-")
    y_translated <- sweep(y_scaled, 2L, anchor, "-")
  }

  translation_scale <- max(abs(x_translated), abs(y_translated))
  if (translation_scale == 0) {
    return(matrix(0, nrow = nrow(x), ncol = nrow(y)))
  }
  x_work <- x_translated / translation_scale
  y_work <- y_translated / translation_scale

  x_squared_norm <- rowSums(x_work^2)
  y_squared_norm <- rowSums(y_work^2)
  cross_product <- tcrossprod(x_work, y_work)
  squared <- outer(x_squared_norm, y_squared_norm, "+") -
    2 * cross_product

  # A squared-norm identity is unreliable only when cancellation is large
  # relative to the terms being combined. Use a deliberately conservative
  # threshold, then repair just those pairs with direct, scale-first norms.
  roundoff_scale <- outer(x_squared_norm, y_squared_norm, "+") +
    2 * abs(cross_product)
  risk_ratio <- max(
    sqrt(.Machine$double.eps),
    64 * .Machine$double.eps * ncol(x)
  )
  risk <- !is.finite(squared) |
    squared <= risk_ratio * roundoff_scale

  if (identical(x, y)) {
    diagonal <- seq_len(nrow(x))
    squared[cbind(diagonal, diagonal)] <- 0
    risk[cbind(diagonal, diagonal)] <- FALSE
  }

  distance <- if (pre_scaled) {
    input_scale * (translation_scale * sqrt(pmax(squared, 0)))
  } else {
    translation_scale * sqrt(pmax(squared, 0))
  }
  risk <- risk | !is.finite(distance)
  .recompute_euclidean_pairs_cpu(distance, x, y, risk)
}

.cosine_unit_rows <- function(x, argument) {
  row_scale <- apply(abs(x), 1L, max)
  if (any(row_scale == 0)) {
    stop(
      sprintf(
        "Cosine distance is undefined for zero-length rows in `%s`.",
        argument
      ),
      call. = FALSE
    )
  }

  scaled <- x / row_scale
  scaled / sqrt(rowSums(scaled^2))
}

.row_batch_size <- function(batch_size, n) {
  integer_batch_size <- suppressWarnings(as.integer(batch_size))
  if (!is.numeric(batch_size) || length(batch_size) != 1L ||
      is.na(batch_size) || !is.finite(batch_size) ||
      is.na(integer_batch_size) || integer_batch_size < 1L ||
      batch_size != integer_batch_size) {
    stop("`batch_size` must be one positive whole number.", call. = FALSE)
  }
  min(integer_batch_size, n)
}

.distance_backend_blocks <- function(backend, x, y, metric,
                                     source_x, source_y, batch_size) {
  if (.backend_has_operation(backend, "algorithm_distance_batched")) {
    return(.backend_call(
      backend, "algorithm_distance_batched",
      x, y, metric, batch_size, source_x, source_y
    ))
  }
  if (batch_size == nrow(x)) {
    return(.backend_call(
      backend, "algorithm_distance", x, y, metric, source_x, source_y
    ))
  }

  result <- matrix(NA_real_, nrow = nrow(x), ncol = nrow(y))
  starts <- seq.int(1L, nrow(x), by = batch_size)
  for (start in starts) {
    rows <- seq.int(
      start,
      length.out = min(batch_size, nrow(x) - start + 1L)
    )
    query <- x[rows, , drop = FALSE]
    block <- .backend_call(
      backend, "algorithm_distance", query, y, metric, query, source_y
    )
    result[rows, ] <- matrix(
      block, nrow = length(rows), ncol = nrow(y)
    )
  }
  result
}

#' Pairwise distances with an optional CUDA backend
#'
#' @param x,y Numeric matrices with observations in rows. When `y` is `NULL`,
#'   computes all pairwise distances within `x`.
#' @param metric `"euclidean"` or `"cosine"`.
#' @param device One of `"auto"`, `"cuda"`, or `"cpu"`.
#' @param batch_size Maximum number of query rows in each compute block. The
#'   final dense result is still allocated in host memory.
#' @return A dense numeric distance matrix with a `device` attribute. Input
#'   observation names are retained as row and column names when present.
#' @details On CPU, Euclidean distances use a common translation and global
#'   scaling before a vectorized calculation. Pairs at risk of cancellation or
#'   non-finite intermediate results are recomputed from direct observation
#'   differences with a scale-first norm. This avoids cancellation from large
#'   shared offsets and avoids avoidable overflow and underflow for extreme
#'   finite values. All built-in backends honor `batch_size`. The native CUDA
#'   backend uploads each input once, keeps the reference matrix and its norms
#'   device-resident, and transfers only completed distance blocks to R. This
#'   bounds operation-owned device memory without silently changing backend.
#' @export
#' @examples
#' cuda_distance(matrix(1:12, 4, 3), device = "cpu")
cuda_distance <- function(x, y = NULL,
                          metric = c("euclidean", "cosine"),
                          device = c("auto", "cuda", "cpu"),
                          batch_size = 256L) {
  source_device <- .learn_source_device(x)
  source_class <- class(x)[[1L]]
  input_x_stage <- .learn_input_stage(x)
  input_y_stage <- if (is.null(y)) NULL else .learn_input_stage(y)
  x <- .learn_matrix(x, min_rows = 1L)
  x_names <- rownames(x)
  self <- is.null(y)
  if (self) {
    y <- x
    y_names <- x_names
  } else {
    y <- .learn_matrix(y, "y", min_rows = 1L)
    y_names <- rownames(y)
  }
  if (ncol(x) != ncol(y)) {
    stop("`x` and `y` must have the same number of columns.", call. = FALSE)
  }
  batch_size <- .row_batch_size(batch_size, nrow(x))
  metric <- match.arg(metric)
  if (metric == "cosine") {
    x_unit <- .cosine_unit_rows(x, "x")
    y_unit <- if (self) x_unit else .cosine_unit_rows(y, "y")
  }
  selection <- .learn_device(device)
  device <- selection$device

  backend <- .learn_selection_backend(selection)
  distance <- .distance_backend_blocks(
    backend,
    if (metric == "cosine") x_unit else x,
    if (metric == "cosine") y_unit else y,
    metric,
    x,
    y,
    batch_size
  )
  if (metric == "cosine") {
    distance <- pmin(pmax(distance, 0), 2)
  }
  if (self) {
    # GPU distance kernels can leave round-off-sized values on the diagonal.
    # Self-distance is exactly zero by contract; enforcing it also prevents
    # downstream bandwidth estimators from treating the diagonal as data.
    diag(distance) <- 0
  }
  if (!is.null(x_names) || !is.null(y_names)) {
    dimnames(distance) <- list(x_names, y_names)
  }
  attr(distance, "device") <- device
  stages <- .learn_add_stage(list(), "input_x_materialization", input_x_stage)
  stages <- .learn_add_stage(stages, "input_y_materialization", input_y_stage)
  stages$distance <- .learn_stage(
    selection,
    backend = backend,
    output_device = "cpu"
  )
  .with_learning_provenance(
    distance,
    stages,
    requested_device = selection$requested_device,
    backend = backend,
    parameters = list(
      metric = metric,
      batch_size = batch_size,
      batches = as.integer(ceiling(nrow(x) / batch_size))
    ),
    source_device = source_device,
    source_class = source_class
  )
}

.knn_distance_state <- function(x, metric, selection = NULL,
                                cosine_values = NULL, device = NULL) {
  if (is.null(selection)) {
    selection <- .learn_device(if (is.null(device)) "cpu" else device)
  }
  values <- if (metric == "cosine") {
    if (is.null(cosine_values)) {
      .cosine_unit_rows(x, "x")
    } else {
      cosine_values
    }
  } else {
    x
  }
  backend <- .learn_selection_backend(selection)
  storage <- .backend_call(
    backend, "algorithm_knn_prepare", values, metric, x
  )
  list(
    values = values,
    storage = storage,
    metric = metric,
    device = selection$device,
    backend = backend
  )
}

.knn_distance_block <- function(state, rows) {
  distance <- .backend_call(
    state$backend,
    "algorithm_knn_block",
    state$storage,
    state$values,
    rows,
    state$metric
  )

  distance <- matrix(
    distance,
    nrow = length(rows),
    ncol = nrow(state$values)
  )
  if (state$metric == "cosine") {
    distance <- pmin(pmax(distance, 0), 2)
  }
  distance
}

#' k-nearest neighbours
#'
#' @param x Numeric matrix or `cudasparse` object with observations in rows.
#' @param k Number of neighbours.
#' @param metric Exact distance metric, `"euclidean"` or `"cosine"`.
#' @param device One of `"auto"`, `"cuda"`, or `"cpu"`.
#' @param batch_size Maximum number of query rows in each dense distance block.
#'   Larger batches may be faster but use more memory.
#' @return A `cuda_knn` list with `index` and `distance` matrices of size
#'   `nrow(x)` by `k`, followed by the selected `metric` and actual `device`.
#'   Neighbours in every row are ordered by distance and then row index.
#'   When `x` has row names, both matrices retain them as query identifiers;
#'   neighbour identities can be recovered with
#'   `rownames(result$index)[result$index]`.
#'
#' @details
#' Neighbours are exact: every row is compared with every other row. The
#' observation itself is always excluded. Equal distances are resolved
#' deterministically in favour of the smaller row index.
#'
#' The implementation constructs at most a
#' `min(batch_size, nrow(x))`-by-`nrow(x)` dense distance block instead of a
#' complete pairwise distance matrix. The native CUDA backend keeps distance
#' blocks and deterministic top-k selection on the GPU, then transfers only the
#' final `n`-by-`k` index and distance matrices. Compatibility backends without
#' device-side selection transfer each distance block to the CPU for stable
#' ordering. On CPU, Euclidean blocks use the same guarded
#' translated-and-scaled implementation as [cuda_distance()].
#' @export
#' @examples
#' cuda_knn(
#'   matrix(rnorm(30), 10, 3),
#'   k = 3,
#'   batch_size = 4,
#'   device = "cpu"
#' )
cuda_knn <- function(x, k = 15L, metric = c("euclidean", "cosine"),
                     device = c("auto", "cuda", "cpu"),
                     batch_size = 256L) {
  sparse_input <- inherits(x, "cudasparse")
  source_device <- .learn_source_device(x)
  source_class <- class(x)[[1L]]
  input_stage <- .learn_input_stage(x)
  sparse_source <- NULL
  if (sparse_input) {
    sparse_source <- .learn_sparse_matrix(x)
    sparse_names <- .sparse_dimnames(sparse_source)
    observation_names <- if (is.null(sparse_names)) NULL else sparse_names[[1L]]
    n_observations <- sparse_source$shape[[1L]]
  } else {
    x <- .learn_matrix(x)
    observation_names <- rownames(x)
    n_observations <- nrow(x)
  }
  integer_k <- suppressWarnings(as.integer(k))
  if (!is.numeric(k) || length(k) != 1L || is.na(k) ||
      !is.finite(k) || is.na(integer_k) ||
      integer_k < 1L || integer_k >= n_observations || k != integer_k) {
    stop("`k` must be a whole number between 1 and nrow(x) - 1.",
         call. = FALSE)
  }
  metric <- match.arg(metric)
  selection <- .learn_device(device)
  device <- selection$device
  batch_size <- .row_batch_size(batch_size, n_observations)
  compute_backend <- .learn_selection_backend(selection)
  backend_sparse <- sparse_input &&
    .backend_has_operation(compute_backend, "algorithm_sparse_knn_prepare")
  sparse_transferred <- FALSE
  if (backend_sparse) {
    sparse_compute <- sparse_source
    source_backend <- if (is.null(sparse_source$backend_id)) {
      "base"
    } else {
      sparse_source$backend_id
    }
    if (!identical(sparse_source$device, "cuda") ||
        !identical(source_backend, compute_backend)) {
      sparse_compute <- cuda_sparse(
        .triplet_matrix(sparse_source),
        format = sparse_source$format,
        device = "cuda"
      )
      sparse_transferred <- TRUE
    }
    state <- list(
      values = NULL,
      storage = .backend_call(
        compute_backend,
        "algorithm_sparse_knn_prepare",
        sparse_compute$storage,
        sparse_compute$shape,
        metric
      ),
      metric = metric,
      device = selection$device,
      backend = compute_backend
    )
  } else {
    if (sparse_input) x <- .learn_sparse_dense(sparse_source)
    cosine_values <- if (metric == "cosine") {
      .cosine_unit_rows(x, "x")
    } else {
      NULL
    }
    state <- .knn_distance_state(x, metric, selection, cosine_values)
  }
  device_topk <- .backend_has_operation(
    state$backend, "algorithm_knn_select"
  )
  if (device_topk) {
    selected <- .backend_call(
      state$backend,
      "algorithm_knn_select",
      state$storage,
      state$values,
      integer_k,
      state$metric,
      batch_size
    )
    index <- selected$index
    neighbour_distance <- selected$distance
  } else {
    reference_index <- seq_len(n_observations)
    index <- matrix(NA_integer_, n_observations, integer_k)
    neighbour_distance <- matrix(NA_real_, n_observations, integer_k)

    starts <- seq.int(1L, n_observations, by = batch_size)
    for (start in starts) {
      rows <- seq.int(
        start,
        length.out = min(batch_size, n_observations - start + 1L)
      )
      distances <- .knn_distance_block(state, rows)
      selected <- vapply(
        seq_along(rows),
        function(i) {
          candidates <- reference_index[-rows[[i]]]
          ordering <- order(
            distances[i, candidates],
            candidates,
            method = "radix"
          )
          candidates[ordering[seq_len(integer_k)]]
        },
        integer(integer_k)
      )
      selected <- t(matrix(
        selected,
        nrow = integer_k,
        ncol = length(rows)
      ))
      selected_distance <- distances[cbind(
        rep(seq_along(rows), each = integer_k),
        as.vector(t(selected))
      )]

      index[rows, ] <- selected
      neighbour_distance[rows, ] <- matrix(
        selected_distance,
        nrow = length(rows),
        ncol = integer_k,
        byrow = TRUE
      )
    }
  }

  if (!is.null(observation_names)) {
    neighbor_names <- paste0("neighbor_", seq_len(integer_k))
    dimnames(index) <- list(observation_names, neighbor_names)
    dimnames(neighbour_distance) <- list(
      observation_names,
      neighbor_names
    )
  }

  output <- structure(
    list(
      index = index,
      distance = neighbour_distance,
      metric = metric,
      device = device
    ),
    class = "cuda_knn"
  )
  stages <- if (sparse_input && length(sparse_source$compute_stages)) {
    sparse_source$compute_stages
  } else {
    .learn_add_stage(list(), "input_materialization", input_stage)
  }
  if (backend_sparse) {
    if (sparse_transferred) {
      stages$sparse_transfer <- .learn_stage(
        selection,
        backend = state$backend,
        output_device = "cuda",
        reason = "sparse_input_transfer"
      )
    }
    stages$sparse_to_dense <- .learn_stage(
      selection,
      backend = state$backend,
      output_device = "cuda",
      reason = "device_resident_conversion"
    )
  } else if (sparse_input) {
    stages$sparse_to_dense <- .learn_cpu_stage(
      backend = "Matrix",
      reason = "algorithm_materialization"
    )
  }
  stages$distance <- .learn_stage(
    selection,
    backend = state$backend,
    output_device = if (device_topk) "cuda" else "cpu"
  )
  stages$neighbor_selection <- if (device_topk) {
    .learn_stage(selection, backend = state$backend, output_device = "cpu")
  } else {
    .learn_cpu_stage()
  }
  .with_learning_provenance(
    output,
    stages,
    requested_device = selection$requested_device,
    backend = if (device_topk) {
      state$backend
    } else if (identical(device, "cuda")) {
      paste0(state$backend, "+base")
    } else {
      "base"
    },
    parameters = list(
      k = integer_k,
      metric = metric,
      batch_size = batch_size
    ),
    source_device = source_device,
    source_class = source_class
  )
}

#' GPU-aware k-means clustering
#'
#' @param x Numeric matrix with observations in rows.
#' @param centers Number of clusters or a matrix of initial centres.
#' @param iter.max Maximum Lloyd iterations.
#' @param tolerance Convergence tolerance for centre movement.
#' @param seed Optional random seed used for initial centres.
#' @param device Device used for the numerical clustering stages.
#' @return A `cuda_kmeans` list containing integer `cluster` assignments,
#'   final `centers`, per-cluster `withinss`, `tot.withinss`, the number of
#'   iteration count in `iter`, a logical `converged` flag, and the actual
#'   distance `device`.
#'   Observation and feature names are retained when supplied.
#' @details The native CUDA backend uploads the observations and initial
#'   centres once, then keeps distance calculation, deterministic assignment,
#'   accumulation, and centre updates on the device. Only the small convergence
#'   movement summary is inspected between iterations; final assignments,
#'   centres, and within-cluster sums are transferred to R. Compatibility
#'   backends without a resident k-means operation retain the established
#'   distance-on-backend and update-on-CPU implementation.
#' @export
#' @examples
#' set.seed(1)
#' x <- rbind(matrix(rnorm(40), 20, 2), matrix(rnorm(40, 4), 20, 2))
#' cuda_kmeans(x, centers = 2, seed = 1, device = "cpu")
cuda_kmeans <- function(x, centers, iter.max = 100L, tolerance = 1e-6,
                        seed = NULL,
                        device = c("auto", "cuda", "cpu")) {
  source_device <- .learn_source_device(x)
  source_class <- class(x)[[1L]]
  input_stage <- .learn_input_stage(x)
  x <- .learn_matrix(x)
  observation_names <- rownames(x)
  feature_names <- colnames(x)
  selection <- .learn_device(device)
  device <- selection$device
  if (!is.numeric(iter.max) || length(iter.max) != 1L ||
      is.na(iter.max) || iter.max < 1 || iter.max != as.integer(iter.max)) {
    stop("`iter.max` must be a positive whole number.", call. = FALSE)
  }
  if (!is.numeric(tolerance) || length(tolerance) != 1L ||
      is.na(tolerance) || !is.finite(tolerance) || tolerance <= 0) {
    stop("`tolerance` must be a positive finite number.", call. = FALSE)
  }
  if (length(centers) == 1L && is.numeric(centers)) {
    k <- as.integer(centers)
    if (is.na(k) || k < 1L || k >= nrow(x) || centers != k) {
      stop("Numeric `centers` must be between 1 and nrow(x) - 1.",
           call. = FALSE)
    }
    centre_matrix <- .with_preserved_seed(
      seed,
      x[sample.int(nrow(x), k), , drop = FALSE]
    )
  } else {
    centre_matrix <- .learn_matrix(as.matrix(centers), "centers",
                                   min_rows = 1L)
    if (ncol(centre_matrix) != ncol(x)) {
      stop("Initial centres must have the same number of columns as `x`.",
           call. = FALSE)
    }
    k <- nrow(centre_matrix)
  }

  compute_backend <- .learn_selection_backend(selection)
  resident_kmeans <- .backend_has_operation(
    compute_backend, "algorithm_kmeans"
  )
  if (resident_kmeans) {
    result <- .backend_call(
      compute_backend,
      "algorithm_kmeans",
      x,
      centre_matrix,
      as.integer(iter.max),
      tolerance
    )
    cluster <- result$cluster
    centre_matrix <- result$centers
    withinss <- result$withinss
    iteration <- result$iter
    converged <- result$converged
  } else {
    converged <- FALSE
    final_distance <- cuda_distance(x, centre_matrix, device = device)
    cluster <- max.col(-final_distance, ties.method = "first")
    for (iteration in seq_len(as.integer(iter.max))) {
      new_centres <- centre_matrix
      for (group in seq_len(k)) {
        members <- x[cluster == group, , drop = FALSE]
        if (nrow(members) > 0L) {
          new_centres[group, ] <- colMeans(members)
        }
      }
      movement <- max(abs(new_centres - centre_matrix))
      centre_matrix <- new_centres
      final_distance <- cuda_distance(x, centre_matrix, device = device)
      cluster <- max.col(-final_distance, ties.method = "first")
      if (movement <= tolerance) {
        converged <- TRUE
        break
      }
    }
    withinss <- vapply(
      seq_len(k),
      function(group) {
        members <- which(cluster == group)
        indices <- cbind(members, rep.int(group, length(members)))
        sum(final_distance[indices]^2)
      },
      numeric(1)
    )
  }

  if (!is.null(observation_names) || !is.null(feature_names)) {
    cluster_names <- paste0("cluster_", seq_len(k))
    names(cluster) <- observation_names
    dimnames(centre_matrix) <- list(cluster_names, feature_names)
    names(withinss) <- cluster_names
  }

  output <- structure(
    list(
      cluster = cluster,
      centers = centre_matrix,
      withinss = withinss,
      tot.withinss = sum(withinss),
      iter = iteration,
      converged = converged,
      device = device
    ),
    class = "cuda_kmeans"
  )
  stages <- .learn_add_stage(list(), "input_materialization", input_stage)
  stages$initialization <- .learn_cpu_stage()
  stages$distance <- .learn_stage(
    selection,
    backend = compute_backend,
    output_device = if (resident_kmeans) "cuda" else "cpu"
  )
  stages$assignment <- if (resident_kmeans) {
    .learn_stage(selection, backend = compute_backend, output_device = "cuda")
  } else {
    .learn_cpu_stage()
  }
  stages$center_update <- if (resident_kmeans) {
    .learn_stage(selection, backend = compute_backend, output_device = "cuda")
  } else {
    .learn_cpu_stage()
  }
  if (resident_kmeans) {
    stages$finalization <- .learn_stage(
      selection,
      backend = compute_backend,
      output_device = "cpu"
    )
  }
  .with_learning_provenance(
    output,
    stages,
    requested_device = selection$requested_device,
    backend = if (resident_kmeans) {
      compute_backend
    } else if (identical(device, "cuda")) {
      paste0(compute_backend, "+base")
    } else {
      "base"
    },
    parameters = list(
      centers = if (length(centers) == 1L) {
        as.integer(centers)
      } else {
        "matrix"
      },
      iter.max = as.integer(iter.max),
      tolerance = tolerance,
      seed = seed
    ),
    source_device = source_device,
    source_class = source_class
  )
}

#' Assign observations with a fitted CUDA-aware k-means model
#'
#' `predict.cuda_kmeans()` computes Euclidean distances to the fitted centres
#' and returns either the closest-centre assignment or the complete distance
#' matrix. Named features may be supplied in any order and are aligned safely.
#'
#' @param object A fitted `cuda_kmeans` object.
#' @param newdata A finite numeric matrix or data frame with observations in
#'   rows and model features in columns. When omitted and `type = "cluster"`,
#'   the training assignments in `object$cluster` are returned.
#' @param type Return closest-centre `"cluster"` assignments or the
#'   observation-by-centre `"distance"` matrix.
#' @param device Device used for the distance calculation. `"model"` reuses
#'   the fitted model's actual distance device; `"auto"`, `"cuda"`, and
#'   `"cpu"` follow the usual cudaverse device-selection rules.
#' @param ... Must be empty.
#' @return For `type = "cluster"`, an integer vector with observation names
#'   and, for recomputed assignments, stage-level provenance. For
#'   `type = "distance"`, a numeric matrix whose columns identify the fitted
#'   centres. Omitting `newdata` returns validated stored training assignments
#'   unchanged and does not create a prediction stage.
#' @seealso [cuda_kmeans()]
#' @method predict cuda_kmeans
#' @export
#' @examples
#' train <- as.matrix(iris[1:100, 1:4])
#' fit <- cuda_kmeans(train, centers = 3, seed = 1, device = "cpu")
#' predict(fit, as.matrix(iris[101:105, 1:4]), device = "cpu")
predict.cuda_kmeans <- function(object, newdata,
                               type = c("cluster", "distance"),
                               device = c("model", "auto", "cuda", "cpu"),
                               ...) {
  .learn_check_prediction_dots(...)
  centers <- .learn_validate_kmeans(object)
  type <- match.arg(type)
  if (missing(newdata)) {
    if (identical(type, "distance")) {
      stop(
        "`newdata` is required when `type = \"distance\"`.",
        call. = FALSE
      )
    }
    .learn_validate_kmeans_clusters(object)
    return(object$cluster)
  }

  source_device <- .learn_source_device(newdata)
  source_class <- class(newdata)[[1L]]
  input_stage <- .learn_input_stage(newdata)
  values <- .learn_prediction_matrix(
    newdata,
    feature_names = colnames(centers),
    n_features = ncol(centers)
  )
  selection <- .learn_prediction_selection(device, object$device)
  distances <- cuda_distance(
    values,
    centers,
    metric = "euclidean",
    device = selection$device
  )
  center_names <- rownames(centers)
  if (is.null(center_names)) {
    center_names <- paste0("cluster_", seq_len(nrow(centers)))
  }
  dimnames(distances) <- list(rownames(values), center_names)
  stages <- .learn_add_stage(
    list(),
    "input_materialization",
    input_stage
  )
  distance_backend <- .learn_selection_backend(selection)
  stages$distance <- .learn_stage(
    selection,
    backend = distance_backend,
    output_device = "cpu"
  )
  distances <- .with_learning_provenance(
    distances,
    stages,
    requested_device = selection$requested_device,
    backend = distance_backend,
    parameters = list(type = "distance", metric = "euclidean"),
    source_device = source_device,
    source_class = source_class
  )
  if (identical(type, "distance")) {
    return(distances)
  }

  cluster <- max.col(-distances, ties.method = "first")
  names(cluster) <- rownames(values)
  attr(cluster, "device") <- attr(distances, "device", exact = TRUE)
  stages <- attr(distances, "compute_stages", exact = TRUE)
  stages$assignment <- .learn_cpu_stage()
  backend <- if (identical(selection$device, "cuda")) {
    paste0(distance_backend, "+base")
  } else {
    "base"
  }
  .with_learning_provenance(
    cluster,
    stages,
    requested_device = selection$requested_device,
    backend = backend,
    parameters = list(type = "cluster", metric = "euclidean"),
    source_device = attr(distances, "source_device", exact = TRUE),
    source_class = attr(distances, "source_class", exact = TRUE)
  )
}

#' @export
print.cuda_svd <- function(x, ...) {
  cat(sprintf(
    "<cuda_svd rank=%s device=%s compute=%s backend=%s>\n",
    length(x$d),
    x$device,
    x$compute_device,
    x$backend
  ))
  invisible(x)
}

#' @export
print.cuda_pca <- function(x, ...) {
  cat(sprintf(
    "<cuda_pca components=%s device=%s compute=%s backend=%s>\n",
    ncol(x$rotation), x$device, x$compute_device, x$backend
  ))
  print(x$rotation, ...)
  invisible(x)
}

#' @export
print.cuda_knn <- function(x, ...) {
  cat(sprintf(
    paste0(
      "<cuda_knn observations=%s k=%s metric=%s ",
      "distance_device=%s compute=%s backend=%s>\n"
    ),
    nrow(x$index), ncol(x$index), x$metric, x$device,
    x$compute_device, x$backend
  ))
  invisible(x)
}

#' @export
print.cuda_kmeans <- function(x, ...) {
  cat(sprintf(
    paste0(
      "<cuda_kmeans clusters=%s iterations=%s converged=%s ",
      "distance_device=%s compute=%s backend=%s>\n"
    ),
    nrow(x$centers), x$iter, x$converged, x$device,
    x$compute_device, x$backend
  ))
  print(x$centers, ...)
  invisible(x)
}
