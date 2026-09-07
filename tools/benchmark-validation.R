benchmark_numeric_validation <- function(actual, reference, rtol, atol) {
  shape_matches <- identical(dim(actual), dim(reference)) &&
    identical(length(actual), length(reference))
  finite <- is.numeric(actual) && is.numeric(reference) &&
    all(is.finite(actual)) && all(is.finite(reference))
  if (!shape_matches || !finite) {
    return(list(
      comparison = "elementwise abs(actual-reference) <= atol+rtol*abs(reference)",
      shape_matches = shape_matches, finite = finite,
      rtol = rtol, atol = atol, passed = FALSE
    ))
  }
  difference <- abs(actual - reference)
  allowance <- atol + rtol * abs(reference)
  ratio <- ifelse(allowance > 0, difference / allowance,
                  ifelse(difference == 0, 0, Inf))
  absolute <- max(c(0, difference))
  list(
    comparison = "elementwise abs(actual-reference) <= atol+rtol*abs(reference)",
    shape_matches = TRUE, finite = TRUE,
    max_absolute_error = absolute,
    # Retained for existing summary readers; this descriptive global statistic
    # is never used as the numerical acceptance gate.
    max_relative_error = absolute / max(c(1, abs(reference))),
    max_tolerance_ratio = max(c(0, ratio)),
    failed_elements = sum(difference > allowance),
    rtol = rtol, atol = atol,
    passed = all(difference <= allowance)
  )
}

benchmark_float32_input <- function(x) {
  # Use exactly representable input values for the double-precision CPU
  # reference, so input quantization is not confused with compute error.
  values <- readBin(writeBin(as.numeric(x), raw(), size = 4L),
                    double(), n = length(x), size = 4L)
  array(values, dim = dim(x), dimnames = dimnames(x))
}

benchmark_pipeline_validation <- function(value, reference) {
  rotation <- value$pca$rotation
  scores <- value$pca$x
  rank_threshold <- max(reference$pca$sdev) *
    max(nrow(reference$pca$x), nrow(reference$pca$rotation)) *
    .Machine$double.eps
  effective_rank <- max(1L, sum(reference$pca$sdev > rank_threshold))
  components <- seq_len(effective_rank)
  # Projectors and reconstructions are invariant to PCA sign changes and
  # orthogonal rotations within the selected subspace.
  projector <- benchmark_numeric_validation(
    tcrossprod(rotation[, components, drop = FALSE]),
    tcrossprod(reference$pca$rotation[, components, drop = FALSE]),
    rtol = 0, atol = 1e-8
  )
  reconstruction <- benchmark_numeric_validation(
    scores[, components, drop = FALSE] %*%
      t(rotation[, components, drop = FALSE]),
    reference$pca$x[, components, drop = FALSE] %*%
      t(reference$pca$rotation[, components, drop = FALSE]),
    rtol = 1e-8, atol = 1e-10
  )
  distance <- benchmark_numeric_validation(
    value$knn$distance, reference$knn$distance,
    rtol = 1e-8, atol = 1e-10
  )
  normalized <- if (is.null(reference$normalized)) {
    list(passed = is.null(value$normalized), max_relative_error = 0)
  } else {
    benchmark_numeric_validation(value$normalized, reference$normalized,
                                 rtol = 1e-10, atol = 1e-10)
  }
  indices_identical <- identical(value$knn$index, reference$knn$index)
  list(
    normalized_max_relative_error = normalized$max_relative_error,
    pca_effective_rank = effective_rank,
    pca_projector_max_absolute_error = projector$max_absolute_error,
    pca_reconstruction_max_relative_error = reconstruction$max_relative_error,
    knn_indices_identical = indices_identical,
    knn_distance_max_relative_error = distance$max_relative_error,
    elementwise_checks = list(normalized = normalized, projector = projector,
                              reconstruction = reconstruction,
                              knn_distance = distance),
    passed = isTRUE(normalized$passed) && isTRUE(projector$passed) &&
      isTRUE(reconstruction$passed) && indices_identical &&
      isTRUE(distance$passed)
  )
}
