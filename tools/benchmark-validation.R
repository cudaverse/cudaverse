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

benchmark_validation_source_sha256 <- function(
    path = file.path("tools", "benchmark-validation.R")) {
  lines <- readLines(path, warn = FALSE, encoding = "UTF-8")
  normalized <- paste(sub("\r$", "", enc2utf8(lines)), collapse = "\n")
  digest::digest(normalized, algo = "sha256", serialize = FALSE)
}

benchmark_float32_matmul_policy <- function() {
  list(
    version = "standard-fp32-dot-product/1",
    comparison = paste(
      "elementwise |actual-reference| <=",
      "inflation*((gamma(2k,u32)+gamma(2k,u64))*S_upper +",
      "underflow_bound), with exact-zero support"
    ),
    sum_absolute_products = "S = |left| %*% |right| for float32-rounded inputs",
    s_upper = "fl64(S)/(1-gamma(2k,u64)), inflated for bound arithmetic",
    gamma = "gamma(n,u) = n*u/(1-n*u)",
    unit_roundoff_float32 = 2^-24,
    unit_roundoff_float64 = 2^-53,
    min_normal_float32 = 2^-126,
    operation_count_per_term = 2L,
    bound_arithmetic_inflation = 1 + 32 * 2^-53,
    scope = paste(
      "Benchmark acceptance under a standard full-mantissa float32",
      "elementary-operation model; finite binary32 inputs, no overflow.",
      "Subnormal products and intermediate sums allow one float32",
      "min-normal absolute error per elementary operation, covering",
      "flush-to-zero. TF32 is excluded. No cuBLAS internal reduction",
      "topology or universal cuBLAS error guarantee is asserted."
    ),
    strict_diagnostic = list(
      comparison = "elementwise |actual-reference| <= atol+rtol*|reference|",
      rtol = 1e-5, atol = 1e-6
    )
  )
}

benchmark_float32_dot_operation_count <- function(inner_dimension, policy) {
  as.double(policy$operation_count_per_term) * as.double(inner_dimension)
}

benchmark_float32_matmul_validation <- function(actual, reference, left,
                                                right) {
  policy <- benchmark_float32_matmul_policy()
  strict <- benchmark_numeric_validation(
    actual, reference,
    rtol = policy$strict_diagnostic$rtol,
    atol = policy$strict_diagnostic$atol
  )
  result <- list(
    policy_version = policy$version,
    comparison = policy$comparison,
    passed = FALSE,
    strict_diagnostic = strict
  )
  unsupported <- function(reason) {
    result$unsupported_reason <- reason
    result
  }
  if (!all(vapply(list(actual, reference, left, right), function(x) {
    is.matrix(x) && is.numeric(x) && all(is.finite(x))
  }, logical(1L)))) {
    return(unsupported(paste(
      "inputs, reference and result must be finite numeric matrices"
    )))
  }
  k <- ncol(left)
  if (nrow(left) < 1L || k < 1L || nrow(right) != k ||
      ncol(right) < 1L ||
      !identical(dim(actual), c(nrow(left), ncol(right))) ||
      !identical(dim(reference), dim(actual))) {
    return(unsupported("matrix dimensions are empty or not conformable"))
  }
  u32 <- policy$unit_roundoff_float32
  u64 <- policy$unit_roundoff_float64
  operations <- benchmark_float32_dot_operation_count(k, policy)
  if (operations * u32 >= 1 || operations * u64 >= 1) {
    return(unsupported("inner dimension exceeds the rounding-bound domain"))
  }
  if (!identical(left, benchmark_float32_input(left)) ||
      !identical(right, benchmark_float32_input(right))) {
    return(unsupported("inputs are not exactly float32-rounded"))
  }
  if (any(abs(left) > 0 & abs(left) < policy$min_normal_float32) ||
      any(abs(right) > 0 & abs(right) < policy$min_normal_float32)) {
    return(unsupported("subnormal float32 inputs are outside this policy"))
  }
  gamma <- function(count, unit_roundoff) {
    count * unit_roundoff / (1 - count * unit_roundoff)
  }
  gamma32 <- gamma(operations, u32)
  gamma64 <- gamma(operations, u64)
  # A binary32 product is exactly representable in binary64. The positive
  # binary64 matrix product T can only underestimate S by its own dot-product
  # rounding; compensate for that before using S in the binary32 error bound.
  sum_products <- abs(left) %*% abs(right)
  inflation <- policy$bound_arithmetic_inflation
  sum_upper <- (sum_products / (1 - gamma64)) * inflation
  # The additive term also covers float32 flush-to-zero for subnormal products
  # or partial sums. The input-subnormal case is rejected above.
  underflow_bound <- operations * policy$min_normal_float32 /
    (1 - operations * u32)
  allowance <- ((gamma32 + gamma64) * sum_upper + underflow_bound) *
    inflation
  max_float32 <- (2 - 2^-23) * 2^127
  if (any(!is.finite(sum_upper)) || any(!is.finite(allowance)) ||
      any(sum_upper < 0) || any(allowance < 0) ||
      any((1 + gamma32) * sum_upper + underflow_bound > max_float32)) {
    return(unsupported(paste(
      "the dot product may overflow float32 or its bound is not finite"
    )))
  }
  difference <- abs(actual - reference)
  if (any(!is.finite(difference))) {
    return(unsupported("result-reference differences are not finite"))
  }
  zero_support <- sum_products == 0
  failed <- difference > allowance |
    (zero_support & (actual != 0 | reference != 0))
  scaled <- ifelse(allowance > 0, difference / allowance,
                   ifelse(difference == 0, 0, Inf))
  result$inner_dimension <- k
  result$unit_roundoff_float32 <- u32
  result$unit_roundoff_float64 <- u64
  result$gamma_float32 <- gamma32
  result$gamma_reference_float64 <- gamma64
  result$underflow_absolute_bound <- underflow_bound
  result$bound_arithmetic_inflation <- inflation
  result$max_sum_absolute_products <- max(sum_products)
  result$max_allowance <- max(allowance)
  result$max_absolute_error <- max(difference)
  result$max_relative_error <- strict$max_relative_error
  result$max_scaled_error <- max(scaled)
  result$failed_elements <- sum(failed)
  result$passed <- !any(failed)
  result
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
