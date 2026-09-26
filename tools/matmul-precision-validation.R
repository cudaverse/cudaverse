# Source tools/benchmark-validation.R before this file. These helpers validate
# the explicitly permitted TF32 experiment; standard FP32 uses the package's
# separate benchmark_float32_matmul_validation() policy.
matmul_precision_tf32_policy <- function() {
  list(
    version = "tf32-permitted-dot-product-model/1",
    comparison = paste(
      "elementwise |actual-reference| <=",
      "(2q+q^2+(1+q)^2*gamma(2k,u32)+gamma(2k,u64))*S_upper + underflow_bound"
    ),
    input_unit_roundoff_tf32 = 2^-11,
    accumulation_unit_roundoff_float32 = 2^-24,
    reference_unit_roundoff_float64 = 2^-53,
    operation_count_per_term = 2L,
    min_normal_float32 = 2^-126,
    bound_arithmetic_inflation = 1 + 32 * 2^-53,
    gamma = "gamma(n,u) = n*u/(1-n*u)",
    s_upper = "fl64(|left| %*% |right|)/(1-gamma(2k,u64)), with binary64 arithmetic inflation",
    input_conversion_error = "Nearest-rounded TF32 inputs: |delta x| <= q*|x|; ties may use either nearest rule",
    accumulation_model = paste(
      "At most 2k elementary operations with relative error <= u32 and",
      "absolute underflow error <= 2^-126 per operation; excludes input",
      "subnormals and overflow. Additive underflow allowance covers result",
      "flush-to-zero within this model."
    ),
    limits = paste(
      "This is an explicit arithmetic-model acceptance contract for bounded",
      "fixtures, not a universal NVIDIA PTX hardware guarantee. PTX leaves",
      "TF32 MMA rounding, order, and subnormal behavior unspecified.",
      "A successful precision='tf32' request permits Tensor Cores; it does",
      "not establish which hardware kernel cuBLAS selected."
    ),
    strict_diagnostic = list(rtol = 1e-5, atol = 1e-6),
    sources = c(
      "https://docs.nvidia.com/cuda/archive/12.8.0/cublas/index.html",
      "https://docs.nvidia.com/cuda/archive/11.1.1/parallel-thread-execution/index.html",
      "https://nhigham.com/wp-content/uploads/2023/10/high93s.pdf"
    )
  )
}

matmul_precision_tf32_validation <- function(actual, reference, left, right) {
  policy <- matmul_precision_tf32_policy()
  strict <- benchmark_numeric_validation(actual, reference, 1e-5, 1e-6)
  result <- list(policy_version = policy$version,
                 comparison = policy$comparison, passed = FALSE,
                 strict_diagnostic = strict)
  unsupported <- function(reason) {
    result$unsupported_reason <- reason
    result
  }
  matrices <- list(actual, reference, left, right)
  if (!all(vapply(matrices, function(x) {
    is.matrix(x) && is.numeric(x) && length(x) > 0L && all(is.finite(x))
  }, logical(1)))) {
    return(unsupported("inputs, reference and result must be nonempty finite numeric matrices"))
  }
  k <- ncol(left)
  if (nrow(right) != k ||
      !identical(dim(actual), c(nrow(left), ncol(right))) ||
      !identical(dim(reference), dim(actual))) {
    return(unsupported("matrix dimensions are not conformable"))
  }
  if (!identical(left, benchmark_float32_input(left)) ||
      !identical(right, benchmark_float32_input(right))) {
    return(unsupported("inputs are not exactly float32-rounded"))
  }
  tau <- policy$min_normal_float32
  if (any(abs(left) > 0 & abs(left) < tau) ||
      any(abs(right) > 0 & abs(right) < tau)) {
    return(unsupported("subnormal float32 inputs are outside this model"))
  }
  q <- policy$input_unit_roundoff_tf32
  u32 <- policy$accumulation_unit_roundoff_float32
  u64 <- policy$reference_unit_roundoff_float64
  operations <- as.double(policy$operation_count_per_term) * k
  if (operations * u32 >= 1 || operations * u64 >= 0.5) {
    return(unsupported("inner dimension exceeds the rounding-bound domain"))
  }
  gamma32 <- operations * u32 / (1 - operations * u32)
  gamma64 <- operations * u64 / (1 - operations * u64)
  inflation <- policy$bound_arithmetic_inflation
  max_float32 <- (2 - 2^-23) * 2^127
  if (max(abs(left), abs(right)) * (1 + q) * inflation > max_float32) {
    return(unsupported("TF32 input conversion may overflow float32"))
  }
  sum_products <- abs(left) %*% abs(right)
  sum_upper <- sum_products / (1 - gamma64) * inflation
  quantization_coefficient <- 2 * q + q^2
  accumulation_coefficient <- (1 + q)^2 * gamma32
  underflow_bound <- operations * tau / (1 - operations * u32)
  allowance <- ((quantization_coefficient + accumulation_coefficient + gamma64) *
                  sum_upper + underflow_bound) * inflation
  partial_sum_upper <- ((1 + q)^2 * (1 + gamma32) * sum_upper +
                         underflow_bound) * inflation
  if (any(!is.finite(sum_upper)) || any(!is.finite(allowance)) ||
      any(sum_upper < 0) || any(allowance < 0) ||
      any(!is.finite(partial_sum_upper)) || any(partial_sum_upper > max_float32)) {
    return(unsupported("the dot product may overflow float32 or its bound is not finite"))
  }
  difference <- abs(actual - reference)
  # With zero absolute-product support, every contribution is exactly zero.
  # Do not let the additive underflow allowance excuse an invented result.
  zero_support <- sum_products == 0
  failed <- difference > allowance | (zero_support & (actual != 0 | reference != 0))
  scaled <- ifelse(allowance > 0, difference / allowance,
                   ifelse(difference == 0, 0, Inf))
  result$inner_dimension <- k
  result$input_unit_roundoff_tf32 <- q
  result$gamma_accumulation_float32 <- gamma32
  result$gamma_reference_float64 <- gamma64
  result$quantization_coefficient <- quantization_coefficient
  result$accumulation_coefficient <- accumulation_coefficient
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
