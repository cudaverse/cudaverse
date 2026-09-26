sys.source(file.path("tools", "benchmark-validation.R"), envir = environment())

for (tolerance in list(c(1e-5, 1e-6), c(1e-10, 1e-10), c(1e-8, 1e-10))) {
  reference <- matrix(c(1e8, 0, 1, -1), 2L)
  actual <- reference
  actual[2L] <- 10 * tolerance[[2L]]
  # The previous global scale passes this deliberately incorrect near-zero
  # element because a different element is large.
  stopifnot(max(abs(actual - reference)) <=
              tolerance[[2L]] + tolerance[[1L]] * max(abs(reference)))
  rejected <- benchmark_numeric_validation(
    actual, reference, rtol = tolerance[[1L]], atol = tolerance[[2L]]
  )
  stopifnot(!rejected$passed, rejected$failed_elements == 1L,
            rejected$max_tolerance_ratio > 1)
  actual[2L] <- tolerance[[2L]] / 2
  stopifnot(benchmark_numeric_validation(
    actual, reference, tolerance[[1L]], tolerance[[2L]]
  )$passed)
}
stopifnot(
  benchmark_numeric_validation(1:4, 1:4, 0, 0)$passed,
  !benchmark_numeric_validation(c(1L, 3L), c(1L, 2L), 0, 0)$passed,
  !benchmark_numeric_validation(matrix(1:4, 2), 1:4, 0, 0)$passed,
  !benchmark_numeric_validation(c(1, Inf), c(1, Inf), 1e-8, 1e-10)$passed,
  !benchmark_numeric_validation(c(1, NA_real_), c(1, 2), 1e-8, 1e-10)$passed
)
rounded <- benchmark_float32_input(matrix(c(pi, 1 / 3, -pi, 0), 2L))
stopifnot(identical(rounded, benchmark_float32_input(rounded)),
          identical(dim(rounded), c(2L, 2L)))
lf_source <- tempfile("benchmark-validator-lf-")
crlf_source <- tempfile("benchmark-validator-crlf-")
changed_source <- tempfile("benchmark-validator-changed-")
writeBin(charToRaw("x <- 1\ny <- 2\n"), lf_source)
writeBin(charToRaw("x <- 1\r\ny <- 2\r\n"), crlf_source)
writeBin(charToRaw("x <- 1\ny <- 3\n"), changed_source)
stopifnot(
  identical(benchmark_validation_source_sha256(lf_source),
            benchmark_validation_source_sha256(crlf_source)),
  !identical(benchmark_validation_source_sha256(lf_source),
             benchmark_validation_source_sha256(changed_source))
)

policy <- benchmark_float32_matmul_policy()
stopifnot(
  identical(policy$version, "standard-fp32-dot-product/1"),
  identical(policy$unit_roundoff_float32, 2^-24),
  identical(benchmark_float32_dot_operation_count(64L, policy), 128),
  identical(
    benchmark_float32_dot_operation_count(.Machine$integer.max, policy),
    2 * as.double(.Machine$integer.max)
  )
)
# A severe cancellation can exceed the original per-entry absolute tolerance
# while remaining well inside the standard FP32 dot-product error envelope.
left <- matrix(rep(c(1, -1), 32L), 1L, 64L)
right <- matrix(rep(1, 64L), 64L, 1L)
reference_dot <- left %*% right
cancelled <- benchmark_float32_matmul_validation(
  matrix(2e-6, 1L, 1L), reference_dot, left, right
)
stopifnot(
  cancelled$passed, cancelled$failed_elements == 0L,
  !cancelled$strict_diagnostic$passed,
  cancelled$strict_diagnostic$failed_elements == 1L,
  cancelled$strict_diagnostic$max_tolerance_ratio > 1,
  cancelled$max_scaled_error < 1
)
stopifnot(!benchmark_float32_matmul_validation(
  matrix(1, 1L, 1L), reference_dot, left, right
)$passed)

# The allowance is local to each output, never scaled by an unrelated large
# dot product elsewhere in the matrix.
left <- diag(c(1, 1000))
right <- diag(c(1, 1000))
reference_dot <- left %*% right
actual <- reference_dot
actual[1L, 1L] <- actual[1L, 1L] + 1e-3
local_failure <- benchmark_float32_matmul_validation(
  actual, reference_dot, left, right
)
stopifnot(!local_failure$passed, local_failure$failed_elements == 1L,
          local_failure$max_scaled_error > 1)
actual <- reference_dot
actual[1L, 1L] <- actual[1L, 1L] + 1e-7
local_pass <- benchmark_float32_matmul_validation(
  actual, reference_dot, left, right
)
stopifnot(local_pass$passed)
local_allowance <- (
  (local_pass$gamma_float32 + local_pass$gamma_reference_float64) /
    (1 - local_pass$gamma_reference_float64) *
    local_pass$bound_arithmetic_inflation +
    local_pass$underflow_absolute_bound
) * local_pass$bound_arithmetic_inflation
actual[1L, 1L] <- reference_dot[1L, 1L] + 2 * local_allowance
stopifnot(!benchmark_float32_matmul_validation(
  actual, reference_dot, left, right
)$passed)

# Exact-zero support, shape, order, and unsupported arithmetic regimes fail
# closed even when another entry has a generous condition-aware allowance.
left <- diag(2L)
right <- matrix(c(0, 1, 2, 0), 2L)
reference_dot <- left %*% right
actual <- reference_dot
actual[1L, 1L] <- 1e-40
stopifnot(!benchmark_float32_matmul_validation(
  actual, reference_dot, left, right
)$passed)
stopifnot(!benchmark_float32_matmul_validation(
  t(reference_dot), reference_dot, left, right
)$passed)
stopifnot(!benchmark_float32_matmul_validation(
  reference_dot[1L, , drop = FALSE], reference_dot, left, right
)$passed)
empty_rows <- benchmark_float32_matmul_validation(
  matrix(numeric(), 0L, 1L), matrix(numeric(), 0L, 1L),
  matrix(numeric(), 0L, 1L), matrix(1, 1L, 1L)
)
empty_columns <- benchmark_float32_matmul_validation(
  matrix(numeric(), 1L, 0L), matrix(numeric(), 1L, 0L),
  matrix(1, 1L, 1L), matrix(numeric(), 1L, 0L)
)
stopifnot(
  !empty_rows$passed, !empty_columns$passed,
  grepl("empty", empty_rows$unsupported_reason, fixed = TRUE),
  grepl("empty", empty_columns$unsupported_reason, fixed = TRUE)
)
stopifnot(!benchmark_float32_matmul_validation(
  matrix(NaN, 2L, 2L), reference_dot, left, right
)$passed)
stopifnot(!benchmark_float32_matmul_validation(
  matrix(pi, 1L, 1L), matrix(pi, 1L, 1L),
  matrix(pi, 1L, 1L), matrix(1, 1L, 1L)
)$passed)
stopifnot(!benchmark_float32_matmul_validation(
  matrix(2^-127, 1L, 1L), matrix(2^-127, 1L, 1L),
  matrix(2^-127, 1L, 1L), matrix(1, 1L, 1L)
)$passed)
stopifnot(!benchmark_float32_matmul_validation(
  matrix(2^200, 1L, 1L), matrix(2^200, 1L, 1L),
  matrix(2^100, 1L, 1L), matrix(2^100, 1L, 1L)
)$passed)

# Recreate the retained 64-by-64 fixture with the sequential FP32 FMA model.
# Its two historical strict failures remain visible but satisfy this policy.
set.seed(20262297L)
left <- benchmark_float32_input(matrix(stats::rnorm(64L * 64L), 64L, 64L))
right <- benchmark_float32_input(matrix(stats::rnorm(64L * 64L), 64L, 64L))
reference_dot <- left %*% right
fma_model <- matrix(0, 64L, 64L)
for (column in seq_len(64L)) {
  fma_model <- benchmark_float32_input(
    fma_model + outer(left[, column], right[column, ])
  )
}
retained_fixture <- benchmark_float32_matmul_validation(
  fma_model, reference_dot, left, right
)
stopifnot(
  retained_fixture$passed,
  retained_fixture$failed_elements == 0L,
  retained_fixture$strict_diagnostic$failed_elements == 2L,
  retained_fixture$strict_diagnostic$max_tolerance_ratio > 1.46
)

reference <- list(
  normalized = matrix(c(1e8, 0, 1, 2), 2L),
  pca = list(sdev = c(2, 1), rotation = diag(2),
             x = matrix(c(1e8, 0, 1, 2), 2L)),
  knn = list(index = matrix(c(2L, 1L, 3L, 3L), 2L),
             distance = matrix(c(1e8, 0, 1, 2), 2L))
)
stopifnot(benchmark_pipeline_validation(reference, reference)$passed)
flipped <- reference
flipped$pca$rotation[, 1L] <- -flipped$pca$rotation[, 1L]
flipped$pca$x[, 1L] <- -flipped$pca$x[, 1L]
stopifnot(benchmark_pipeline_validation(flipped, reference)$passed)
rotated <- reference
basis <- matrix(c(0, 1, -1, 0), 2L)
rotated$pca$rotation <- rotated$pca$rotation %*% basis
rotated$pca$x <- rotated$pca$x %*% basis
stopifnot(benchmark_pipeline_validation(rotated, reference)$passed)
for (field in c("normalized", "reconstruction", "distance", "ties")) {
  actual <- reference
  if (field == "normalized") actual$normalized[2L] <- 1e-6
  if (field == "reconstruction") actual$pca$x[2L] <- 1e-6
  if (field == "distance") actual$knn$distance[2L] <- 1e-6
  if (field == "ties") actual$knn$index[1L] <- 3L
  stopifnot(!benchmark_pipeline_validation(actual, reference)$passed)
}
message("Benchmark elementwise, PCA-invariance, and exact-index tests passed.")
