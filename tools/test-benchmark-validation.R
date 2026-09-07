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
