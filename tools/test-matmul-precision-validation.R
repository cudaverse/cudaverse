sys.source(file.path("tools", "benchmark-validation.R"), envir = environment())
sys.source(file.path("tools", "matmul-precision-validation.R"), envir = environment())

check_tf32 <- matmul_precision_tf32_validation
f32 <- benchmark_float32_input
nearest_tf32_model <- function(x) {
  step <- 2^(floor(log2(abs(x))) - 10)
  value <- round(x / step) * step
  value[x == 0] <- 0
  value
}
set.seed(20260926)
left <- f32(matrix(rnorm(7L * 17L), 7L, 17L))
right <- f32(matrix(rnorm(17L * 9L), 17L, 9L))
reference <- left %*% right
modeled <- f32(nearest_tf32_model(left) %*% nearest_tf32_model(right))
accepted <- check_tf32(modeled, reference, left, right)
stopifnot(accepted$passed, accepted$failed_elements == 0L,
          accepted$max_scaled_error <= 1,
          !accepted$strict_diagnostic$passed)
corrupted <- modeled
corrupted[2L, 3L] <- reference[2L, 3L] + 2 * accepted$max_allowance
stopifnot(!check_tf32(corrupted, reference, left, right)$passed,
          !check_tf32(t(modeled), reference, left, right)$passed)

# A very large scale in another cell cannot excuse local indexing corruption.
left <- f32(diag(c(2^40, 1)))
right <- f32(diag(c(2^40, 1)))
reference <- left %*% right
corrupted <- reference
corrupted[2L, 2L] <- 1.1
stopifnot(check_tf32(reference, reference, left, right)$passed,
          !check_tf32(corrupted, reference, left, right)$passed)
left <- f32(matrix(c(1, 2, 3, 4), 2L))
right <- f32(matrix(c(2, 1, 4, 3), 2L))
reference <- left %*% right
stopifnot(!check_tf32(t(reference), reference, left, right)$passed,
          !check_tf32(reference[2:1, ], reference, left, right)$passed)

# Zero support stays exact even inside the absolute underflow allowance.
zero <- matrix(0, 2L, 2L)
corrupted <- zero
corrupted[1L] <- 2^-149
stopifnot(check_tf32(zero, zero, zero, zero)$passed,
          !check_tf32(corrupted, zero, zero, zero)$passed)

# Normal inputs can produce a subnormal product, covered by the stated model.
tiny <- matrix(2^-80, 1L)
tiny_reference <- tiny %*% tiny
stopifnot(check_tf32(matrix(0, 1L), tiny_reference, tiny, tiny)$passed)
subnormal <- matrix(2^-140, 1L)
one <- matrix(1, 1L)
stopifnot(!check_tf32(subnormal, subnormal, subnormal, one)$passed)
largest <- matrix((2 - 2^-23) * 2^127, 1L)
small <- matrix(2^-100, 1L)
conversion <- check_tf32(largest %*% small, largest %*% small, largest, small)
stopifnot(!conversion$passed,
          identical(conversion$unsupported_reason, "TF32 input conversion may overflow float32"))
large <- matrix(2^100, 1L)
stopifnot(!check_tf32(large %*% large, large %*% large, large, large)$passed,
          !check_tf32(matrix(Inf, 1L), one, one, one)$passed,
          !check_tf32(matrix(NA_real_, 1L), one, one, one)$passed,
          !check_tf32(matrix(pi, 1L), matrix(pi, 1L), matrix(pi, 1L), one)$passed,
          !check_tf32(matrix(numeric(), 0L, 0L), matrix(numeric(), 0L, 0L),
                      matrix(numeric(), 0L, 0L), matrix(numeric(), 0L, 0L))$passed)

# Tiny integer controls need exact agreement in the driver in addition to the
# broad arithmetic bound. TF32 represents these inputs and outputs exactly.
left <- f32(matrix(as.double(seq_len(15L) %% 7L - 3L), 3L, 5L))
right <- f32(matrix(as.double(seq_len(35L) %% 9L - 4L), 5L, 7L))
reference <- left %*% right
stopifnot(check_tf32(reference, reference, left, right)$passed,
          identical(nearest_tf32_model(left), left),
          identical(nearest_tf32_model(right), right))
message("TF32 componentwise model, range guards, and corruption checks passed.")
