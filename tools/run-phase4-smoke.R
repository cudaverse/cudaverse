required <- c("cudaverse", "Matrix")
missing <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing)) {
  stop("Install required package(s): ", paste(missing, collapse = ", "))
}

assert <- function(condition, message) {
  if (!isTRUE(condition)) stop(message, call. = FALSE)
}

old_options <- options(cudaverse.cuda_backends = NULL)
on.exit(options(old_options), add = TRUE)

diagnostics <- cudaverse::cuda_diagnostics()
native <- diagnostics$backend_diagnostics$native
assert(identical(diagnostics$selected_backend, "native"),
       "`device = auto` did not select the native backend.")
assert(isTRUE(native$runtime_complete),
       "The native CUDA runtime is incomplete.")
assert(isTRUE(native$capability_compatible),
       "The native backend contract or capability set is incompatible.")
assert(isTRUE(native$self_test$passed),
       "The native runtime self-test did not pass.")
assert(isTRUE(native$auto_eligible),
       "The native backend is not auto-eligible.")
assert(!length(native$missing_auto_capabilities),
       "The native backend is missing auto-selection capabilities.")

selection <- cudaverse::cuda_select_device("auto")
assert(identical(selection$backend, "native"),
       "Automatic device selection did not return native CUDA.")
assert(!selection$fallback, "Automatic native selection reported fallback.")
assert(identical(selection$selection_reason, "native_auto_eligible"),
       "Automatic native selection did not record its eligibility reason.")

left <- matrix(seq_len(15) / 7, 5, 3)
right <- matrix(seq_len(12) / 11, 3, 4)
for (dtype in c("float32", "float64")) {
  left_tensor <- cudaverse::cuda_tensor(left, dtype = dtype, device = "auto")
  right_tensor <- cudaverse::cuda_tensor(right, dtype = dtype, device = "auto")
  product <- cudaverse::tensor_matmul(
    left_tensor,
    right_tensor
  )
  tolerance <- if (identical(dtype, "float32")) 1e-5 else 1e-10
  assert(
    isTRUE(all.equal(
      cudaverse::to_cpu(product),
      left %*% right,
      tolerance = tolerance
    )),
    paste(dtype, "native matmul parity failed.")
  )
  assert(
    identical(
      cudaverse::tensor_device(product),
      c(device = "cuda", backend = "native")
    ),
    paste(dtype, "matmul did not remain on native CUDA.")
  )
  vector <- cudaverse::cuda_tensor(
    c(2, 3, 4), dtype = dtype, device = "auto"
  )
  broadcast <- left_tensor + vector
  assert(
    isTRUE(all.equal(
      cudaverse::to_cpu(broadcast),
      sweep(left, 2L, c(2, 3, 4), "+"),
      tolerance = tolerance
    )),
    paste(dtype, "broadcast arithmetic parity failed.")
  )
  reshaped <- cudaverse::tensor_reshape(left_tensor, c(3L, 5L))
  assert(
    isTRUE(all.equal(
      cudaverse::to_cpu(reshaped),
      array(left, dim = c(3L, 5L)),
      tolerance = tolerance
    )),
    paste(dtype, "reshape parity failed.")
  )
  assert(
    isTRUE(all.equal(
      cudaverse::to_cpu(t(left_tensor)),
      t(left),
      tolerance = tolerance
    )),
    paste(dtype, "transpose parity failed.")
  )
}
rm(product, left_tensor, right_tensor, vector, broadcast, reshaped)

set.seed(20260810)
values <- matrix(stats::rpois(64 * 16, lambda = 2), 64, 16)
values[, 1L] <- values[, 1L] + 1
source <- Matrix::Matrix(values, sparse = TRUE)
sparse <- cudaverse::cuda_sparse(source, device = "auto")
normalized <- cudaverse::sparse_normalize(
  sparse,
  margin = "rows",
  scale_factor = 1000,
  log1p = TRUE
)
fit <- cudaverse::cuda_pca(
  normalized,
  n_components = 6L,
  device = "auto"
)
neighbors <- cudaverse::cuda_knn(
  fit$x,
  k = 5L,
  device = "auto",
  batch_size = 17L
)
reference_values <- log1p(values * 1000 / rowSums(values))
reference_fit <- cudaverse::cuda_pca(
  reference_values,
  n_components = 6L,
  device = "cpu"
)
reference_neighbors <- cudaverse::cuda_knn(
  reference_fit$x,
  k = 5L,
  device = "cpu",
  batch_size = 17L
)
assert(
  isTRUE(all.equal(
    tcrossprod(fit$rotation),
    tcrossprod(reference_fit$rotation),
    tolerance = 1e-8
  )),
  "Native sparse PCA subspace parity failed."
)
assert(identical(neighbors$index, reference_neighbors$index),
       "Native stable top-k indices differ from CPU.")
assert(
  isTRUE(all.equal(
    neighbors$distance,
    reference_neighbors$distance,
    tolerance = 1e-8
  )),
  "Native kNN distance parity failed."
)
pipeline_provenance <- cudaverse::cuda_provenance(neighbors)
assert(identical(attr(pipeline_provenance, "schema"), "cudaverse-stage/1"),
       "The pipeline provenance schema changed.")
assert(all(pipeline_provenance$backend == "native"),
       "The automatic resident pipeline left the native backend.")

factory <- cudaverse:::.native_backend_factory()
factory$synchronize()
rm(sparse, normalized, fit, neighbors, reference_fit, reference_neighbors)
gc()
baseline <- cudaverse:::.native_memory_tracker(reset = TRUE)$current
self_test <- cudaverse:::.native_self_test(reset = TRUE)
factory$synchronize()
gc()
final <- cudaverse:::.native_memory_tracker()$current
assert(self_test$passed, "The reset runtime self-test failed.")
assert(identical(final, baseline),
       "The runtime self-test retained tracked device memory.")

cat("PHASE4_GPU_SMOKE=PASS\n")
cat("selected_backend=native\n")
cat("self_test_ms=", native$self_test$duration_ms, "\n", sep = "")
cat("pipeline=native sparse normalization -> PCA -> distance -> stable top-k\n")
