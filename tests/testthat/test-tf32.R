test_that("native TF32 is per-call, labeled, and preserves exact control products", {
  skip_if_not(identical(Sys.getenv("CUDAVERSE_NATIVE_TESTS"), "true"))
  skip_if_not(isTRUE(cudaverse:::.native_diagnostics()$auto_eligible))
  skip_if(identical(Sys.getenv("NVIDIA_TF32_OVERRIDE"), "0"),
          "TF32 was explicitly disabled for this process")
  old <- options(cudaverse.cuda_backends = "native")
  on.exit(options(old), add = TRUE)
  left <- matrix(as.numeric((seq_len(15) %% 7) - 3), 3, 5,
                 dimnames = list(paste0("r", 1:3), paste0("k", 1:5)))
  right <- matrix(as.numeric((seq_len(35) %% 9) - 4), 5, 7,
                  dimnames = list(colnames(left), paste0("c", 1:7)))
  x <- cuda_tensor(left, "cuda", "float32")
  y <- cuda_tensor(right, "cuda", "float32")
  before <- to_cpu(tensor_matmul(x, y))
  result <- tryCatch(tensor_matmul(x, y, precision = c(mode = "tf32")),
                     error = identity)
  if (inherits(result, "error") && grepl(
      "TF32 matmul requires compute capability >= 8.0",
      conditionMessage(result), ignore.case = TRUE)) {
    skip(conditionMessage(result))
  }
  if (inherits(result, "error")) stop(result)
  expect_identical(to_cpu(result), left %*% right)
  expect_identical(result$dtype, "float32")
  expect_identical(tensor_shape(result), c(3L, 7L))
  expect_identical(dimnames(result), list(rownames(left), colnames(right)))
  expect_identical(tensor_device(result), c(device = "cuda", backend = "native"))
  provenance <- cuda_provenance(result)
  expect_identical(provenance$stage, "matrix_multiply")
  expect_identical(provenance$selection_reason, "matmul_tf32_permitted")
  expect_false(provenance$fallback)
  expect_identical(to_cpu(tensor_matmul(x, y)), before)

  identity <- cuda_tensor(diag(5), "cuda", "float32")
  permutation <- diag(5)[, c(5, 1, 4, 2, 3)]
  expect_identical(unname(to_cpu(tensor_matmul(
    cuda_tensor(unname(left), "cuda", "float32"), identity,
    precision = "tf32"
  ))), unname(left))
  expect_identical(unname(to_cpu(tensor_matmul(
    cuda_tensor(unname(left), "cuda", "float32"),
    cuda_tensor(permutation, "cuda", "float32"), precision = "tf32"
  ))), unname(left %*% permutation))

  expect_error(tensor_matmul(cuda_tensor(x, "cuda", "float64"), y,
                             precision = "tf32"), "dtype `float32`")
  expect_error(tensor_matmul(x, cuda_tensor(y, "cuda", "float64"),
                             precision = "tf32"), "dtype `float32`")
  expect_error(tensor_matmul(x, to_device(y, "cpu"), precision = "tf32"),
               "two native CUDA tensors")
  expect_error(tensor_matmul(to_device(x, "cpu"), y, precision = "tf32"),
               "two native CUDA tensors")
  expect_identical(to_cpu(tensor_matmul(x, y)), before)
})

test_that("disabling TF32 rejects the request without leaking or changing standard", {
  skip_if_not(identical(Sys.getenv("CUDAVERSE_NATIVE_TESTS"), "true"))
  skip_if_not(isTRUE(cudaverse:::.native_diagnostics()$auto_eligible))
  old_options <- options(cudaverse.cuda_backends = "native")
  old_override <- Sys.getenv("NVIDIA_TF32_OVERRIDE", unset = NA_character_)
  on.exit({
    options(old_options)
    if (is.na(old_override)) Sys.unsetenv("NVIDIA_TF32_OVERRIDE") else
      Sys.setenv(NVIDIA_TF32_OVERRIDE = old_override)
  }, add = TRUE)
  x <- cuda_tensor(matrix(c(1, -2, 3, 0.25), 2), "cuda", "float32")
  before <- to_cpu(tensor_matmul(x, x))
  invisible(gc())
  baseline <- cudaverse:::.native_memory_tracker()$current
  Sys.setenv(NVIDIA_TF32_OVERRIDE = "0")
  for (iteration in 1:3) {
    expect_error(tensor_matmul(x, x, precision = "tf32"), "NVIDIA_TF32_OVERRIDE")
  }
  invisible(gc())
  expect_identical(cudaverse:::.native_memory_tracker()$current, baseline)
  expect_identical(to_cpu(tensor_matmul(x, x)), before)
})
