test_that("backend registry exposes the stable operation contract", {
  backend <- cudaverse:::.backend_get("base")

  expect_identical(backend$name, "base")
  expect_identical(backend$device, "cpu")
  expect_true(all(c(
    "diagnostics", "capabilities", "from_host", "to_host", "cast",
    "matmul", "reduce", "synchronize", "release", "error_translate"
  ) %in% names(backend)))
  expect_true(all(vapply(
    backend[setdiff(names(backend), c("name", "device"))],
    is.function,
    logical(1)
  )))
})

test_that("diagnostics extend rather than replace legacy fields", {
  diagnostics <- cuda_diagnostics()

  expect_true(all(c(
    "torch_installed", "torch_version", "cuda_available",
    "cuda_device_count", "reason", "detection_error"
  ) %in% names(diagnostics)))
  expect_type(diagnostics$available_backends, "character")
  expect_true("base" %in% diagnostics$available_backends)
  expect_type(diagnostics$selected_backend, "character")
  expect_length(diagnostics$selected_backend, 1L)
  expect_named(diagnostics$backend_diagnostics, c("torch", "native"))
})

test_that("native probe failures do not change the legacy detection field", {
  testthat::local_mocked_bindings(
    .backend_diagnostics = function(name) {
      if (identical(name, "torch")) {
        return(list(
          installed = FALSE, available = FALSE, device_count = 0L,
          version = NA_character_, reason = "torch_not_installed",
          detection_error = NULL
        ))
      }
      list(
        installed = TRUE, available = FALSE, device_count = 0L,
        version = NA_character_, reason = "backend_error",
        detection_error = "injected native probe failure"
      )
    }
  )

  diagnostics <- cuda_diagnostics()
  expect_null(diagnostics$detection_error)
  expect_identical(
    diagnostics$backend_diagnostics$native$detection_error,
    "injected native probe failure"
  )
})

test_that("CPU tensor and algorithm adapters match their R references", {
  x <- matrix(seq_len(6), 2, 3)
  y <- matrix(seq_len(6), 3, 2)
  product <- tensor_matmul(
    cuda_tensor(x, device = "cpu"),
    cuda_tensor(y, device = "cpu")
  )

  expect_equal(to_cpu(product), x %*% y)
  expect_identical(tensor_device(product), c(device = "cpu", backend = "base"))
  expect_equal(
    as.vector(cuda_distance(x, device = "cpu")),
    as.vector(as.matrix(stats::dist(x)))
  )
})

test_that("missing backend capabilities return structured conditions", {
  factory <- list(
    name = "contract-test",
    device = "cuda",
    diagnostics = function() list(available = FALSE),
    capabilities = function() "matmul",
    from_host = identity,
    to_host = identity,
    matmul = function(x, y) x,
    synchronize = function() invisible(TRUE),
    release = function(x) invisible(TRUE),
    error_translate = cudaverse:::.backend_default_error_translate(
      "contract-test"
    )
  )
  cudaverse:::.backend_register(factory, replace = TRUE)
  on.exit(
    rm(list = "contract-test", envir = cudaverse:::.cudaverse_backends),
    add = TRUE
  )

  condition <- tryCatch(
    cudaverse:::.backend_call("contract-test", "reduce", NULL),
    error = identity
  )
  expect_s3_class(condition, "cudaverse_backend_capability_error")
  expect_s3_class(condition, "cudaverse_backend_error")
  expect_identical(condition$backend, "contract-test")
  expect_identical(condition$operation, "reduce")
})

test_that("optional backend operations are discovered without changing contract", {
  expect_false(cudaverse:::.backend_has_operation("base", "missing_method"))
  expect_true(cudaverse:::.backend_has_operation("base", "algorithm_distance"))
})
