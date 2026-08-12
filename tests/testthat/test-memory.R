test_that("CPU memory reports are explicit and printable", {
  info <- cuda_memory_info("cpu")

  expect_s3_class(info, "cuda_memory_info")
  expect_false(info$available)
  expect_identical(info$requested_device, "cpu")
  expect_identical(info$device, "cpu")
  expect_identical(info$backend, "base")
  expect_identical(info$reason, "cpu_backend_selected")
  expect_true(all(vapply(
    info[c(
      "total_bytes", "free_bytes", "used_bytes", "allocated_bytes",
      "allocated_peak_bytes", "reserved_bytes", "reserved_peak_bytes"
    )],
    function(value) is.numeric(value) && length(value) == 1L && is.na(value),
    logical(1L)
  )))
  expect_output(print(info), "available=FALSE device=cpu backend=base")
})

test_that("automatic CPU fallback remains a structured memory report", {
  selection <- structure(
    list(
      requested_device = "auto",
      device = "cpu",
      backend = "base",
      selection_reason = "driver_unavailable",
      fallback = TRUE
    ),
    class = "cuda_device_selection"
  )
  testthat::local_mocked_bindings(
    cuda_select_device = function(device) selection,
    .package = "cudaverse"
  )

  info <- cuda_memory_info("auto")

  expect_false(info$available)
  expect_true(info$fallback)
  expect_identical(info$selection_reason, "driver_unavailable")
  expect_identical(info$reason, "cpu_backend_selected")
  expect_null(info$error)
})

test_that("memory reports normalize optional backend telemetry", {
  selection <- structure(
    list(
      requested_device = "auto",
      device = "cuda",
      backend = "observed",
      selection_reason = "test_backend",
      fallback = FALSE
    ),
    class = "cuda_device_selection"
  )
  snapshot <- list(
    available = TRUE,
    total_bytes = 16 * 1024^3,
    free_bytes = 12 * 1024^3,
    used_bytes = 4 * 1024^3,
    allocated_bytes = 1024,
    allocated_peak_bytes = 2048,
    reserved_bytes = 4096,
    reserved_peak_bytes = 8192,
    reason = "test_reported"
  )
  testthat::local_mocked_bindings(
    cuda_select_device = function(device) selection,
    .backend_has_operation = function(backend, operation) TRUE,
    .backend_call = function(backend, operation, ...) snapshot,
    .package = "cudaverse"
  )

  info <- cuda_memory_info("auto")

  expect_s3_class(info, "cuda_memory_info")
  expect_true(info$available)
  expect_identical(info$backend, "observed")
  expect_identical(info$total_bytes, 16 * 1024^3)
  expect_identical(info$allocated_peak_bytes, 2048)
  expect_null(info$error)
  expect_output(print(info), "Physical: 4.00 GiB used / 16.00 GiB total")
  expect_output(print(info), "1.00 KiB current / 2.00 KiB peak")
})

test_that("memory query failures remain structured diagnostics", {
  selection <- structure(
    list(
      requested_device = "auto",
      device = "cuda",
      backend = "failing",
      selection_reason = "test_backend",
      fallback = FALSE
    ),
    class = "cuda_device_selection"
  )
  testthat::local_mocked_bindings(
    cuda_select_device = function(device) selection,
    .backend_has_operation = function(backend, operation) TRUE,
    .backend_call = function(backend, operation, ...) {
      stop("injected telemetry failure", call. = FALSE)
    },
    .package = "cudaverse"
  )

  info <- cuda_memory_info("auto")

  expect_false(info$available)
  expect_identical(info$reason, "memory_query_failed")
  expect_match(info$error, "injected telemetry failure")
  expect_output(print(info), "Error: injected telemetry failure")
})

test_that("invalid backend memory metadata is rejected", {
  selection <- structure(
    list(
      requested_device = "auto",
      device = "cuda",
      backend = "invalid",
      selection_reason = "test_backend",
      fallback = FALSE
    ),
    class = "cuda_device_selection"
  )
  snapshot <- cudaverse:::.backend_empty_memory_info("invalid")
  snapshot$available <- NA

  info <- cudaverse:::.cuda_memory_result(snapshot, selection)

  expect_false(info$available)
  expect_identical(info$reason, "invalid_backend_report")
  expect_match(info$error, "invalid memory availability metadata")
})

test_that("torch memory telemetry follows the common byte contract", {
  skip_if_not_installed("torch")
  skip_if_not(isTRUE(tryCatch(
    torch::cuda_is_available(), error = function(error) FALSE
  )))

  info <- cudaverse:::.torch_memory_info()

  expect_true(info$available)
  expect_identical(info$reason, "torch_allocator_reported")
  expect_gte(info$allocated_bytes, 0)
  expect_gte(info$allocated_peak_bytes, info$allocated_bytes)
  expect_gte(info$reserved_bytes, info$allocated_bytes)
  expect_gte(info$reserved_peak_bytes, info$reserved_bytes)
  expect_true(all(is.na(c(
    info$total_bytes, info$free_bytes, info$used_bytes
  ))))
})
