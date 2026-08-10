test_that("backend capability matrix covers the exported API", {
  path <- system.file(
    "reports", "backend-capability-matrix.csv", package = "cudaverse"
  )
  expect_true(nzchar(path))
  matrix <- utils::read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)

  expect_setequal(matrix$api, getNamespaceExports("cudaverse"))
  expect_identical(anyDuplicated(matrix$api), 0L)
  expect_false(anyNA(matrix))
  expect_true(all(nzchar(as.matrix(matrix))))

  allowed <- c("direct", "hybrid", "cpu_only", "metadata", "probe", "host")
  for (column in c("cpu_base", "torch_cuda", "native_cuda")) {
    expect_setequal(intersect(unique(matrix[[column]]), allowed), unique(matrix[[column]]))
  }
  expect_setequal(
    unique(matrix$contract_case),
    c("diagnostics", "tensor", "sparse", "algorithm", "graph", "embedding")
  )
})
