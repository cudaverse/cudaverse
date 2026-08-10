test_that("existing sparse objects honor a new drop_zeros request", {
  source <- Matrix::sparseMatrix(
    i = c(1L, 2L),
    j = c(1L, 2L),
    x = c(0, 2),
    dims = c(2L, 2L)
  )
  retained <- cuda_sparse(
    source,
    device = "cpu",
    drop_zeros = FALSE
  )
  dropped <- cuda_sparse(
    retained,
    device = "cpu",
    drop_zeros = TRUE
  )

  expect_length(retained$values, 2L)
  expect_true(any(retained$values == 0))
  expect_length(dropped$values, 1L)
  expect_false(any(dropped$values == 0))
  expect_equal(as.matrix(to_dgCMatrix(dropped)), as.matrix(source))
})

test_that("provenance inspection re-exports the canonical generic", {
  expect_identical(cuda_provenance, cuda_provenance)
  expect_true(utils::isS3stdGeneric(cuda_provenance))
})

test_that("sparse construction exposes the shared provenance contract", {
  x <- cuda_sparse(diag(3), device = "cpu")
  provenance <- cuda_provenance(x)

  expect_s3_class(provenance, "cuda_provenance")
  expect_identical(provenance$stage, "sparse_materialization")
  expect_identical(provenance$requested_device, "cpu")
  expect_identical(provenance$device, "cpu")
  expect_identical(provenance$backend, "Matrix")
  expect_identical(provenance$selection_reason, "explicit_cpu")
  expect_identical(provenance$output_device, "cpu")
  expect_identical(x$provenance_schema, "cudaverse-stage/1")
  expect_identical(x$compute_device, "cpu")

  info <- sparse_info(x)
  expect_identical(info$provenance_schema, "cudaverse-stage/1")
  expect_identical(info$compute_device, "cpu")
})

test_that("sparse transpose records same-device backend execution", {
  x <- cuda_sparse(matrix(c(1, 0, 2, 0, 3, 0), 2), device = "cpu")
  result <- t(x)
  provenance <- cuda_provenance(result)

  expect_identical(provenance$stage, "sparse_transpose")
  expect_identical(provenance$device, "cpu")
  expect_identical(provenance$backend, "Matrix")
  expect_identical(provenance$output_device, "cpu")
  expect_identical(result$compute_device, "cpu")
})

test_that("sparse products and reductions report actual CPU stages", {
  x <- cuda_sparse(diag(3), device = "cpu")
  product <- sparse_matmul_dense(x, matrix(1:6, 3, 2))
  product_provenance <- cuda_provenance(product)

  expect_identical(product_provenance$stage, "sparse_multiply")
  expect_identical(product_provenance$device, "cpu")
  expect_identical(product_provenance$output_device, "cpu")
  expect_identical(product$device, "cpu")

  vector <- sparse_matvec(x, 1:3)
  expect_identical(cuda_provenance(vector)$stage, "sparse_multiply")

  row_result <- sparse_row_sums(x)
  column_result <- sparse_col_sums(x)
  expect_identical(cuda_provenance(row_result)$stage, "row_reduction")
  expect_identical(
    cuda_provenance(column_result)$stage,
    "column_reduction"
  )
  expect_identical(
    attr(cuda_provenance(row_result), "compute_device"),
    "cpu"
  )
})

test_that("Matrix materialization carries explicit CPU provenance", {
  x <- cuda_sparse(diag(3), device = "cpu")
  materialized <- to_dgCMatrix(x)
  provenance <- cuda_provenance(materialized)

  expect_s4_class(materialized, "dgCMatrix")
  expect_identical(provenance$stage, "sparse_materialization")
  expect_identical(provenance$selection_reason, "explicit_materialization")
  expect_identical(provenance$output_device, "cpu")
})

test_that("CUDA sparse normalization records CPU compute and upload", {
  factory <- cudaverse:::.base_backend_factory()
  factory$name <- "sparse-upload-test"
  factory$device <- "cuda"
  factory$sparse_from_coo <- function(i, j, values, shape, format = "csr") {
    list(i = i, j = j, values = values, shape = shape, format = format)
  }
  cudaverse:::.backend_register(factory, replace = TRUE)
  on.exit(
    rm(list = "sparse-upload-test", envir = cudaverse:::.cudaverse_backends),
    add = TRUE
  )

  x <- cuda_sparse(diag(3), device = "cpu")
  x$device <- "cuda"
  x$backend_id <- "sparse-upload-test"
  normalized <- sparse_normalize(x)
  provenance <- cuda_provenance(normalized)
  stages <- attr(provenance, "compute_stages", exact = TRUE)

  expect_named(stages, c("normalization", "normalization_upload"))
  expect_identical(stages$normalization$device, "cpu")
  expect_identical(stages$normalization$backend, "Matrix")
  expect_identical(stages$normalization$output_device, "cpu")
  expect_identical(stages$normalization_upload$device, "cuda")
  expect_identical(
    stages$normalization_upload$backend,
    "sparse-upload-test"
  )
  expect_identical(stages$normalization_upload$output_device, "cuda")
})
