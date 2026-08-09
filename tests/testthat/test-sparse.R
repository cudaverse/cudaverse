test_that("Matrix sparse inputs preserve values and dimensions", {
  source <- Matrix::sparseMatrix(
    i = c(1, 3, 4),
    j = c(2, 1, 3),
    x = c(2, 5, -1),
    dims = c(4, 3)
  )
  x <- cuda_sparse(source, device = "cpu")

  expect_s3_class(x, "cudasparse")
  expect_identical(dim(x), c(4L, 3L))
  expect_equal(as.matrix(to_dgCMatrix(x)), as.matrix(source))
  expect_equal(sparse_info(x)$nnz, 3)
})

test_that("sparse construction and conversion preserve dimension labels", {
  source <- Matrix::Matrix(
    matrix(
      c(1, 0, 2, 0, 3, 0),
      nrow = 2,
      dimnames = list(
        feature = c("gene_a", "gene_b"),
        sample = c("sample_a", "sample_b", "sample_c")
      )
    ),
    sparse = TRUE
  )
  x <- cuda_sparse(source, device = "cpu")

  expect_identical(dimnames(x), dimnames(source))
  expect_identical(dimnames(to_dgCMatrix(x)), dimnames(source))
  expect_identical(dimnames(as_coo(x)), dimnames(source))
  expect_identical(dimnames(as_csr(as_coo(x))), dimnames(source))

  unnamed <- cuda_sparse(unname(as.matrix(source)), device = "cpu")
  expect_null(dimnames(unnamed))
  expect_true(all(vapply(
    dimnames(to_dgCMatrix(unnamed)),
    is.null,
    logical(1)
  )))

  broken <- x
  broken$dimnames <- list("too_short")
  expect_error(to_dgCMatrix(broken), "one entry per sparse dimension")
})

test_that("COO and CSR metadata are consistent", {
  x <- cuda_sparse(diag(4), format = "csr", device = "cpu")
  expect_identical(sparse_info(as_coo(x))$format, "coo")
  expect_identical(sparse_info(as_csr(as_coo(x)))$format, "csr")
  expect_identical(x$row_ptr, 0:4)
})

test_that("sparse dense multiplication matches Matrix", {
  source <- Matrix::rsparsematrix(6, 4, density = 0.3)
  dense <- matrix(seq_len(12), 4, 3)
  x <- cuda_sparse(source, device = "cpu")

  product <- to_cpu(sparse_matmul_dense(x, dense))
  expect_equal(product, as.matrix(source %*% dense))
  expect_equal(
    as.numeric(sparse_matvec(x, 1:4)),
    as.vector(source %*% (1:4))
  )
})

test_that("sparse operations preserve compatible row and column labels", {
  source <- Matrix::Matrix(
    matrix(
      c(1, 0, 2, 0, 3, 0),
      nrow = 2,
      dimnames = list(
        feature = c("gene_a", "gene_b"),
        sample = c("sample_a", "sample_b", "sample_c")
      )
    ),
    sparse = TRUE
  )
  dense <- matrix(
    1:6,
    nrow = 3,
    dimnames = list(
      sample = c("sample_a", "sample_b", "sample_c"),
      component = c("PC1", "PC2")
    )
  )
  x <- cuda_sparse(source, device = "cpu")

  product <- sparse_matmul_dense(x, dense)
  expect_identical(
    dimnames(to_cpu(product)),
    list(
      feature = c("gene_a", "gene_b"),
      component = c("PC1", "PC2")
    )
  )
  expect_identical(
    names(sparse_matvec(x, setNames(1:3, colnames(source)))),
    rownames(source)
  )
  expect_identical(names(sparse_row_sums(x)), rownames(source))
  expect_identical(names(sparse_col_sums(x)), colnames(source))

  rownames(dense) <- rev(rownames(dense))
  expect_error(
    sparse_matmul_dense(x, dense),
    "inner dimension names are incompatible"
  )
})

test_that("sparse reductions match Matrix", {
  source <- Matrix::rsparsematrix(5, 4, density = 0.4)
  x <- cuda_sparse(source, device = "cpu")

  expect_equal(
    as.numeric(sparse_row_sums(x)),
    as.numeric(Matrix::rowSums(source))
  )
  expect_equal(
    as.numeric(sparse_col_sums(x)),
    as.numeric(Matrix::colSums(source))
  )
})

test_that("sparse normalization preserves structure and matches dense R", {
  source <- matrix(
    c(1, 0, 3, 2, 4, 0, 5, 1, 2, 3, 0, 6),
    nrow = 4,
    dimnames = list(
      observation = paste0("sample_", 1:4),
      feature = paste0("gene_", 1:3)
    )
  )
  sparse <- cuda_sparse(source, device = "cpu")

  rows <- sparse_normalize(
    sparse, margin = "rows", scale_factor = 100, log1p = FALSE
  )
  expected_rows <- source * 100 / rowSums(source)
  expect_s3_class(rows, "cudasparse")
  expect_identical(rows$i, sparse$i)
  expect_identical(rows$j, sparse$j)
  expect_identical(dimnames(rows), dimnames(sparse))
  expect_equal(as.matrix(to_dgCMatrix(rows)), expected_rows, tolerance = 1e-12)

  columns <- sparse_normalize(
    sparse, margin = "columns", scale_factor = 10, log1p = TRUE
  )
  expected_columns <- log1p(sweep(source, 2L, colSums(source), "/") * 10)
  expect_equal(
    as.matrix(to_dgCMatrix(columns)),
    expected_columns,
    tolerance = 1e-12
  )
  expect_identical(cuda_provenance(rows)$stage, "normalization")
})

test_that("PCA and kNN accept sparse inputs without changing CPU results", {
  set.seed(2026)
  source <- matrix(stats::rpois(240, lambda = 3), nrow = 40, ncol = 6)
  source[source < 2] <- 0
  rownames(source) <- paste0("sample_", seq_len(nrow(source)))
  colnames(source) <- paste0("feature_", seq_len(ncol(source)))
  sparse <- sparse_normalize(
    cuda_sparse(source, device = "cpu"),
    margin = "rows",
    scale_factor = 100,
    log1p = TRUE
  )
  dense <- as.matrix(to_dgCMatrix(sparse))

  sparse_pca <- cuda_pca(sparse, 3L, device = "cpu")
  dense_pca <- cuda_pca(dense, 3L, device = "cpu")
  expect_equal(
    tcrossprod(sparse_pca$rotation),
    tcrossprod(dense_pca$rotation),
    tolerance = 1e-10
  )
  expect_identical(rownames(sparse_pca$x), rownames(source))
  expect_true(all(c("normalization", "sparse_to_dense", "decomposition") %in%
                    cuda_provenance(sparse_pca)$stage))

  sparse_knn <- cuda_knn(sparse, k = 5L, device = "cpu", batch_size = 11L)
  dense_knn <- cuda_knn(dense, k = 5L, device = "cpu", batch_size = 11L)
  expect_identical(sparse_knn$index, dense_knn$index)
  expect_equal(sparse_knn$distance, dense_knn$distance, tolerance = 1e-12)
  expect_true(all(c("normalization", "sparse_to_dense", "distance",
                    "neighbor_selection") %in%
                    cuda_provenance(sparse_knn)$stage))
})

test_that("invalid inputs fail clearly", {
  expect_error(cuda_sparse(matrix(c(1, NA), 1), device = "cpu"), "finite")
  expect_error(cuda_sparse(letters[1:3], device = "cpu"), "matrix")
  expect_error(
    cuda_sparse(diag(2), device = "cpu", drop_zeros = NA),
    "TRUE or FALSE"
  )

  x <- cuda_sparse(diag(3), device = "cpu")
  expect_error(sparse_matvec(x, 1:2), "one value per column")
  expect_error(sparse_matmul_dense(x, matrix(1:8, 4)), "not conformable")
  expect_error(sparse_normalize(x, scale_factor = 0), "positive finite")
  expect_error(sparse_normalize(x, log1p = NA), "TRUE or FALSE")

  negative <- cuda_sparse(matrix(c(1, -1, 2, 3), 2), device = "cpu")
  expect_error(sparse_normalize(negative), "non-negative")
  empty_margin <- cuda_sparse(matrix(c(1, 0, 0, 0), 2), device = "cpu")
  expect_error(sparse_normalize(empty_margin), "positive finite sum")
})

test_that("large sparse printing avoids materializing all entries", {
  x <- cuda_sparse(diag(5), device = "cpu")
  old_options <- options(cudaverse.max_print = 3)
  on.exit(options(old_options), add = TRUE)

  output <- capture.output(print(x))

  expect_true(any(grepl("stored values omitted", output)))
  expect_false(any(grepl("5 x 5 sparse Matrix", output, fixed = TRUE)))
})

test_that("CUDA sparse conversion preserves dimension labels when available", {
  skip_if_not(cuda_available())
  source <- Matrix::Matrix(
    matrix(
      c(1, 0, 2, 0, 3, 0),
      nrow = 2,
      dimnames = list(
        feature = c("gene_a", "gene_b"),
        sample = c("sample_a", "sample_b", "sample_c")
      )
    ),
    sparse = TRUE
  )
  gpu <- cuda_sparse(source, device = "cuda")

  expect_identical(dimnames(gpu), dimnames(source))
  expect_identical(dimnames(to_dgCMatrix(gpu)), dimnames(source))
})
