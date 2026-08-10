test_that("benchmark contract contains the complete smoke and full profiles", {
  path <- system.file("benchmarks", "contract.csv", package = "cudaverse")
  expect_true(nzchar(path))
  contract <- utils::read.csv(
    path, stringsAsFactors = FALSE, check.names = FALSE,
    na.strings = c("", "NA")
  )

  expect_identical(anyDuplicated(contract$case_id), 0L)
  expect_setequal(contract$profile, c("smoke", "full"))
  expect_setequal(
    contract$family,
    c("matmul", "dense_pca_knn", "sparse_pca_knn")
  )
  full <- contract[contract$profile == "full", , drop = FALSE]
  expect_true(all(full$warmups == 5L))
  expect_true(all(full$timed_runs == 10L))
  expect_setequal(
    full$rows[full$family == "matmul"],
    c(256L, 1024L, 4096L)
  )
  expect_setequal(
    paste(full$rows[full$family == "dense_pca_knn"],
          full$columns[full$family == "dense_pca_knn"], sep = "x"),
    c("1000x50", "10000x100", "50000x128")
  )
  expect_setequal(
    paste(full$rows[full$family == "sparse_pca_knn"],
          full$columns[full$family == "sparse_pca_knn"], sep = "x"),
    c("1000x50", "10000x100", "50000x128")
  )
  expect_true(all(full$k[!is.na(full$k)] == 15L))
  expect_true(all(full$dtype[full$family != "matmul"] == "float64"))
})
