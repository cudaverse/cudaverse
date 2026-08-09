test_matrix <- function() {
  matrix(c(
    1, 2, 3,
    2, 3, 4,
    4, 2, 1,
    5, 3, 2,
    8, 7, 9,
    9, 8, 8
  ), ncol = 3, byrow = TRUE)
}

test_that("SVD reconstructs the input", {
  x <- test_matrix()
  fit <- cuda_svd(x, device = "cpu")
  reconstructed <- fit$u %*% diag(fit$d) %*% t(fit$v)

  expect_equal(reconstructed, x, tolerance = 1e-10)
  expect_identical(fit$device, "cpu")
})

test_that("PCA matches prcomp variance and dimensions", {
  x <- test_matrix()
  fit <- cuda_pca(x, n_components = 2, device = "cpu")
  expected <- stats::prcomp(x, rank. = 2)

  expect_s3_class(fit, "cuda_pca")
  expect_identical(dim(fit$x), c(6L, 2L))
  expect_equal(unname(fit$sdev), expected$sdev[1:2])
  expect_identical(names(fit$sdev), c("PC1", "PC2"))
})

test_that("fitted PCA models project new observations safely", {
  train <- test_matrix()
  colnames(train) <- paste0("feature_", seq_len(ncol(train)))
  rownames(train) <- paste0("train_", seq_len(nrow(train)))
  newdata <- train[c(2L, 5L), , drop = FALSE] + 0.25
  rownames(newdata) <- c("new_a", "new_b")

  fit <- cuda_pca(
    train,
    n_components = 2,
    center = TRUE,
    scale. = TRUE,
    device = "cpu"
  )
  expected <- sweep(newdata, 2L, fit$center, "-")
  expected <- sweep(expected, 2L, fit$scale, "/") %*% fit$rotation
  projected <- predict(
    fit,
    newdata[, rev(colnames(newdata)), drop = FALSE],
    device = "cpu"
  )

  expect_equal(as.vector(projected), as.vector(expected), tolerance = 1e-12)
  expect_identical(
    dimnames(projected),
    list(rownames(newdata), c("PC1", "PC2"))
  )
  expect_identical(attr(projected, "device"), "cpu")
  expect_identical(predict(fit), fit$x)
})

test_that("PCA prediction validates the model and feature identity", {
  x <- test_matrix()
  colnames(x) <- c("a", "b", "c")
  fit <- cuda_pca(x, n_components = 2, device = "cpu")
  newdata <- x[1L, , drop = FALSE]

  expect_error(
    predict(fit, unname(newdata), device = "cpu"),
    "must have column names"
  )
  colnames(newdata) <- c("a", "b", "other")
  expect_error(
    predict(fit, newdata, device = "cpu"),
    "missing: c; unexpected: other"
  )
  expect_error(
    predict(fit, x[1L, 1:2, drop = FALSE], device = "cpu"),
    "exactly 3 model features"
  )
  expect_error(
    predict(fit, x[1L, , drop = FALSE], typo = TRUE),
    "Unused argument"
  )

  broken <- fit
  broken$scale <- 0
  expect_error(
    predict(broken, x[1L, , drop = FALSE], device = "cpu"),
    "invalid `\\$scale`"
  )

  broken <- fit
  broken$device <- "accelerator"
  expect_error(
    predict(broken, x[1L, , drop = FALSE], device = "cpu"),
    "valid `\\$device`"
  )

  broken <- fit
  rownames(broken$rotation)[1L] <- NA_character_
  expect_error(
    predict(broken, x[1L, , drop = FALSE], device = "cpu"),
    "invalid feature names"
  )

  broken <- fit
  names(broken$center) <- rev(names(broken$center))
  expect_error(
    predict(broken, x[1L, , drop = FALSE], device = "cpu"),
    "names do not match"
  )
})

test_that("PCA retrieval validates stored training scores", {
  fit <- cuda_pca(test_matrix(), n_components = 2, device = "cpu")

  broken <- fit
  broken$x[1L, 1L] <- Inf
  expect_error(
    predict(broken),
    "invalid stored training scores"
  )

  broken <- fit
  broken$x <- broken$x[, 1L, drop = FALSE]
  expect_error(
    predict(broken),
    "invalid stored training scores"
  )

  broken <- fit
  colnames(broken$x) <- c("wrong_1", "wrong_2")
  expect_error(
    predict(broken),
    "invalid stored training scores"
  )
})

test_that("algorithms preserve observation and feature identifiers", {
  x <- test_matrix()
  rownames(x) <- paste0("cell_", seq_len(nrow(x)))
  colnames(x) <- paste0("feature_", seq_len(ncol(x)))

  decomposition <- cuda_svd(x, nu = 2, nv = 2, device = "cpu")
  expect_identical(rownames(decomposition$u), rownames(x))
  expect_identical(rownames(decomposition$v), colnames(x))
  expect_identical(colnames(decomposition$u), c("SVD1", "SVD2"))

  pca <- cuda_pca(x, n_components = 2, device = "cpu")
  expect_identical(rownames(pca$x), rownames(x))
  expect_identical(rownames(pca$rotation), colnames(x))
  expect_identical(colnames(pca$x), c("PC1", "PC2"))
  expect_identical(names(pca$center), colnames(x))

  distance <- cuda_distance(x, device = "cpu")
  expect_identical(dimnames(distance), list(rownames(x), rownames(x)))

  neighbors <- cuda_knn(x, k = 2, device = "cpu", batch_size = 2)
  expect_identical(rownames(neighbors$index), rownames(x))
  neighbor_labels <- matrix(
    rownames(neighbors$index)[as.vector(neighbors$index)],
    nrow = nrow(neighbors$index),
    dimnames = dimnames(neighbors$index)
  )
  expect_identical(dim(neighbor_labels), dim(neighbors$index))
  expect_true(all(neighbor_labels %in% rownames(x)))
  expect_identical(dimnames(neighbors$distance), dimnames(neighbors$index))

  clusters <- cuda_kmeans(x, centers = 2, seed = 1, device = "cpu")
  expect_identical(names(clusters$cluster), rownames(x))
  expect_identical(colnames(clusters$centers), colnames(x))
  expect_identical(rownames(clusters$centers), c("cluster_1", "cluster_2"))
  expect_identical(names(clusters$withinss), rownames(clusters$centers))
})

test_that("distance supports Euclidean and cosine metrics", {
  x <- test_matrix()
  euclidean <- cuda_distance(x, device = "cpu")
  cosine <- cuda_distance(x, metric = "cosine", device = "cpu")

  expect_equal(euclidean, as.matrix(stats::dist(x)), tolerance = 1e-10,
               ignore_attr = TRUE)
  expect_equal(diag(cosine), rep(0, nrow(x)), tolerance = 1e-10)
})

test_that("CUDA self-distance diagonals are exactly zero", {
  skip_if_not(cuda_available())
  x <- matrix(seq(0.1, 3, length.out = 35), nrow = 7)

  for (metric in c("euclidean", "cosine")) {
    distance <- cuda_distance(x, metric = metric, device = "cuda")
    expect_identical(diag(distance), numeric(nrow(x)))
  }
})

test_that("distance supports a single query observation", {
  x <- matrix(c(1, 2, 3), nrow = 1)
  y <- matrix(c(1, 2, 4, 3, 2, 1), nrow = 2, byrow = TRUE)

  between <- cuda_distance(x, y, device = "cpu")
  self <- cuda_distance(x, device = "cpu")

  expect_identical(dim(between), c(1L, 2L))
  expect_equal(as.vector(between), c(1, sqrt(8)), tolerance = 1e-12)
  expect_identical(dim(self), c(1L, 1L))
  expect_equal(as.vector(self), 0)
})

test_that("CPU Euclidean distance is stable for large offsets and magnitudes", {
  centers <- matrix(1e8 + c(0, 1), ncol = 1)
  query <- matrix(1e8 + 0.9, nrow = 1)
  offset_distance <- cuda_distance(query, centers, device = "cpu")

  expect_true(all(as.vector(offset_distance) > 0))
  expect_equal(
    as.vector(offset_distance),
    c(0.9, 0.1),
    tolerance = 1e-7
  )

  huge <- cuda_distance(
    matrix(c(1e300, 1e300), nrow = 1),
    matrix(c(0, 0), nrow = 1),
    device = "cpu"
  )
  tiny <- cuda_distance(
    matrix(c(1e-300, 1e-300), nrow = 1),
    matrix(c(0, 0), nrow = 1),
    device = "cpu"
  )
  expect_true(is.finite(huge[[1L]]))
  expect_equal(huge[[1L]], sqrt(2) * 1e300, tolerance = 1e-12)
  expect_equal(tiny[[1L]], sqrt(2) * 1e-300, tolerance = 1e-12)
})

test_that("CPU Euclidean distance repairs risky pairs after extreme scaling", {
  largest <- .Machine$double.xmax
  x <- rbind(
    extreme = c(-largest, 0),
    tiny = c(1e-300, 1e-300)
  )
  y <- rbind(
    opposite = c(largest, 0),
    zero = c(0, 0)
  )
  distance <- cuda_distance(x, y, device = "cpu")

  expect_true(is.infinite(distance["extreme", "opposite"]))
  expect_equal(distance["tiny", "zero"], sqrt(2) * 1e-300, tolerance = 1e-12)

  dimensions <- 10000L
  high_dimensional <- cuda_distance(
    matrix(rep(1e300, dimensions), nrow = 1L),
    matrix(rep(0, dimensions), nrow = 1L),
    device = "cpu"
  )
  expect_true(is.finite(high_dimensional[[1L]]))
  expect_equal(high_dimensional[[1L]], 1e302, tolerance = 1e-12)
})

test_that("cosine distance rejects zero rows before backend dispatch", {
  zero_x <- rbind(c(0, 0), c(1, 0))
  zero_y <- rbind(c(1, 0), c(0, 0))
  valid <- rbind(c(1, 0), c(0, 1))

  expect_error(
    cuda_distance(zero_x, metric = "cosine", device = "cpu"),
    "zero-length rows"
  )
  expect_error(
    cuda_distance(valid, zero_y, metric = "cosine", device = "cpu"),
    "zero-length rows"
  )
  expect_error(
    cuda_distance(zero_x, metric = "cosine", device = "cuda"),
    "zero-length rows"
  )
})

test_that("cosine normalization is stable across extreme finite scales", {
  x <- rbind(
    c(1e300, 1e300),
    c(1e-300, 0)
  )
  distance <- cuda_distance(x, metric = "cosine", device = "cpu")

  expect_true(all(is.finite(distance)))
  expect_equal(diag(distance), c(0, 0), tolerance = 1e-12)
  expect_equal(
    distance[1, 2],
    1 - 1 / sqrt(2),
    tolerance = 1e-12
  )
})

test_that("k-nearest neighbours exclude each observation", {
  fit <- cuda_knn(test_matrix(), k = 2, device = "cpu")

  expect_named(
    fit,
    c(
      "index",
      "distance",
      "metric",
      "device",
      "provenance_schema",
      "requested_device",
      "compute_device",
      "compute_stages",
      "backend",
      "parameters",
      "source_device",
      "source_class"
    )
  )
  expect_identical(dim(fit$index), c(6L, 2L))
  expect_false(any(fit$index == row(fit$index)))
  expect_true(all(fit$distance >= 0))
})

test_that("batched exact neighbours match full pairwise distances", {
  x <- test_matrix()
  reference_index <- seq_len(nrow(x))

  for (metric in c("euclidean", "cosine")) {
    full_distance <- cuda_distance(x, metric = metric, device = "cpu")
    expected_index <- vapply(
      reference_index,
      function(i) {
        candidates <- reference_index[-i]
        ordering <- order(
          full_distance[i, candidates],
          candidates,
          method = "radix"
        )
        candidates[ordering[1:2]]
      },
      integer(2)
    )
    expected_index <- t(expected_index)
    expected_distance <- matrix(
      full_distance[cbind(
        rep(reference_index, each = 2L),
        as.vector(t(expected_index))
      )],
      nrow = nrow(x),
      byrow = TRUE
    )

    for (batch_size in c(1L, 2L, 100L)) {
      fit <- cuda_knn(
        x,
        k = 2,
        metric = metric,
        device = "cpu",
        batch_size = batch_size
      )

      expect_identical(fit$index, expected_index)
      expect_equal(fit$distance, expected_distance, tolerance = 1e-12)
      expect_identical(fit$metric, metric)
      expect_identical(fit$device, "cpu")
    }
  }
})

test_that("distance blocks never exceed the requested query batch", {
  x <- test_matrix()
  state <- cudaverse:::.knn_distance_state(
    x,
    metric = "euclidean",
    device = "cpu"
  )
  blocks <- list(1:2, 3:4, 5:6)
  distance <- lapply(
    blocks,
    function(rows) cudaverse:::.knn_distance_block(state, rows)
  )

  expect_true(all(vapply(distance, nrow, integer(1)) <= 2L))
  expect_true(all(vapply(distance, ncol, integer(1)) == nrow(x)))
  expect_equal(
    do.call(rbind, distance),
    cuda_distance(x, device = "cpu"),
    tolerance = 1e-12,
    ignore_attr = TRUE
  )
})

test_that("kNN ties and self exclusion are deterministic", {
  tied <- matrix(c(0, 2, 4), ncol = 1)
  fit <- cuda_knn(
    tied,
    k = 1,
    device = "cpu",
    batch_size = 1
  )

  expect_identical(as.vector(fit$index), c(2L, 1L, 2L))
  expect_equal(as.vector(fit$distance), c(2, 2, 2))

  duplicated <- matrix(c(0, 0, 1), ncol = 1)
  duplicate_fit <- cuda_knn(
    duplicated,
    k = 1,
    device = "cpu",
    batch_size = 2
  )

  expect_identical(as.vector(duplicate_fit$index), c(2L, 1L, 1L))
  expect_false(any(duplicate_fit$index == row(duplicate_fit$index)))
})

test_that("CPU kNN preserves close distances on a large offset", {
  x <- matrix(1e8 + c(0, 0.9, 1), ncol = 1)
  fit <- cuda_knn(
    x,
    k = 1,
    device = "cpu",
    batch_size = 2
  )

  expect_identical(as.vector(fit$index), c(2L, 3L, 2L))
  expect_equal(
    as.vector(fit$distance),
    c(0.9, 0.1, 0.1),
    tolerance = 1e-7
  )
  expect_true(all(fit$distance > 0))
})

test_that("kNN validates batch sizes and cosine rows clearly", {
  x <- test_matrix()

  for (batch_size in list(0, -1, 1.5, Inf, NA_real_, numeric())) {
    expect_error(
      cuda_knn(x, k = 2, batch_size = batch_size, device = "cpu"),
      "positive whole number"
    )
  }
  expect_error(
    cuda_knn(x, k = Inf, device = "cpu"),
    "between 1 and nrow"
  )

  zero <- rbind(c(0, 0), c(1, 0), c(0, 1))
  expect_error(
    cuda_knn(
      zero,
      k = 1,
      metric = "cosine",
      device = "cpu",
      batch_size = 1
    ),
    "zero-length rows"
  )
})

test_that("CUDA batched neighbours agree with CPU reference", {
  skip_if_not(cuda_available())
  x <- test_matrix()
  cpu <- cuda_knn(x, k = 2, device = "cpu", batch_size = 2)
  gpu <- cuda_knn(x, k = 2, device = "cuda", batch_size = 2)

  expect_identical(gpu$index, cpu$index)
  expect_equal(gpu$distance, cpu$distance, tolerance = 1e-8)
  expect_identical(gpu$device, "cuda")
})

test_that("k-means returns coherent clusters", {
  set.seed(1)
  x <- rbind(
    matrix(rnorm(40, 0, 0.2), 20, 2),
    matrix(rnorm(40, 5, 0.2), 20, 2)
  )
  fit <- cuda_kmeans(x, 2, seed = 1, device = "cpu")

  expect_s3_class(fit, "cuda_kmeans")
  expect_length(fit$cluster, 40)
  expect_identical(dim(fit$centers), c(2L, 2L))
  expect_true(all(fit$cluster %in% 1:2))
})

test_that("CPU k-means separates close groups on a large offset", {
  x <- matrix(1e8 + c(0, 0.1, 0.9, 1), ncol = 1)
  initial <- matrix(1e8 + c(0, 1), ncol = 1)
  fit <- cuda_kmeans(
    x,
    centers = initial,
    device = "cpu"
  )

  expect_identical(as.vector(fit$cluster), c(1L, 1L, 2L, 2L))
  expect_equal(
    as.vector(fit$centers),
    1e8 + c(0.05, 0.95),
    tolerance = 1e-7
  )
  expect_true(all(fit$withinss > 0))
})

test_that("fitted k-means models assign new observations safely", {
  x <- test_matrix()
  colnames(x) <- c("a", "b", "c")
  rownames(x) <- paste0("train_", seq_len(nrow(x)))
  fit <- cuda_kmeans(x, centers = 2, seed = 1, device = "cpu")
  newdata <- rbind(
    new_a = fit$centers[1L, ] + 0.01,
    new_b = fit$centers[2L, ] - 0.01
  )

  distances <- predict(
    fit,
    newdata[, c("c", "a", "b"), drop = FALSE],
    type = "distance",
    device = "cpu"
  )
  clusters <- predict(
    fit,
    newdata[, c("c", "a", "b"), drop = FALSE],
    device = "cpu"
  )

  expect_identical(dim(distances), c(2L, 2L))
  expect_identical(
    dimnames(distances),
    list(rownames(newdata), rownames(fit$centers))
  )
  expect_identical(as.vector(clusters), c(1L, 2L))
  expect_identical(names(clusters), rownames(newdata))
  expect_identical(predict(fit), fit$cluster)
})

test_that("k-means prediction handles one row and rejects ambiguity", {
  x <- test_matrix()
  colnames(x) <- c("a", "b", "c")
  fit <- cuda_kmeans(x, centers = 2, seed = 1, device = "cpu")
  one <- x[1L, , drop = FALSE]

  expect_length(predict(fit, one, device = "cpu"), 1L)
  expect_identical(
    dim(predict(fit, one, type = "distance", device = "cpu")),
    c(1L, 2L)
  )
  expect_error(
    predict(fit, type = "distance"),
    "`newdata` is required"
  )
  expect_error(
    predict(fit, unname(one), device = "cpu"),
    "must have column names"
  )
  colnames(one) <- c("a", "b", "other")
  expect_error(
    predict(fit, one, device = "cpu"),
    "feature names do not match"
  )

  broken <- fit
  broken$centers[1L, 1L] <- NA_real_
  expect_error(
    predict(broken, x[1L, , drop = FALSE], device = "cpu"),
    "invalid centres"
  )

  broken <- fit
  broken$device <- NA_character_
  expect_error(
    predict(broken, x[1L, , drop = FALSE], device = "cpu"),
    "valid `\\$device`"
  )

  broken <- fit
  colnames(broken$centers)[1L] <- NA_character_
  expect_error(
    predict(broken, x[1L, , drop = FALSE], device = "cpu"),
    "invalid feature names"
  )
})

test_that("k-means retrieval validates stored training assignments", {
  fit <- cuda_kmeans(
    test_matrix(),
    centers = 2,
    seed = 1,
    device = "cpu"
  )

  broken <- fit
  broken$cluster <- as.numeric(broken$cluster)
  expect_error(
    predict(broken),
    "invalid stored assignments"
  )

  broken <- fit
  broken$cluster[1L] <- 0L
  expect_error(
    predict(broken),
    "invalid stored assignments"
  )

  broken <- fit
  broken$cluster <- integer()
  expect_error(
    predict(broken),
    "invalid stored assignments"
  )
})

test_that("saved CUDA models can be predicted explicitly on CPU", {
  x <- test_matrix()
  colnames(x) <- c("a", "b", "c")
  newdata <- x[1:2, , drop = FALSE]

  cpu_pca <- cuda_pca(x, n_components = 2, device = "cpu")
  saved_cuda_pca <- cpu_pca
  saved_cuda_pca$device <- "cuda"
  pca_override <- predict(saved_cuda_pca, newdata, device = "cpu")
  expect_equal(
    as.vector(pca_override),
    as.vector(predict(cpu_pca, newdata, device = "cpu")),
    tolerance = 1e-12
  )
  expect_identical(
    cuda_provenance(pca_override)$requested_device,
    "cpu"
  )
  expect_identical(
    cuda_provenance(pca_override)$selection_reason,
    "explicit_cpu"
  )

  cpu_kmeans <- cuda_kmeans(x, centers = 2, seed = 1, device = "cpu")
  saved_cuda_kmeans <- cpu_kmeans
  saved_cuda_kmeans$device <- "cuda"
  kmeans_override <- predict(saved_cuda_kmeans, newdata, device = "cpu")
  expect_identical(
    as.vector(kmeans_override),
    as.vector(predict(cpu_kmeans, newdata, device = "cpu"))
  )
  expect_identical(
    cuda_provenance(kmeans_override)$requested_device,
    c("cpu", "fixed-cpu")
  )
  expect_identical(
    cuda_provenance(kmeans_override)$selection_reason,
    c("explicit_cpu", "algorithm_cpu_only")
  )
})

test_that("CUDA predictions agree with CPU predictions", {
  skip_if_not(cuda_available())
  x <- test_matrix()
  colnames(x) <- c("a", "b", "c")
  rownames(x) <- paste0("row_", seq_len(nrow(x)))
  newdata <- x[1:2, c("c", "a", "b"), drop = FALSE]

  pca <- cuda_pca(x, n_components = 2, device = "cpu")
  cpu_pca <- predict(pca, newdata, device = "cpu")
  gpu_pca <- predict(pca, newdata, device = "cuda")
  expect_equal(
    as.vector(gpu_pca),
    as.vector(cpu_pca),
    tolerance = 1e-8
  )
  expect_identical(attr(gpu_pca, "device"), "cuda")

  kmeans <- cuda_kmeans(x, centers = 2, seed = 1, device = "cpu")
  cpu_cluster <- predict(kmeans, newdata, device = "cpu")
  gpu_cluster <- predict(kmeans, newdata, device = "cuda")
  expect_identical(as.vector(gpu_cluster), as.vector(cpu_cluster))
  expect_identical(attr(gpu_cluster, "device"), "cuda")

  cuda_pca_model <- cuda_pca(x, n_components = 2, device = "cuda")
  inherited_pca <- predict(cuda_pca_model, newdata)
  explicit_pca <- predict(cuda_pca_model, newdata, device = "cuda")
  expect_equal(
    as.vector(inherited_pca),
    as.vector(explicit_pca),
    tolerance = 1e-8
  )
  expect_identical(
    cuda_provenance(inherited_pca)$requested_device,
    "inherited"
  )

  cuda_kmeans_model <- cuda_kmeans(
    x,
    centers = 2,
    seed = 1,
    device = "cuda"
  )
  inherited_cluster <- predict(cuda_kmeans_model, newdata)
  explicit_cluster <- predict(
    cuda_kmeans_model,
    newdata,
    device = "cuda"
  )
  expect_identical(
    as.vector(inherited_cluster),
    as.vector(explicit_cluster)
  )
  expect_identical(
    cuda_provenance(inherited_cluster)$requested_device,
    c("inherited", "fixed-cpu")
  )
})

test_that("k-means final assignments and sums match returned centers", {
  set.seed(1)
  x <- matrix(rnorm(60), 30, 2)
  fit <- cuda_kmeans(
    x,
    centers = 3,
    iter.max = 1,
    seed = 1001,
    device = "cpu"
  )
  distances <- cuda_distance(x, fit$centers, device = "cpu")
  expected_cluster <- max.col(-distances, ties.method = "first")
  expected_withinss <- vapply(
    seq_len(nrow(fit$centers)),
    function(group) {
      members <- which(expected_cluster == group)
      sum(distances[cbind(members, rep.int(group, length(members)))]^2)
    },
    numeric(1)
  )

  expect_identical(fit$cluster, expected_cluster)
  expect_equal(fit$withinss, expected_withinss)
  expect_equal(fit$tot.withinss, sum(expected_withinss))
  expect_identical(fit$iter, 1L)
  expect_false(fit$converged)
})

test_that("k-means handles ties and empty clusters deterministically", {
  tied <- matrix(c(-2, 0, 1), ncol = 1)
  tied_fit <- cuda_kmeans(
    tied,
    centers = matrix(c(-1, 1), ncol = 1),
    device = "cpu"
  )

  expect_identical(tied_fit$cluster, c(1L, 1L, 2L))

  duplicated <- matrix(1, nrow = 3, ncol = 1)
  empty_fit <- cuda_kmeans(
    duplicated,
    centers = matrix(c(1, 1), ncol = 1),
    device = "cpu"
  )

  expect_identical(empty_fit$cluster, rep(1L, 3))
  expect_equal(empty_fit$centers, matrix(c(1, 1), ncol = 1))
  expect_identical(empty_fit$withinss, c(0, 0))
  expect_true(empty_fit$converged)
})

test_that("seeded k-means does not mutate the caller RNG state", {
  x <- test_matrix()
  set.seed(99)
  before <- .Random.seed

  cuda_kmeans(x, 2, seed = 1, device = "cpu")

  expect_identical(.Random.seed, before)
  expect_error(
    cuda_kmeans(x, 2, seed = 1.5, device = "cpu"),
    "whole number"
  )
})

test_that("invalid algorithm inputs fail clearly", {
  x <- test_matrix()
  expect_error(cuda_pca(x, n_components = 10, device = "cpu"), "between")
  expect_error(cuda_knn(x, k = nrow(x), device = "cpu"), "nrow")
  expect_error(
    cuda_distance(x, matrix(1:8, 4, 2), device = "cpu"),
    "same number of columns"
  )
  expect_error(cuda_pca(x, center = NA, device = "cpu"), "TRUE or FALSE")
  expect_error(cuda_pca(x, scale. = 1, device = "cpu"), "TRUE or FALSE")
})
