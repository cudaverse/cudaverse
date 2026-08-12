stable_topk_reference <- function(distance, rows, k) {
  selected <- vapply(
    seq_along(rows),
    function(query) {
      candidates <- seq_len(ncol(distance))[-rows[[query]]]
      candidates[order(
        distance[query, candidates],
        candidates,
        method = "radix"
      )[seq_len(k)]]
    },
    integer(k)
  )
  index <- t(matrix(selected, nrow = k, ncol = length(rows)))
  selected_distance <- matrix(
    distance[cbind(
      rep(seq_along(rows), each = k),
      as.vector(t(index))
    )],
    nrow = length(rows),
    ncol = k,
    byrow = TRUE
  )
  list(index = index, distance = selected_distance)
}

test_that("CPU stable top-k uses distance then original row order", {
  distance <- rbind(
    c(0, 1, 1, 2, Inf, 1),
    c(2, 0, -0, 2, 2, 2),
    c(Inf, 4, 0, 4, 3, 3)
  )
  rows <- c(1L, 2L, 3L)
  expected <- stable_topk_reference(distance, rows, 4L)

  result <- .Call(
    cudaverse:::C_cudaverse_cpu_stable_topk,
    distance,
    rows,
    4L
  )

  expect_identical(result$index, expected$index)
  expect_identical(result$distance, expected$distance)
  expect_identical(result$index[1L, ], c(2L, 3L, 6L, 4L))
  expect_true(all(result$index != rows[row(result$index)]))
})

test_that("CPU stable top-k rejects malformed native inputs", {
  distance <- matrix(runif(12), 3L, 4L)

  expect_error(
    .Call(
      cudaverse:::C_cudaverse_cpu_stable_topk,
      matrix(as.integer(1:12), 3L, 4L),
      1:3,
      2L
    ),
    "double matrix"
  )
  distance[1L, 2L] <- NaN
  expect_error(
    .Call(
      cudaverse:::C_cudaverse_cpu_stable_topk,
      distance,
      1:3,
      2L
    ),
    "NA or NaN"
  )
  distance[1L, 2L] <- 1
  expect_error(
    .Call(
      cudaverse:::C_cudaverse_cpu_stable_topk,
      distance,
      c(1L, 2L, 5L),
      2L
    ),
    "outside"
  )
  expect_error(
    .Call(
      cudaverse:::C_cudaverse_cpu_stable_topk,
      distance,
      1:3,
      2
    ),
    "one integer"
  )
})

test_that("CPU stable top-k matches radix order across bounded random cases", {
  for (seed in seq_len(20L)) {
    set.seed(seed)
    candidate_count <- sample(4:25, 1L)
    query_count <- sample(seq_len(candidate_count), 1L)
    rows <- sample(seq_len(candidate_count), query_count)
    distance <- matrix(
      sample(c(seq(-2, 2, by = 0.25), Inf),
             query_count * candidate_count,
             replace = TRUE),
      query_count,
      candidate_count
    )
    for (k in unique(c(1L, min(3L, candidate_count - 1L),
                       candidate_count - 1L))) {
      expected <- stable_topk_reference(distance, rows, k)
      result <- .Call(
        cudaverse:::C_cudaverse_cpu_stable_topk,
        distance,
        as.integer(rows),
        as.integer(k)
      )

      expect_identical(result$index, expected$index)
      expect_identical(result$distance, expected$distance)
    }
  }
})

test_that("base kNN stable top-k matches the full exact reference", {
  set.seed(1801)
  values <- matrix(rnorm(180), 30L, 6L)
  values[c(8L, 19L), ] <- values[3L, ]

  for (metric in c("euclidean", "cosine")) {
    reference_values <- if (identical(metric, "cosine")) {
      values / sqrt(rowSums(values^2))
    } else {
      values
    }
    distance <- if (identical(metric, "euclidean")) {
      as.matrix(stats::dist(values))
    } else {
      pmin(pmax(1 - tcrossprod(reference_values), 0), 2)
    }
    expected <- stable_topk_reference(
      distance,
      seq_len(nrow(values)),
      7L
    )

    for (batch_size in c(1L, 7L, nrow(values))) {
      result <- cuda_knn(
        values,
        k = 7L,
        metric = metric,
        batch_size = batch_size,
        device = "cpu"
      )

      expect_identical(result$index, expected$index)
      expect_equal(result$distance, expected$distance, tolerance = 1e-12)
      provenance <- cuda_provenance(result)
      expect_identical(provenance$backend, c("base", "base"))
      expect_identical(provenance$device, c("cpu", "cpu"))
      expect_identical(provenance$output_device, c("cpu", "cpu"))
      expect_false(any(provenance$fallback))
    }
  }
})

test_that("base backend advertises the stable top-k operation", {
  factory <- cudaverse:::.base_backend_factory()

  expect_true("stable-topk" %in% factory$capabilities())
  expect_true(is.function(factory$algorithm_knn_select))
  expect_true("algorithm_knn_select" %in%
                cudaverse:::.backend_operations(factory))
})
