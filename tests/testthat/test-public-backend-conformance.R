.conformance_matrix <- function() {
  path <- system.file(
    "reports", "backend-capability-matrix.csv", package = "cudaverse"
  )
  utils::read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
}

.conformance_specs <- function() {
  specs <- list(list(name = "base", device = "cpu", backend = "base"))
  require_cuda <- identical(
    tolower(Sys.getenv("CUDAVERSE_REQUIRE_CUDA", unset = "false")),
    "true"
  )
  native_tests <- identical(
    tolower(Sys.getenv("CUDAVERSE_NATIVE_TESTS", unset = "false")),
    "true"
  )
  diagnostics <- cuda_diagnostics()

  requested <- if (require_cuda) {
    c("torch", "native")
  } else if (native_tests) {
    "native"
  } else {
    character()
  }
  for (backend in requested) {
    details <- diagnostics$backend_diagnostics[[backend]]
    if (!is.list(details) || !isTRUE(details$available)) {
      stop(
        sprintf(
          "Required conformance backend `%s` is unavailable: %s",
          backend,
          if (is.list(details)) details$reason else "not registered"
        ),
        call. = FALSE
      )
    }
    if (identical(backend, "native") && !isTRUE(details$auto_eligible)) {
      stop("The required native conformance backend is not auto-eligible.",
           call. = FALSE)
    }
    specs[[length(specs) + 1L]] <- list(
      name = backend,
      device = "cuda",
      backend = backend
    )
  }
  specs
}

.with_conformance_backend <- function(spec, code) {
  old <- options(cudaverse.cuda_backends = spec$backend)
  on.exit(options(old), add = TRUE)
  force(code)
}

.conformance_tolerance <- function(spec) {
  if (identical(spec$device, "cpu")) 1e-10 else 1e-8
}

.expect_conformance_device <- function(x, spec) {
  expect_identical(
    tensor_device(x),
    c(device = spec$device, backend = spec$backend)
  )
}

test_that("every export belongs to one executable conformance case", {
  matrix <- .conformance_matrix()
  expect_setequal(matrix$api, getNamespaceExports("cudaverse"))
  expect_identical(anyDuplicated(matrix$api), 0L)
  expect_setequal(
    unique(matrix$contract_case),
    c("diagnostics", "tensor", "sparse", "algorithm", "graph", "embedding")
  )
  expect_true(all(table(matrix$contract_case) > 0L))
})

test_that("diagnostics and provenance retain their public contract", {
  diagnostics <- cuda_diagnostics()
  expect_s3_class(diagnostics, "cuda_diagnostics")
  expect_type(cuda_available(), "logical")
  expect_length(cuda_available(), 1L)

  selection <- cuda_select_device("cpu")
  expect_identical(selection$device, "cpu")
  expect_identical(selection$backend, "base")

  stage <- cuda_stage(
    requested_device = "cpu",
    device = "cpu",
    backend = "base",
    selection_reason = "explicit_cpu",
    output_device = "cpu"
  )
  provenance <- cuda_provenance(list(contract = stage))
  expect_s3_class(provenance, "cuda_provenance")
  expect_identical(provenance$stage, "contract")
  expect_identical(attr(provenance, "schema"), "cudaverse-stage/1")
})

test_that("automatic fallback is visible and explicit CUDA remains strict", {
  unavailable <- structure(
    list(
      torch_installed = FALSE,
      torch_version = NA_character_,
      cuda_available = FALSE,
      cuda_device_count = 0L,
      reason = "conformance_runtime_unavailable",
      detection_error = NULL
    ),
    class = "cuda_diagnostics"
  )
  testthat::local_mocked_bindings(
    cuda_diagnostics = function() unavailable
  )

  automatic <- cuda_select_device("auto")
  expect_identical(automatic$device, "cpu")
  expect_identical(automatic$backend, "base")
  expect_true(automatic$fallback)
  expect_identical(
    automatic$selection_reason,
    "conformance_runtime_unavailable"
  )

  condition <- tryCatch(cuda_select_device("cuda"), error = identity)
  expect_s3_class(condition, "cudaverse_cuda_unavailable")
  expect_identical(
    condition$diagnostics$reason,
    "conformance_runtime_unavailable"
  )
})

test_that("tensor exports conform across required backends", {
  values <- matrix(
    c(1.5, -2, 3, 4.5, 5, -6, 7, 8, 9, 10, -11, 12),
    nrow = 3L,
    dimnames = list(
      observation = paste0("sample_", 1:3),
      feature = paste0("feature_", 1:4)
    )
  )
  right <- matrix(
    seq_len(8) / 4,
    nrow = 4L,
    dimnames = list(
      feature = colnames(values),
      component = c("score_1", "score_2")
    )
  )

  for (spec in .conformance_specs()) {
    .with_conformance_backend(spec, {
      tolerance <- .conformance_tolerance(spec)
      tensor <- cuda_tensor(values, device = spec$device, dtype = "float64")
      expect_s3_class(tensor, "cudatensor")
      expect_identical(tensor_shape(tensor), c(3L, 4L))
      .expect_conformance_device(tensor, spec)
      expect_equal(to_cpu(tensor), values, tolerance = tolerance)
      expect_equal(as.array(tensor), values, tolerance = tolerance)
      expect_equal(as.matrix(tensor), values, tolerance = tolerance)

      transferred <- to_device(
        cuda_tensor(values, device = "cpu", dtype = "float64"),
        spec$device
      )
      .expect_conformance_device(transferred, spec)
      expect_equal(to_cpu(transferred), values, tolerance = tolerance)

      offset <- cuda_tensor(
        setNames(c(0.5, -1, 2, 3), colnames(values)),
        device = spec$device,
        dtype = "float64"
      )
      broadcast <- tensor_broadcast_to(offset, c(3L, 4L))
      expected_broadcast <- matrix(
        rep(c(0.5, -1, 2, 3), each = 3L),
        nrow = 3L,
        dimnames = list(NULL, colnames(values))
      )
      expect_equal(to_cpu(broadcast), expected_broadcast,
                   tolerance = tolerance)
      expect_equal(to_cpu(tensor + broadcast), values + expected_broadcast,
                   tolerance = tolerance)

      product <- tensor_matmul(
        tensor,
        cuda_tensor(right, device = spec$device, dtype = "float64")
      )
      expect_equal(to_cpu(product), values %*% right, tolerance = tolerance)
      summed <- to_cpu(tensor_sum(tensor, dim = 1L))
      averaged <- to_cpu(tensor_mean(tensor, dim = 2L))
      expect_equal(as.vector(summed), unname(colSums(values)),
                   tolerance = tolerance)
      expect_equal(as.vector(averaged), unname(rowMeans(values)),
                   tolerance = tolerance)
      expect_identical(dimnames(summed), list(feature = colnames(values)))
      expect_identical(
        dimnames(averaged), list(observation = rownames(values))
      )
      expect_equal(
        as.vector(to_cpu(tensor_reshape(tensor, c(2L, 6L)))),
        as.vector(values),
        tolerance = tolerance
      )
      expect_equal(to_cpu(t(tensor)), t(values), tolerance = tolerance)

      subset <- tensor[c(3L, 1L), c(4L, 2L), drop = FALSE]
      expect_equal(to_cpu(subset), values[c(3L, 1L), c(4L, 2L), drop = FALSE],
                   tolerance = tolerance)
      tensor[1L, 2L] <- 42
      values_replaced <- values
      values_replaced[1L, 2L] <- 42
      expect_equal(to_cpu(tensor), values_replaced, tolerance = tolerance)
    })
  }
})

test_that("sparse exports conform across required backends", {
  values <- matrix(
    c(1, 0, 3, 2, 4, 0, 5, 1, 2, 3, 0, 6),
    nrow = 4L,
    dimnames = list(
      observation = paste0("sample_", 1:4),
      feature = paste0("gene_", 1:3)
    )
  )
  dense_right <- matrix(
    seq_len(6) / 3,
    nrow = 3L,
    dimnames = list(
      feature = colnames(values),
      component = c("PC1", "PC2")
    )
  )

  for (spec in .conformance_specs()) {
    .with_conformance_backend(spec, {
      tolerance <- .conformance_tolerance(spec)
      sparse <- cuda_sparse(values, format = "csr", device = spec$device)
      expect_s3_class(sparse, "cudasparse")
      expect_identical(dim(sparse), c(4L, 3L))
      expect_identical(dimnames(sparse), dimnames(values))
      expect_identical(sparse_info(sparse)$format, "csr")
      expect_identical(sparse_info(as_coo(sparse))$format, "coo")
      expect_identical(sparse_info(as_csr(as_coo(sparse)))$format, "csr")
      expect_equal(as.matrix(to_dgCMatrix(sparse)), values,
                   tolerance = tolerance)
      expect_equal(as.matrix(to_dgCMatrix(t(sparse))), t(values),
                   tolerance = tolerance)
      row_sums <- sparse_row_sums(sparse)
      col_sums <- sparse_col_sums(sparse)
      expect_equal(as.numeric(row_sums), unname(rowSums(values)),
                   tolerance = tolerance)
      expect_equal(as.numeric(col_sums), unname(colSums(values)),
                   tolerance = tolerance)
      expect_identical(names(row_sums), rownames(values))
      expect_identical(names(col_sums), colnames(values))
      expect_equal(
        sparse_matvec(sparse, setNames(1:3, colnames(values))),
        as.vector(values %*% (1:3)),
        tolerance = tolerance,
        ignore_attr = TRUE
      )
      product <- sparse_matmul_dense(sparse, dense_right)
      expect_equal(to_cpu(product), values %*% dense_right,
                   tolerance = tolerance)

      normalized <- sparse_normalize(
        sparse,
        margin = "rows",
        scale_factor = 100,
        log1p = FALSE
      )
      expect_equal(
        as.matrix(to_dgCMatrix(normalized)),
        values * 100 / rowSums(values),
        tolerance = tolerance
      )
      expect_identical(normalized$device, spec$device)
    })
  }
})

test_that("algorithm exports conform across required backends", {
  values <- rbind(
    c(0.2, 1.1, 2.7), c(1.3, 0.4, 3.2),
    c(2.1, 1.8, 0.7), c(3.4, 2.2, 1.5),
    c(8.2, 9.1, 7.4), c(9.3, 8.4, 8.7),
    c(7.6, 9.8, 9.2), c(8.9, 7.5, 10.1)
  )
  rownames(values) <- paste0("sample_", seq_len(nrow(values)))
  colnames(values) <- paste0("feature_", seq_len(ncol(values)))
  centers <- values[c(1L, 5L), , drop = FALSE]
  reference_distance <- cuda_distance(values, device = "cpu")
  reference_knn <- cuda_knn(values, k = 3L, batch_size = 3L, device = "cpu")
  reference_kmeans <- cuda_kmeans(values, centers = centers, device = "cpu")

  for (spec in .conformance_specs()) {
    .with_conformance_backend(spec, {
      tolerance <- .conformance_tolerance(spec)
      decomposition <- cuda_svd(values, device = spec$device)
      reconstructed <- decomposition$u %*% diag(decomposition$d) %*%
        t(decomposition$v)
      expect_equal(unname(reconstructed), unname(values),
                   tolerance = tolerance)

      pca <- cuda_pca(values, n_components = 2L, device = spec$device)
      reference_rotation <- stats::prcomp(values)$rotation[, 1:2, drop = FALSE]
      expect_equal(
        tcrossprod(pca$rotation),
        tcrossprod(reference_rotation),
        tolerance = tolerance
      )
      predicted <- predict(pca, values[1:2, , drop = FALSE],
                           device = spec$device)
      expect_identical(dim(predicted), c(2L, 2L))

      distance <- cuda_distance(values, device = spec$device)
      expect_equal(distance, reference_distance, tolerance = tolerance,
                   ignore_attr = TRUE)
      knn <- cuda_knn(
        values, k = 3L, batch_size = 3L, device = spec$device
      )
      expect_identical(knn$index, reference_knn$index)
      expect_equal(knn$distance, reference_knn$distance,
                   tolerance = tolerance)

      kmeans <- cuda_kmeans(values, centers = centers, device = spec$device)
      expect_identical(kmeans$cluster, reference_kmeans$cluster)
      expect_equal(kmeans$centers, reference_kmeans$centers,
                   tolerance = tolerance)
      expect_equal(
        predict(kmeans, values[1:2, , drop = FALSE], device = spec$device),
        reference_kmeans$cluster[1:2],
        ignore_attr = TRUE
      )
    })
  }
})

test_that("invalid inputs fail consistently and leave each backend reusable", {
  zero_row <- rbind(c(0, 0), c(1, 1), c(2, 3))
  constant <- cbind(1:4, rep(1, 4), 4:1)
  negative <- matrix(c(1, -1, 2, 3), nrow = 2L)

  for (spec in .conformance_specs()) {
    .with_conformance_backend(spec, {
      expect_error(
        cuda_distance(zero_row, metric = "cosine", device = spec$device),
        "zero-length rows"
      )
      expect_error(
        cuda_pca(constant, n_components = 2L, scale. = TRUE,
                 device = spec$device),
        "constant features"
      )
      expect_error(
        sparse_normalize(cuda_sparse(negative, device = spec$device)),
        "non-negative"
      )
      expect_error(
        tensor_matmul(
          cuda_tensor(matrix(1:6, 2L), device = spec$device),
          cuda_tensor(matrix(1:8, 4L), device = spec$device)
        ),
        "not conformable"
      )

      probe <- tensor_matmul(
        cuda_tensor(diag(2), device = spec$device, dtype = "float64"),
        cuda_tensor(diag(2), device = spec$device, dtype = "float64")
      )
      expect_equal(to_cpu(probe), diag(2),
                   tolerance = .conformance_tolerance(spec))
    })
  }
})

test_that("graph exports retain their intentional CPU boundary", {
  skip_if_not_installed("igraph")
  values <- rbind(
    c(0, 0), c(0, 1), c(1, 0), c(1, 1),
    c(5, 5), c(5, 6), c(6, 5), c(6, 6)
  )

  for (spec in .conformance_specs()) {
    .with_conformance_backend(spec, {
      neighbors <- cuda_knn(values, k = 3L, device = spec$device)
      graph <- cuda_knn_graph(neighbors, weighting = "gaussian")
      expect_s3_class(graph, "cuda_graph")
      expect_identical(graph$source_device, spec$device)
      expect_s4_class(as_adjacency_matrix(graph), "dgCMatrix")
      expect_identical(cuda_provenance(graph)$device, "cpu")

      louvain <- cuda_louvain(graph)
      leiden <- cuda_leiden(graph, n_iterations = 2L)
      expect_length(louvain$membership, nrow(values))
      expect_length(leiden$membership, nrow(values))
      expect_identical(cuda_provenance(louvain)$device, "cpu")
      expect_identical(cuda_provenance(leiden)$device, "cpu")
    })
  }
})

test_that("embedding exports retain direct and intentional CPU stages", {
  values <- matrix(
    sin(seq_len(72) / 5),
    nrow = 24L,
    dimnames = list(paste0("sample_", 1:24), paste0("feature_", 1:3))
  )

  for (spec in .conformance_specs()) {
    .with_conformance_backend(spec, {
      embedding <- cuda_diffusion_map(
        values, n_components = 2L, device = spec$device
      )
      expect_s3_class(embedding, "cuda_embedding")
      expect_identical(
        embedding_coordinates(embedding),
        embedding$coordinates
      )
      provenance <- cuda_provenance(embedding)
      expect_identical(provenance$device[provenance$stage == "kernel"], "cpu")
      expect_identical(
        provenance$device[provenance$stage == "eigendecomposition"],
        "cpu"
      )
      expect_identical(
        provenance$device[provenance$stage == "distance"],
        spec$device
      )
    })
  }

  if (requireNamespace("Rtsne", quietly = TRUE)) {
    tsne <- cuda_tsne(
      values, n_components = 2L, perplexity = 3, seed = 1,
      max_iter = 250
    )
    expect_identical(tsne$backend, "Rtsne")
    expect_identical(cuda_provenance(tsne)$device, "cpu")
  }
  if (requireNamespace("uwot", quietly = TRUE)) {
    umap <- cuda_umap(
      values, n_components = 2L, n_neighbors = 5L, n_epochs = 10L, seed = 1
    )
    expect_identical(umap$backend, "uwot")
    expect_identical(cuda_provenance(umap)$device, "cpu")
  }
})
