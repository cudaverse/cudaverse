.embedding_has_compute_stages <- function(x) {
  if (is.list(x) && "compute_stages" %in% names(x)) {
    return(TRUE)
  }
  !is.null(attr(x, "compute_stages", exact = TRUE))
}

.embedding_source_provenance <- function(x) {
  if (!.embedding_has_compute_stages(x)) {
    return(NULL)
  }
  cuda_provenance(x)
}

.embedding_name <- function(x, argument) {
  if (!is.character(x) || length(x) != 1L || is.na(x) || !nzchar(x)) {
    stop(
      sprintf("`%s` must be one non-empty character string.", argument),
      call. = FALSE
    )
  }
  x
}

.embedding_sce_recorded_dim <- function(record) {
  if (!is.list(record)) {
    return(NULL)
  }
  recorded <- record[["reduced_dim"]]
  if (is.null(recorded) && is.list(record[["outputs"]])) {
    recorded <- record[["outputs"]][["reduced_dim"]]
  }
  recorded
}

.embedding_sce_record <- function(x, reduced_dim = NULL) {
  if (!requireNamespace("SingleCellExperiment", quietly = TRUE) ||
      !requireNamespace("S4Vectors", quietly = TRUE)) {
    stop(
      paste0(
        "Install the 'SingleCellExperiment' package with ",
        "`BiocManager::install(\"SingleCellExperiment\")` to use ",
        "SingleCellExperiment inputs."
      ),
      call. = FALSE
    )
  }
  record <- S4Vectors::metadata(x)[["cudacellr"]]
  if (!is.null(record) && !is.null(reduced_dim) &&
      !identical(.embedding_sce_recorded_dim(record), reduced_dim)) {
    return(NULL)
  }
  if (!is.null(record) && !is.list(record)) {
    stop(
      "`metadata(x)$cudacellr` must be a list when present.",
      call. = FALSE
    )
  }
  if (!is.null(record) &&
      !identical(record[["schema"]], "cudacellr-sce/1")) {
    stop(
      paste0(
        "`metadata(x)$cudacellr` does not follow the supported ",
        "`cudacellr-sce/1` schema."
      ),
      call. = FALSE
    )
  }
  if (!is.null(record)) {
    .embedding_name(
      .embedding_sce_recorded_dim(record),
      "metadata(x)$cudacellr$reduced_dim"
    )
  }
  record
}

.embedding_sce_reduced_dim <- function(x, reduced_dim, record) {
  available <- SingleCellExperiment::reducedDimNames(x)
  available <- as.character(available)
  choose_available <- function(value, source) {
    value <- .embedding_name(value, source)
    matches <- which(available == value)
    if (length(matches) == 1L) {
      return(value)
    }
    if (length(matches) > 1L) {
      stop(
        sprintf(
          "Reduced dimension %s is ambiguous because it occurs more than once.",
          sQuote(value)
        ),
        call. = FALSE
      )
    }
    choices <- if (length(available)) {
      paste(sQuote(available), collapse = ", ")
    } else {
      "<none>"
    }
    stop(
      sprintf(
        "Reduced dimension %s is not present in `x`; available values: %s.",
        sQuote(value),
        choices
      ),
      call. = FALSE
    )
  }

  if (!is.null(reduced_dim)) {
    return(choose_available(reduced_dim, "reduced_dim"))
  }

  recorded <- .embedding_sce_recorded_dim(record)
  if (!is.null(recorded)) {
    return(choose_available(
      recorded,
      "metadata(x)$cudacellr$reduced_dim"
    ))
  }

  pca_matches <- which(available == "PCA")
  if (length(pca_matches) == 1L) {
    return("PCA")
  }
  if (length(pca_matches) > 1L) {
    stop(
      "Reduced dimension \"PCA\" is ambiguous because it occurs more than once.",
      call. = FALSE
    )
  }
  if (!length(available)) {
    stop(
      paste0(
        "`x` has no reduced dimensions. Add one with `reducedDim()` or ",
        "supply a cudacellr result."
      ),
      call. = FALSE
    )
  }
  stop(
    paste0(
      "`reduced_dim` is required because cudacellr metadata and a standard ",
      "\"PCA\" reduced dimension are both absent. Available values: ",
      paste(sQuote(available), collapse = ", "),
      "."
    ),
    call. = FALSE
  )
}

.embedding_sce_provenance <- function(record) {
  if (is.null(record) || is.null(record[["compute_stages"]])) {
    return(NULL)
  }
  proxy <- list(
    provenance_schema = record[["provenance_schema"]],
    compute_device = record[["compute_device"]],
    compute_stages = record[["compute_stages"]]
  )
  cuda_provenance(proxy)
}

.embedding_sce_source_device <- function(record, provenance,
                                         source_compute_device) {
  candidate <- NULL
  if (!is.null(record)) {
    candidate <- record[["source_device"]]
    if (is.null(candidate)) {
      candidate <- record[["pca_device"]]
    }
  }
  if (!is.null(candidate)) {
    candidate <- .embedding_name(
      candidate,
      "metadata(x)$cudacellr$source_device"
    )
    if (!candidate %in% c("cpu", "cuda", "unknown")) {
      stop(
        paste0(
          "`metadata(x)$cudacellr$source_device` must be ",
          "\"cpu\", \"cuda\", or \"unknown\"."
        ),
        call. = FALSE
      )
    }
    return(candidate)
  }
  if (!is.null(provenance)) {
    pca_stage <- grep(
      "^pca_.*(preprocessing|decomposition)$",
      provenance$stage
    )
    if (length(pca_stage)) {
      return(provenance$device[[utils::tail(pca_stage, 1L)]])
    }
  }
  if (source_compute_device %in% c("cpu", "cuda")) {
    return(source_compute_device)
  }
  "unknown"
}

.embedding_sce_input <- function(x, reduced_dim) {
  record <- .embedding_sce_record(x, reduced_dim = reduced_dim)
  selected <- .embedding_sce_reduced_dim(x, reduced_dim, record)
  values <- as.matrix(SingleCellExperiment::reducedDim(x, selected))
  if (nrow(values) != ncol(x)) {
    stop(
      sprintf(
        "Reduced dimension %s must contain one row per cell.",
        sQuote(selected)
      ),
      call. = FALSE
    )
  }
  cell_names <- colnames(x)
  if (!is.null(cell_names)) {
    rownames(values) <- cell_names
  }

  source_provenance <- .embedding_sce_provenance(record)
  source_compute_device <- if (!is.null(source_provenance)) {
    attr(source_provenance, "compute_device", exact = TRUE)
  } else if (!is.null(record) && !is.null(record[["compute_device"]])) {
    compute_device <- .embedding_name(
      record[["compute_device"]],
      "metadata(x)$cudacellr$compute_device"
    )
    if (!compute_device %in% c("cpu", "cuda", "hybrid")) {
      stop(
        paste0(
          "`metadata(x)$cudacellr$compute_device` must be ",
          "\"cpu\", \"cuda\", or \"hybrid\"."
        ),
        call. = FALSE
      )
    }
    compute_device
  } else {
    "unknown"
  }
  source_device <- .embedding_sce_source_device(
    record,
    source_provenance,
    source_compute_device
  )

  list(
    matrix = unname(values),
    row_names = rownames(values),
    source_device = source_device,
    source_compute_device = source_compute_device,
    source_class = class(x)[[1L]],
    source_provenance = source_provenance,
    reduced_dim = selected
  )
}

.embedding_input <- function(x, reduced_dim = NULL) {
  if (inherits(x, "SingleCellExperiment")) {
    input <- .embedding_sce_input(x, reduced_dim)
    if (!is.matrix(input$matrix) || !is.numeric(input$matrix) ||
        nrow(input$matrix) < 3L || ncol(input$matrix) < 1L ||
        anyNA(input$matrix) || any(!is.finite(input$matrix))) {
      stop(
        paste0(
          "The selected reduced dimension must be a finite numeric matrix ",
          "with at least three rows."
        ),
        call. = FALSE
      )
    }
    return(input)
  }
  if (!is.null(reduced_dim)) {
    stop(
      "`reduced_dim` is only supported for SingleCellExperiment inputs.",
      call. = FALSE
    )
  }
  source_device <- "cpu"
  source_compute_device <- "cpu"
  source_class <- class(x)[[1L]]
  source_provenance <- .embedding_source_provenance(x)
  if (!is.null(source_provenance)) {
    source_compute_device <- attr(
      source_provenance,
      "compute_device",
      exact = TRUE
    )
  }
  if (inherits(x, "cudacell_workflow")) {
    source_device <- x$pca$device %||% "unknown"
    if (is.null(source_provenance)) {
      source_compute_device <- x$compute_device %||% source_device
    }
    x <- x$pca$x
  } else if (inherits(x, "cuda_pca")) {
    source_device <- x$device %||% "unknown"
    if (is.null(source_provenance)) {
      source_compute_device <- x$compute_device %||% source_device
    }
    x <- x$x
  } else if (inherits(x, "cudatensor")) {
    source_device <- x$device %||% "unknown"
    if (is.null(source_provenance)) {
      source_compute_device <- x$compute_device %||% source_device
    }
    x <- as.array(x)
  } else if (is.data.frame(x)) {
    x <- as.matrix(x)
  }
  if (!is.matrix(x) || !is.numeric(x) || nrow(x) < 3L || ncol(x) < 1L ||
      anyNA(x) || any(!is.finite(x))) {
    stop(
      "`x` must resolve to a finite numeric matrix with at least three rows.",
      call. = FALSE
    )
  }
  list(
    matrix = unname(x),
    row_names = rownames(x),
    source_device = source_device,
    source_compute_device = source_compute_device,
    source_class = source_class,
    source_provenance = source_provenance,
    reduced_dim = NULL
  )
}

.embedding_diffusion_distance_stages <- function(distances) {
  stages <- attr(
    cuda_provenance(distances),
    "compute_stages",
    exact = TRUE
  )
  output <- list()
  if (!is.null(stages$input_x_materialization)) {
    output$distance_input <- stages$input_x_materialization
  }
  output$distance <- stages$distance
  output
}

`%||%` <- function(x, y) {
  if (is.null(x) || length(x) == 0L) y else x
}

.embedding_components <- function(n_components, maximum) {
  if (!is.numeric(n_components) || length(n_components) != 1L ||
      is.na(n_components) || !is.finite(n_components) ||
      n_components < 1 || n_components > maximum ||
      n_components != as.integer(n_components)) {
    stop(
      sprintf("`n_components` must be a whole number between 1 and %s.",
              maximum),
      call. = FALSE
    )
  }
  as.integer(n_components)
}

.embedding_seed <- function(seed) {
  if (is.null(seed)) {
    return(NULL)
  }
  if (!is.numeric(seed) || length(seed) != 1L || is.na(seed) ||
      !is.finite(seed)) {
    stop("`seed` must be NULL or one finite whole number.", call. = FALSE)
  }
  integer_seed <- suppressWarnings(as.integer(seed))
  if (is.na(integer_seed) || seed != integer_seed) {
    stop("`seed` must be NULL or one finite whole number.", call. = FALSE)
  }
  integer_seed
}

.with_embedding_seed <- function(seed, code) {
  seed <- .embedding_seed(seed)
  if (is.null(seed)) {
    return(force(code))
  }

  had_seed <- exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  if (had_seed) {
    old_seed <- get(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  }
  on.exit({
    if (had_seed) {
      assign(".Random.seed", old_seed, envir = .GlobalEnv)
    } else if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
      rm(".Random.seed", envir = .GlobalEnv)
    }
  }, add = TRUE)

  set.seed(seed)
  force(code)
}

.new_embedding <- function(coordinates, method, backend, input, parameters,
                           compute_device = "cpu", compute_stages = NULL) {
  if (is.null(compute_stages)) {
    compute_stages <- list(
      embedding = cuda_stage(
        requested_device = "fixed-cpu",
        device = "cpu",
        backend = backend,
        selection_reason = "algorithm_cpu_only",
        fallback = FALSE,
        output_device = "cpu"
      )
    )
  }
  provenance <- cuda_provenance(compute_stages)
  actual_compute_device <- attr(
    provenance,
    "compute_device",
    exact = TRUE
  )
  if (!identical(compute_device, actual_compute_device)) {
    stop(
      "`compute_device` does not match the recorded compute stages.",
      call. = FALSE
    )
  }
  compute_stages <- attr(provenance, "compute_stages", exact = TRUE)
  if (!is.null(input$row_names)) {
    rownames(coordinates) <- input$row_names
  }
  colnames(coordinates) <- paste0(
    toupper(method),
    seq_len(ncol(coordinates))
  )
  structure(
    list(
      coordinates = coordinates,
      method = method,
      backend = backend,
      compute_device = compute_device,
      compute_stages = compute_stages,
      provenance_schema = attr(provenance, "schema", exact = TRUE),
      source_device = input$source_device,
      source_compute_device = input$source_compute_device,
      source_class = input$source_class,
      source_provenance = input$source_provenance,
      parameters = parameters
    ),
    class = "cuda_embedding"
  )
}

#' UMAP embedding
#'
#' UMAP currently uses the CPU `uwot` backend. GPU-aware cudaverse inputs are
#' accepted and their source device is retained in the result metadata.
#'
#' @param x Numeric observation-by-feature matrix, compatible cudaverse result,
#'   or a `SingleCellExperiment` with a reduced dimension.
#' @param n_components Output dimensions.
#' @param n_neighbors Number of nearest neighbours.
#' @param min_dist Minimum UMAP distance.
#' @param metric Distance metric passed to `uwot::umap()`.
#' @param n_epochs Optional training epochs.
#' @param seed Optional random seed.
#' @param ... Additional arguments passed to `uwot::umap()`.
#' @param reduced_dim For a `SingleCellExperiment`, the reduced-dimension name
#'   to embed. When `NULL`, a cudacellr metadata choice is used first, followed
#'   by a uniquely named `"PCA"`. Other names must be selected explicitly.
#' @return A `cuda_embedding` list containing `coordinates`, `method`,
#'   `backend`, `compute_device`, per-stage `compute_stages`, source metadata,
#'   and algorithm `parameters`.
#' @export
#' @examples
#' if (requireNamespace("uwot", quietly = TRUE)) {
#'   cuda_umap(matrix(rnorm(120), 40, 3), n_neighbors = 5, seed = 1)
#' }
cuda_umap <- function(x, n_components = 2L, n_neighbors = 15L,
                      min_dist = 0.1, metric = "euclidean",
                      n_epochs = NULL, seed = NULL, ...,
                      reduced_dim = NULL) {
  if (!requireNamespace("uwot", quietly = TRUE)) {
    stop("Install the 'uwot' package to compute UMAP embeddings.",
         call. = FALSE)
  }
  input <- .embedding_input(x, reduced_dim = reduced_dim)
  n_components <- .embedding_components(
    n_components,
    max(1L, nrow(input$matrix) - 2L)
  )
  if (!is.numeric(n_neighbors) || length(n_neighbors) != 1L ||
      is.na(n_neighbors) || n_neighbors < 2L ||
      n_neighbors >= nrow(input$matrix) ||
      n_neighbors != as.integer(n_neighbors)) {
    stop("`n_neighbors` must be between 2 and nrow(x) - 1.",
         call. = FALSE)
  }
  if (!is.numeric(min_dist) || length(min_dist) != 1L ||
      is.na(min_dist) || !is.finite(min_dist) || min_dist < 0) {
    stop("`min_dist` must be a finite non-negative number.",
         call. = FALSE)
  }
  if (!is.character(metric) || length(metric) != 1L || is.na(metric)) {
    stop("`metric` must be one character string.", call. = FALSE)
  }
  arguments <- list(
    X = input$matrix,
    n_neighbors = as.integer(n_neighbors),
    n_components = n_components,
    min_dist = min_dist,
    metric = metric,
    ret_model = FALSE,
    verbose = FALSE,
    ...
  )
  if (!is.null(n_epochs)) {
    arguments$n_epochs <- n_epochs
  }
  coordinates <- .with_embedding_seed(
    seed,
    do.call(uwot::umap, arguments)
  )
  .new_embedding(
    coordinates,
    method = "umap",
    backend = "uwot",
    input = input,
    parameters = c(
      list(
        n_components = n_components,
        n_neighbors = as.integer(n_neighbors),
        min_dist = min_dist,
        metric = metric,
        n_epochs = n_epochs,
        seed = .embedding_seed(seed)
      ),
      if (is.null(input$reduced_dim)) {
        list()
      } else {
        list(reduced_dim = input$reduced_dim)
      }
    )
  )
}

#' t-SNE embedding
#'
#' t-SNE currently uses the CPU `Rtsne` backend.
#'
#' @inheritParams cuda_umap
#' @param perplexity t-SNE perplexity.
#' @param theta Barnes-Hut accuracy/speed trade-off.
#' @param ... Additional arguments passed to `Rtsne::Rtsne()`.
#' @return A `cuda_embedding`; see [cuda_umap()] for the stable result fields.
#' @export
#' @examples
#' if (requireNamespace("Rtsne", quietly = TRUE)) {
#'   cuda_tsne(matrix(rnorm(120), 40, 3), perplexity = 5, seed = 1)
#' }
cuda_tsne <- function(x, n_components = 2L, perplexity = 30,
                      theta = 0.5, seed = NULL, ...,
                      reduced_dim = NULL) {
  input <- .embedding_input(x, reduced_dim = reduced_dim)
  n_components <- .embedding_components(
    n_components,
    min(3L, nrow(input$matrix) - 2L)
  )
  if (!is.numeric(perplexity) || length(perplexity) != 1L ||
      is.na(perplexity) || !is.finite(perplexity) || perplexity <= 0 ||
      3 * perplexity >= nrow(input$matrix) - 1L) {
    stop("`perplexity` must satisfy 3 * perplexity < nrow(x) - 1.",
         call. = FALSE)
  }
  if (!is.numeric(theta) || length(theta) != 1L || is.na(theta) ||
      !is.finite(theta) || theta < 0 || theta > 1) {
    stop("`theta` must be between 0 and 1.", call. = FALSE)
  }
  if (!requireNamespace("Rtsne", quietly = TRUE)) {
    stop("Install the 'Rtsne' package to compute t-SNE embeddings.",
         call. = FALSE)
  }
  fit <- .with_embedding_seed(
    seed,
    Rtsne::Rtsne(
      input$matrix,
      dims = n_components,
      perplexity = perplexity,
      theta = theta,
      pca = FALSE,
      check_duplicates = FALSE,
      verbose = FALSE,
      ...
    )
  )
  .new_embedding(
    fit$Y,
    method = "tsne",
    backend = "Rtsne",
    input = input,
    parameters = c(
      list(
        n_components = n_components,
        perplexity = perplexity,
        theta = theta,
        seed = .embedding_seed(seed)
      ),
      if (is.null(input$reduced_dim)) {
        list()
      } else {
        list(reduced_dim = input$reduced_dim)
      }
    )
  )
}

#' Diffusion-map-style embedding
#'
#' Pairwise distances can use the cudaverse CUDA path. Kernel construction
#' and eigendecomposition currently run on the CPU.
#'
#' @param x Numeric observation-by-feature matrix, compatible cudaverse result,
#'   or a `SingleCellExperiment` with a reduced dimension.
#' @param n_components Output dimensions.
#' @param sigma Gaussian kernel bandwidth. Defaults to the median positive
#'   pairwise distance.
#' @param diffusion_time Non-negative diffusion time exponent.
#' @param metric Euclidean or cosine distance.
#' @param device Device passed to [cuda_distance()].
#' @param reduced_dim For a `SingleCellExperiment`, the reduced-dimension name
#'   to embed. See [cuda_umap()] for automatic selection.
#' @return A `cuda_embedding` with the stable fields documented by
#'   [cuda_umap()], stage-level distance/kernel/eigendecomposition provenance,
#'   an optional `distance_input` stage when resident native storage is reused,
#'   and an additional `eigenvalues` element.
#' @export
#' @examples
#' cuda_diffusion_map(
#'   matrix(rnorm(120), 40, 3),
#'   n_components = 2,
#'   device = "cpu"
#' )
cuda_diffusion_map <- function(x, n_components = 2L, sigma = NULL,
                               diffusion_time = 1,
                               metric = c("euclidean", "cosine"),
                               device = c("auto", "cuda", "cpu"),
                               reduced_dim = NULL) {
  requested_device <- match.arg(device)
  input <- .embedding_input(x, reduced_dim = reduced_dim)
  n_components <- .embedding_components(
    n_components,
    nrow(input$matrix) - 2L
  )
  if (!is.numeric(diffusion_time) || length(diffusion_time) != 1L ||
      is.na(diffusion_time) || !is.finite(diffusion_time) ||
      diffusion_time < 0) {
    stop("`diffusion_time` must be finite and non-negative.",
         call. = FALSE)
  }
  metric <- match.arg(metric)
  distances <- cuda_distance(
    input$matrix,
    metric = metric,
    device = requested_device
  )
  distance_stages <- .embedding_diffusion_distance_stages(distances)
  positive <- distances[distances > 0 & is.finite(distances)]
  if (is.null(sigma)) {
    sigma <- if (length(positive)) stats::median(positive) else 1
  }
  if (!is.numeric(sigma) || length(sigma) != 1L || is.na(sigma) ||
      !is.finite(sigma) || sigma <= 0) {
    stop("`sigma` must be a positive finite number.", call. = FALSE)
  }
  kernel <- exp(-(distances^2) / (2 * sigma^2))
  diag(kernel) <- 0
  degree <- rowSums(kernel)
  if (any(degree <= 0)) {
    stop("The diffusion kernel contains isolated observations.",
         call. = FALSE)
  }
  inverse_root_degree <- 1 / sqrt(degree)
  normalized <- kernel * tcrossprod(inverse_root_degree)
  eigen_count <- n_components + 1L
  if (nrow(normalized) > 500L &&
      requireNamespace("RSpectra", quietly = TRUE)) {
    decomposition <- RSpectra::eigs_sym(
      normalized,
      k = eigen_count,
      which = "LA"
    )
    order_index <- order(decomposition$values, decreasing = TRUE)
    values <- decomposition$values[order_index]
    vectors <- decomposition$vectors[, order_index, drop = FALSE]
    backend <- "RSpectra"
  } else {
    decomposition <- eigen(normalized, symmetric = TRUE)
    values <- decomposition$values[seq_len(eigen_count)]
    vectors <- decomposition$vectors[, seq_len(eigen_count), drop = FALSE]
    backend <- "base-eigen"
  }
  retained_values <- pmax(values[-1L], 0)
  coordinates <- (
    vectors[, -1L, drop = FALSE] * inverse_root_degree
  ) * rep(retained_values^diffusion_time, each = nrow(vectors))
  distance_device <- attr(distances, "device") %||% "cpu"
  compute_device <- if (identical(distance_device, "cpu")) {
    "cpu"
  } else {
    "hybrid"
  }
  result <- .new_embedding(
    coordinates,
    method = "diffusion",
    backend = backend,
    input = input,
    parameters = c(
      list(
        n_components = n_components,
        sigma = sigma,
        diffusion_time = diffusion_time,
        metric = metric,
        requested_device = requested_device
      ),
      if (is.null(input$reduced_dim)) {
        list()
      } else {
        list(reduced_dim = input$reduced_dim)
      }
    ),
    compute_device = compute_device,
    compute_stages = c(
      distance_stages,
      list(kernel = cuda_stage(
        requested_device = "fixed-cpu",
        device = "cpu",
        backend = "base",
        selection_reason = "algorithm_cpu_only",
        output_device = "cpu"
      ),
      eigendecomposition = cuda_stage(
        requested_device = "fixed-cpu",
        device = "cpu",
        backend = backend,
        selection_reason = "algorithm_cpu_only",
        output_device = "cpu"
      ))
    )
  )
  result$eigenvalues <- retained_values
  result
}

#' Extract embedding coordinates
#'
#' @param x A `cuda_embedding`.
#' @return Numeric coordinate matrix.
#' @export
#' @examples
#' fit <- cuda_diffusion_map(
#'   matrix(rnorm(60), 20, 3),
#'   n_components = 2,
#'   device = "cpu"
#' )
#' embedding_coordinates(fit)
embedding_coordinates <- function(x) {
  if (!inherits(x, "cuda_embedding")) {
    stop("`x` must be a cuda_embedding object.", call. = FALSE)
  }
  x$coordinates
}

#' @export
print.cuda_embedding <- function(x, ...) {
  cat(sprintf(
    "<cuda_embedding method=%s observations=%s dimensions=%s backend=%s compute_device=%s>\n",
    x$method, nrow(x$coordinates), ncol(x$coordinates), x$backend,
    x$compute_device
  ))
  invisible(x)
}
