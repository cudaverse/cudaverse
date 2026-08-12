arguments <- commandArgs(trailingOnly = TRUE)
site <- if (length(arguments)) arguments[[1L]] else "docs"
if (!dir.exists(site)) {
  stop("The rendered pkgdown directory does not exist: ", site,
       call. = FALSE)
}

failures <- character()
require_gate <- function(value, message) {
  if (!isTRUE(value)) failures <<- c(failures, message)
}

required <- c(
  "index.html",
  "CUDAVERSE-0.4-ROADMAP.html",
  file.path("articles", "gpu-setup.html"),
  file.path("articles", "backend-provenance.html"),
  file.path("articles", "backend-support.html"),
  file.path("reference", "index.html"),
  file.path("reference", "cuda_diagnostics.html"),
  file.path("news", "index.html"),
  "sitemap.xml"
)
for (path in required) {
  require_gate(file.exists(file.path(site, path)),
               paste("required public page is missing:", path))
}

internal_pages <- c(
  "CRAN-RELEASE.html",
  "CUDAVERSE-0.2-CAPABILITY-MATRIX.html",
  "CUDAVERSE-0.3-CHECKPOINTS.html",
  "CUDAVERSE-0.3-CANDIDATE-AUDIT.html",
  "CUDAVERSE-0.3-SPARSE-FLOAT32-DECISION.html",
  "CUDAVERSE-0.4-CHECKPOINTS.html",
  "NATIVE-CUDA-CONSOLIDATION.html",
  "NATIVE-CUDA-PHASE4-RC.html",
  "NATIVE-CUDA-ROADMAP.html"
)
for (path in internal_pages) {
  require_gate(!file.exists(file.path(site, path)),
               paste("maintainer evidence became a public page:", path))
}

published <- list.files(site, recursive = TRUE, full.names = TRUE)
published <- published[grepl("\\.(html|xml|json|txt|md)$", published,
                             ignore.case = TRUE)]
content <- paste(
  vapply(
    published,
    function(path) paste(readLines(path, warn = FALSE), collapse = "\n"),
    character(1L)
  ),
  collapse = "\n"
)
forbidden <- c(
  "cudatensr", "cudasparsr", "cudalearnr", "cudagraphR", "cudaembedr",
  "cudaverseCUDA", "former packages", "combined packages"
)
for (term in forbidden) {
  require_gate(
    !grepl(tolower(term), tolower(content), fixed = TRUE),
    paste("public site contains internal package-history term:", term)
  )
}

index_path <- file.path(site, "index.html")
index <- if (file.exists(index_path)) {
  paste(readLines(index_path, warn = FALSE), collapse = "\n")
} else {
  ""
}
required_home <- c(
  "cudaverse/cudaverse@develop/0.4",
  "https://github.com/cudaverse/cudaverse/blob/develop/0.4/inst/benchmarks/README.md",
  "https://cudaverse.github.io/cudaverse/articles/gpu-setup.html",
  "https://cudaverse.github.io/cudaverse/articles/backend-support.html"
)
for (target in required_home) {
  require_gate(grepl(target, index, fixed = TRUE),
               paste("homepage stable target is missing:", target))
}
require_gate(
  !grepl('href="[^"]*vignettes/[^"]*\\.Rmd"', index),
  "homepage contains a source Rmd link that pkgdown cannot serve"
)
require_gate(
  !grepl('href="[^"]*inst/benchmarks/README\\.html"', index),
  "homepage links to an unpublished benchmark HTML page"
)

if (length(failures)) {
  stop(
    "Public pkgdown boundary failed:\n- ",
    paste(unique(failures), collapse = "\n- "),
    call. = FALSE
  )
}
message("Public pkgdown pages and documentation boundary passed: ", site)
