if (!requireNamespace("jsonlite", quietly = TRUE)) {
  stop("Testing consolidation reports requires `jsonlite`.", call. = FALSE)
}

fixture <- jsonlite::read_json(
  file.path("inst", "reports", "native", "consolidation-rtx2000.json"),
  simplifyVector = FALSE
)
fixture$software$cudaverse <- "0.3.0.9000"
fixture$benchmark_regression$role <- "advisory"
fixture$benchmark_regression$release_gate <- FALSE
fixture$benchmark_regression$authority <- "cudaverse-benchmark/1 full profile"
fixture$benchmark_regression$installed_size$cudaverse$baseline_definition <-
  "legacy Phase 3 installed footprint baseline"
fixture$benchmark_regression$passed <- FALSE
fixture$benchmark_regression$cases[[1L]]$passed <- FALSE
fixture$overall_pass <- TRUE

path <- tempfile("cudaverse-consolidation-", fileext = ".json")
on.exit(unlink(path, force = TRUE), add = TRUE)

run_checker <- function(value) {
  jsonlite::write_json(
    value, path, auto_unbox = TRUE, pretty = TRUE, digits = 16,
    null = "null", na = "null"
  )
  old <- Sys.getenv("CUDAVERSE_CONSOLIDATION_REPORT", unset = NA_character_)
  on.exit({
    Sys.unsetenv("CUDAVERSE_CONSOLIDATION_REPORT")
    if (!is.na(old)) {
      Sys.setenv(CUDAVERSE_CONSOLIDATION_REPORT = old)
    }
  }, add = TRUE)
  Sys.setenv(CUDAVERSE_CONSOLIDATION_REPORT = path)
  tryCatch({
    sys.source(
      file.path("tools", "check-consolidation-report.R"),
      envir = new.env(parent = globalenv())
    )
    NULL
  }, error = identity)
}

stopifnot(is.null(run_checker(fixture)))

bad <- fixture
bad$benchmark_regression$release_gate <- TRUE
error <- run_checker(bad)
stopifnot(
  inherits(error, "error"),
  grepl(
    "historical regression data was presented as a release gate",
    conditionMessage(error), fixed = TRUE
  )
)

message("Consolidation advisory/release-gate tests passed.")
