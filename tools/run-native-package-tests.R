if (!requireNamespace("jsonlite", quietly = TRUE) ||
    !requireNamespace("testthat", quietly = TRUE)) {
  stop(
    "Native package reporting requires jsonlite and testthat.",
    call. = FALSE
  )
}

enabled <- tolower(Sys.getenv("CUDAVERSE_NATIVE_TESTS", unset = ""))
if (!enabled %in% c("true", "1", "yes")) {
  stop(
    "Set `CUDAVERSE_NATIVE_TESTS=true` for the explicit RTX package gate.",
    call. = FALSE
  )
}
output <- Sys.getenv("CUDAVERSE_NATIVE_PACKAGE_TEST_REPORT", unset = "")
if (!nzchar(output)) {
  stop(
    "Set `CUDAVERSE_NATIVE_PACKAGE_TEST_REPORT` to an output path.",
    call. = FALSE
  )
}

command_lines <- function(command, args) {
  tryCatch(
    system2(command, args, stdout = TRUE, stderr = TRUE),
    error = function(error) conditionMessage(error)
  )
}
source_state <- function(path = ".") {
  commit <- command_lines("git", c("-C", path, "rev-parse", "HEAD"))
  changes <- command_lines(
    "git", c("-C", path, "status", "--porcelain", "--untracked-files=all")
  )
  list(
    commit = if (length(commit)) unname(commit[[1L]]) else "unavailable",
    tracked_dirty = length(changes) > 0L,
    status = unname(changes)
  )
}

source <- source_state(".")
test_error <- NULL
results <- tryCatch(
  testthat::test_local(
    ".", reporter = "summary", stop_on_failure = FALSE,
    stop_on_warning = FALSE, load_package = "source"
  ),
  error = function(error) {
    test_error <<- conditionMessage(error)
    NULL
  }
)
diagnostics <- if (is.null(results)) {
  simpleError("Package source did not load.")
} else {
  tryCatch(cudaverse::cuda_diagnostics(), error = identity)
}
if (inherits(diagnostics, "error")) {
  native <- list(
    available = FALSE,
    detection_error = conditionMessage(diagnostics)
  )
  native_ready <- FALSE
} else {
  native <- diagnostics$backend_diagnostics$native
  native_ready <-
    is.list(native) && isTRUE(native$available) &&
    isTRUE(native$runtime_complete) && isTRUE(native$self_test$passed) &&
    identical(diagnostics$selected_backend, "native") &&
    "native" %in% unlist(
      diagnostics$auto_eligible_backends,
      recursive = TRUE, use.names = FALSE
    )
}
expectations <- if (is.null(results)) {
  list()
} else {
  unlist(lapply(results, function(value) value$results), recursive = FALSE)
}
count_class <- function(class_name) {
  sum(vapply(expectations, inherits, logical(1L), class_name))
}
counts <- list(
  tests = length(results),
  expectations = length(expectations),
  failures = count_class("expectation_failure"),
  errors = count_class("expectation_error"),
  skips = count_class("expectation_skip"),
  warnings = count_class("expectation_warning")
)
passed <- is.null(test_error) && counts$tests > 0L &&
  counts$expectations > 0L && counts$failures == 0L &&
  counts$errors == 0L && counts$skips == 0L

report <- list(
  schema = "cudaverse-native-package-tests/1",
  generated_at_utc = format(Sys.time(), tz = "UTC", usetz = TRUE),
  source = list(cudaverse = source),
  hardware = list(nvidia_smi = command_lines(
    "nvidia-smi",
    c(
      "--query-gpu=name,driver_version,memory.total,compute_cap",
      "--format=csv,noheader,nounits"
    )
  )),
  software = list(
    R = R.version.string,
    cudaverse = unname(read.dcf("DESCRIPTION", fields = "Version")[[1L]]),
    native_diagnostics = native
  ),
  testthat = c(counts, list(
    native_ready = native_ready,
    runner_error = test_error,
    passed = passed && native_ready && !source$tracked_dirty
  )),
  overall_pass = passed && native_ready && !source$tracked_dirty
)
dir.create(dirname(output), recursive = TRUE, showWarnings = FALSE)
jsonlite::write_json(
  report, output, auto_unbox = TRUE, pretty = TRUE, digits = 16,
  null = "null", na = "null"
)
if (!isTRUE(report$overall_pass)) {
  stop(
    "Native RTX package gate failed; retained report: ",
    normalizePath(output, winslash = "/", mustWork = TRUE),
    call. = FALSE
  )
}
message(
  "Native RTX package gate passed without skips: ",
  normalizePath(output, winslash = "/", mustWork = TRUE)
)
