if (!requireNamespace("jsonlite", quietly = TRUE) ||
    !requireNamespace("digest", quietly = TRUE)) {
  stop(
    "Checking benchmark summaries requires jsonlite and digest.",
    call. = FALSE
  )
}

input <- Sys.getenv("CUDAVERSE_BENCHMARK_REPORT", unset = "")
summary_path <- Sys.getenv("CUDAVERSE_BENCHMARK_SUMMARY", unset = "")
if (!nzchar(input) || !file.exists(input)) {
  stop("Set CUDAVERSE_BENCHMARK_REPORT to an existing report.", call. = FALSE)
}
if (!nzchar(summary_path) || !file.exists(summary_path)) {
  stop("Set CUDAVERSE_BENCHMARK_SUMMARY to an existing summary.",
       call. = FALSE)
}

sha256_before_read <- digest::digest(
  input, algo = "sha256", file = TRUE, serialize = FALSE
)
report <- jsonlite::read_json(input, simplifyVector = FALSE)
sha256_after_read <- digest::digest(
  input, algo = "sha256", file = TRUE, serialize = FALSE
)
if (!identical(sha256_before_read, sha256_after_read)) {
  stop("The benchmark report changed while it was being read.", call. = FALSE)
}
scalar <- function(x, default = NA) {
  value <- unlist(x, recursive = TRUE, use.names = FALSE)
  if (!length(value)) default else value[[1L]]
}
logical_value <- function(x) isTRUE(as.logical(scalar(x, FALSE)))
summary <- paste(readLines(summary_path, warn = FALSE), collapse = "\n")
failures <- character()
require_text <- function(pattern, message, fixed = TRUE) {
  if (!grepl(pattern, summary, fixed = fixed)) {
    failures <<- c(failures, message)
  }
}

if (!logical_value(report$complete)) {
  failures <- c(failures, "the benchmark report is incomplete")
}
if (grepl("DRAFT", summary, fixed = TRUE)) {
  failures <- c(failures, "the retained summary is marked as a draft")
}
commit <- as.character(scalar(report$source$commit, ""))
sha256 <- sha256_after_read
require_text(
  paste0("Source commit: `", commit, "`"),
  "summary does not identify the report source commit"
)
require_text(
  paste0("Report SHA-256: `", sha256, "`"),
  "summary does not identify the exact report SHA-256"
)
require_text(
  "Report complete: `true`",
  "summary does not identify a complete report"
)
require_text(
  "CUDA runtime bundled by cudaverse",
  "summary omits the bundled-runtime footprint"
)
require_text(
  "Stage sampling:",
  "summary omits the stage-sampling boundary"
)
require_text(
  "Memory sampling:",
  "summary omits the memory-sampling boundary"
)
require_text(
  "Ratios compare ten-run sample medians descriptively.",
  "summary overstates descriptive timing ratios"
)

backends <- unlist(
  report$contract$backends, recursive = TRUE, use.names = FALSE
)
for (case_id in names(report$cases)) {
  require_text(case_id, paste(case_id, "is absent from the summary"))
  present <- names(report$cases[[case_id]]$backends)
  for (backend in backends) {
    if (!backend %in% present) {
      failures <- c(
        failures,
        paste(case_id, "does not contain backend", backend)
      )
    }
    require_text(
      paste0("| ", case_id, " | ", backend, " |"),
      paste(case_id, backend, "table row is absent")
    )
  }
}

if (length(failures)) {
  stop(
    "Benchmark summary failed:\n- ",
    paste(unique(failures), collapse = "\n- "),
    call. = FALSE
  )
}
message("Benchmark summary matches the exact complete report: ", summary_path)
