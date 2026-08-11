if (!requireNamespace("jsonlite", quietly = TRUE) ||
    !requireNamespace("digest", quietly = TRUE)) {
  stop("Building native candidate reports requires jsonlite and digest.",
       call. = FALSE)
}
sys.source(
  file.path("tools", "native-candidate-report-io.R"),
  envir = environment()
)

input_path <- function(name) {
  path <- Sys.getenv(name, unset = "")
  if (!nzchar(path) || !file.exists(path)) {
    stop("Set `", name, "` to an existing report.", call. = FALSE)
  }
  path
}
consolidation_path <- input_path("CUDAVERSE_CONSOLIDATION_REPORT")
package_tests_path <- input_path("CUDAVERSE_NATIVE_PACKAGE_TEST_REPORT")
output <- Sys.getenv("CUDAVERSE_NATIVE_CANDIDATE_REPORT", unset = "")
if (!nzchar(output)) {
  stop("Set `CUDAVERSE_NATIVE_CANDIDATE_REPORT` to the output path.",
       call. = FALSE)
}

read_report <- function(path) {
  jsonlite::read_json(path, simplifyVector = FALSE)
}
sha256 <- function(path) {
  digest::digest(path, algo = "sha256", file = TRUE, serialize = FALSE)
}
report <- build_native_candidate_report(
  read_report(consolidation_path),
  read_report(package_tests_path),
  sha256(consolidation_path),
  sha256(package_tests_path)
)
failures <- validate_native_candidate_report(report)
if (length(failures)) {
  stop(
    "Native candidate report failed before write:\n- ",
    paste(failures, collapse = "\n- "), call. = FALSE
  )
}
dir.create(dirname(output), recursive = TRUE, showWarnings = FALSE)
jsonlite::write_json(
  report, output, auto_unbox = TRUE, pretty = TRUE, digits = 16,
  null = "null", na = "null"
)
message("Wrote passing native candidate report to ", output)
