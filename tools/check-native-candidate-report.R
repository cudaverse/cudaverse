if (!requireNamespace("jsonlite", quietly = TRUE)) {
  stop("Checking native candidate reports requires jsonlite.", call. = FALSE)
}
sys.source(
  file.path("tools", "native-candidate-report-io.R"),
  envir = environment()
)
input <- Sys.getenv("CUDAVERSE_NATIVE_CANDIDATE_REPORT", unset = "")
if (!nzchar(input) || !file.exists(input)) {
  stop("Set `CUDAVERSE_NATIVE_CANDIDATE_REPORT` to an existing report.",
       call. = FALSE)
}
report <- jsonlite::read_json(input, simplifyVector = FALSE)
failures <- validate_native_candidate_report(report)
if (length(failures)) {
  stop(
    "Native candidate report failed:\n- ",
    paste(failures, collapse = "\n- "), call. = FALSE
  )
}
message("Native candidate report passed all gates: ", input)
