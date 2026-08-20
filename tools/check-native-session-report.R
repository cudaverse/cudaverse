if (!requireNamespace("jsonlite", quietly = TRUE)) {
  stop("Checking native session evidence requires `jsonlite`.")
}
sys.source(
  file.path("tools", "native-session-report-io.R"),
  envir = environment()
)

input <- Sys.getenv(
  "CUDAVERSE_SESSION_REPORT",
  unset = file.path(
    "inst", "reports", "native", "session-contract-rtx2000.json"
  )
)
report <- jsonlite::read_json(input, simplifyVector = FALSE)
failures <- validate_native_session_report(
  report, expected_version = "0.4.0.9000"
)

if (length(failures)) {
  stop(
    "Native session report failed:\n- ",
    paste(unique(failures), collapse = "\n- "),
    call. = FALSE
  )
}
message(
  "Native session report passed: ", input,
  " (pids ", paste(vapply(
    report$sessions,
    function(session) native_session_number(session$pid),
    numeric(1L)
  ), collapse = ", "), ")"
)
