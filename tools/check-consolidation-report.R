if (!requireNamespace("jsonlite", quietly = TRUE)) {
  stop("Checking the consolidation report requires `jsonlite`.")
}

input <- Sys.getenv(
  "CUDAVERSE_CONSOLIDATION_REPORT",
  unset = file.path(
    "inst", "reports", "native", "consolidation-rtx2000.json"
  )
)
report <- jsonlite::read_json(input, simplifyVector = FALSE)

scalar <- function(x) unlist(x, recursive = TRUE, use.names = FALSE)[[1L]]
number <- function(x) as.numeric(scalar(x))
logical_value <- function(x) isTRUE(as.logical(scalar(x)))

failures <- character()
require_gate <- function(value, message) {
  if (!isTRUE(value)) failures <<- c(failures, message)
}

require_gate(
  identical(scalar(report$schema), "cudaverse-native-consolidation/1"),
  "unexpected consolidation schema"
)
require_gate(
  grepl("RTX 2000", paste(unlist(report$hardware$nvidia_smi), collapse = " "),
        fixed = TRUE),
  "report was not generated on the RTX 2000"
)
require_gate(
  identical(scalar(report$software$cudaverse), "0.2.0.9000"),
  "unexpected cudaverse version"
)
require_gate(
  nchar(scalar(report$source$cudaverse$commit)) == 40L &&
    !logical_value(report$source$cudaverse$tracked_dirty),
  "report did not record one clean cudaverse source commit"
)
require_gate(
  logical_value(report$software$native_diagnostics$available) &&
    logical_value(report$software$native_diagnostics$runtime_complete) &&
    logical_value(report$software$native_diagnostics$self_test$passed),
  "native diagnostics or cached self-test did not pass"
)

required_cases <- c("1000x50@0.1", "5000x100@0.03", "10000x128@0.01")
required_backends <- c("base", "native", "torch")
require_gate(
  identical(sort(names(report$benchmarks)), sort(required_cases)),
  "required benchmark cases are missing"
)
for (case_name in required_cases) {
  case <- report$benchmarks[[case_name]]
  require_gate(
    identical(sort(names(case)), sort(required_backends)),
    paste("required backends are missing for", case_name)
  )
  for (backend in required_backends) {
    result <- case[[backend]]
    require_gate(
      identical(scalar(result$status), "complete") &&
        logical_value(result$validation$passed),
      paste(case_name, backend, "benchmark/parity failed")
    )
  }
}

validation <- report$hardware_validation
for (gate in c(
  "auto_selection", "dtype_surface", "dense_lifecycle", "lifecycle",
  "shared_ownership", "structured_error", "injected_cuda_error",
  "interruption", "resident_pipeline", "stable_ties"
)) {
  require_gate(
    logical_value(validation[[gate]]$passed),
    paste("hardware gate failed:", gate)
  )
}
for (gate in c("dense_lifecycle", "lifecycle")) {
  require_gate(
    number(validation[[gate]]$cycles) == 1000L &&
      number(validation[[gate]]$whole_device_absolute_difference_bytes) <=
        1024^2 &&
      number(validation[[gate]]$tracked_current_difference_bytes) == 0,
    paste("lifecycle ceiling failed:", gate)
  )
}
require_gate(
  number(report$installed_size_bytes$cudaverse) <=
    number(report$benchmark_regression$installed_size$cudaverse$limit_bytes),
  "integrated installed size exceeded the combined-package baseline limit"
)
require_gate(
  logical_value(report$benchmark_regression$passed),
  "benchmark regression gate failed"
)
require_gate(logical_value(report$overall_pass), "overall report gate failed")

if (length(failures)) {
  stop(
    "Consolidation report failed:\n- ",
    paste(failures, collapse = "\n- "),
    call. = FALSE
  )
}

message("Consolidation report passed all machine-readable gates: ", input)
