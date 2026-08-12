if (!requireNamespace("jsonlite", quietly = TRUE)) {
  stop("Checking native session evidence requires `jsonlite`.")
}

input <- Sys.getenv(
  "CUDAVERSE_SESSION_REPORT",
  unset = file.path(
    "inst", "reports", "native", "session-contract-rtx2000.json"
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
  identical(scalar(report$schema), "cudaverse-native-session/1"),
  "unexpected report schema"
)
require_gate(
  grepl("RTX 2000", scalar(report$hardware$nvidia_smi), fixed = TRUE),
  "report was not generated on the RTX 2000"
)
require_gate(
  grepl("^[0-9a-f]{40}$", scalar(report$source$commit)),
  "source commit is missing"
)
require_gate(
  !logical_value(report$source$tracked_dirty),
  "tracked source was dirty before the session contract"
)
require_gate(
  identical(scalar(report$source$version), "0.4.0.9000"),
  "unexpected source version"
)
require_gate(logical_value(report$passed), "session contract did not pass")
require_gate(
  logical_value(report$contract$isolated_install) &&
    number(report$contract$vanilla_sessions) == 2L &&
    logical_value(report$contract$distinct_processes) &&
    logical_value(report$contract$native_backend_required) &&
    logical_value(report$contract$exact_allocator_cleanup) &&
    logical_value(report$contract$exited_process_check),
  "session isolation contract is incomplete"
)
require_gate(
  identical(
    scalar(report$contract$injected_error),
    "CUDA_ERROR_OUT_OF_MEMORY"
  ),
  "injected-error contract changed"
)
require_gate(length(report$sessions) == 2L, "exactly two sessions are required")

pids <- integer()
required_workflow <- c(
  "native-self-test", "dense-shared-view", "sparse-normalize",
  "resident-pca-knn", "memory-telemetry", "injected-oom",
  "post-error-matmul", "allocator-cleanup", "clean-exit"
)
for (index in seq_along(report$sessions)) {
  session <- report$sessions[[index]]
  require_gate(
    number(session$session) == index,
    paste("session", index, "has an invalid index")
  )
  pid <- number(session$pid)
  pids <- c(pids, as.integer(pid))
  require_gate(pid > 0, paste("session", index, "has an invalid pid"))
  require_gate(
    identical(scalar(session$commit), scalar(report$source$commit)),
    paste("session", index, "commit does not match the source")
  )
  require_gate(
    number(session$baseline_bytes) == 0 && number(session$final_bytes) == 0,
    paste("session", index, "did not return to an exact zero baseline")
  )
  require_gate(
    logical_value(session$process_absent_after_exit),
    paste("session", index, "remained in the NVIDIA process table")
  )
  workflow <- as.character(unlist(
    session$workflow, recursive = TRUE, use.names = FALSE
  ))
  require_gate(
    identical(workflow, required_workflow),
    paste("session", index, "workflow evidence is incomplete")
  )
}
require_gate(!anyDuplicated(pids), "session process ids are not distinct")

if (length(failures)) {
  stop(
    "Native session report failed:\n- ",
    paste(unique(failures), collapse = "\n- "),
    call. = FALSE
  )
}
message(
  "Native session report passed: ", input,
  " (pids ", paste(pids, collapse = ", "), ")"
)
