native_session_scalar <- function(x, default = NA) {
  value <- unlist(x, recursive = TRUE, use.names = FALSE)
  if (!length(value) || is.na(value[[1L]])) default else value[[1L]]
}

native_session_text <- function(x) {
  as.character(native_session_scalar(x, ""))
}

native_session_number <- function(x) {
  suppressWarnings(as.numeric(native_session_scalar(x, NA_real_)))
}

native_session_logical <- function(x) {
  isTRUE(as.logical(native_session_scalar(x, FALSE)))
}

native_session_required_workflow <- function() {
  c(
    "native-self-test", "dense-shared-view", "sparse-normalize",
    "resident-pca-knn", "memory-telemetry", "injected-oom",
    "post-error-matmul", "allocator-cleanup", "clean-exit"
  )
}

validate_native_session_report <- function(
  report, expected_commit = NULL, expected_version = NULL,
  expected_hardware = NULL
) {
  failures <- character()
  require_gate <- function(value, message) {
    if (!isTRUE(value)) failures <<- c(failures, message)
  }

  source <- report$source
  commit <- native_session_text(source$commit)
  version <- native_session_text(source$version)
  hardware <- native_session_text(report$hardware$nvidia_smi)
  require_gate(
    identical(native_session_text(report$schema),
              "cudaverse-native-session/1"),
    "unexpected report schema"
  )
  require_gate(grepl("RTX 2000", hardware, fixed = TRUE),
               "report was not generated on the RTX 2000")
  require_gate(grepl("^[0-9a-f]{40}$", commit),
               "source commit is missing")
  require_gate(!native_session_logical(source$tracked_dirty),
               "tracked source was dirty before the session contract")
  if (!is.null(expected_commit)) {
    require_gate(identical(commit, expected_commit),
                 "source commit does not match the candidate")
  }
  if (!is.null(expected_version)) {
    require_gate(identical(version, expected_version),
                 "source version does not match the candidate")
  }
  if (!is.null(expected_hardware)) {
    require_gate(identical(hardware, expected_hardware),
                 "session and package-test reports identify different hardware")
  }
  require_gate(native_session_logical(report$passed),
               "session contract did not pass")

  contract <- report$contract
  require_gate(
    native_session_logical(contract$isolated_install) &&
      identical(native_session_number(contract$vanilla_sessions), 2) &&
      native_session_logical(contract$distinct_processes) &&
      native_session_logical(contract$native_backend_required) &&
      native_session_logical(contract$exact_allocator_cleanup) &&
      native_session_logical(contract$exited_process_check),
    "session isolation contract is incomplete"
  )
  require_gate(
    identical(native_session_text(contract$injected_error),
              "CUDA_ERROR_OUT_OF_MEMORY"),
    "injected-error contract changed"
  )

  sessions <- report$sessions
  require_gate(length(sessions) == 2L, "exactly two sessions are required")
  pids <- integer()
  if (length(sessions) == 2L) {
    for (index in seq_along(sessions)) {
      session <- sessions[[index]]
      pid <- native_session_number(session$pid)
      pids <- c(pids, as.integer(pid))
      require_gate(
        identical(native_session_number(session$session),
                  as.numeric(index)),
        paste("session", index, "has an invalid index")
      )
      require_gate(is.finite(pid) && pid > 0,
                   paste("session", index, "has an invalid pid"))
      require_gate(
        identical(native_session_text(session$commit), commit),
        paste("session", index, "commit does not match the source")
      )
      require_gate(
        identical(native_session_text(session$version), version),
        paste("session", index, "version does not match the source")
      )
      require_gate(
        identical(native_session_number(session$baseline_bytes), 0) &&
          identical(native_session_number(session$final_bytes), 0),
        paste("session", index, "did not return to an exact zero baseline")
      )
      require_gate(
        native_session_logical(session$process_absent_after_exit),
        paste("session", index, "remained in the NVIDIA process table")
      )
      workflow <- as.character(unlist(
        session$workflow, recursive = TRUE, use.names = FALSE
      ))
      require_gate(
        identical(workflow, native_session_required_workflow()),
        paste("session", index, "workflow evidence is incomplete")
      )
    }
    require_gate(!anyDuplicated(pids),
                 "session process ids are not distinct")
  }

  unique(failures)
}
