benchmark_validate_idle_gpu_requirement <- function(profile, backends, required) {
  if (identical(profile, "full") && any(backends != "base") &&
      !isTRUE(required)) {
    stop("Full retained CUDA benchmarks require idle-GPU inspection; use ",
         "a smoke profile for non-retained experiments.", call. = FALSE)
  }
  invisible(TRUE)
}

benchmark_parse_compute_pids <- function(lines) {
  lines <- trimws(as.character(lines))
  if (anyNA(lines)) {
    stop("Could not parse GPU compute process inventory; refusing to assume ",
         "the GPU is idle.", call. = FALSE)
  }
  lines <- lines[nzchar(lines)]
  if (!length(lines) || identical(lines, "No running processes found")) {
    return(integer())
  }
  valid <- grepl("^[1-9][0-9]*$", lines)
  values <- suppressWarnings(as.numeric(lines))
  if (!all(valid) || any(!is.finite(values)) ||
      any(values > .Machine$integer.max)) {
    stop("Could not parse GPU compute process inventory; refusing to assume ",
         "the GPU is idle.", call. = FALSE)
  }
  sort(unique(as.integer(values)))
}

benchmark_gpu_compute_pids <- function(command = "nvidia-smi") {
  output <- tryCatch(
    system2(
      command,
      c(
        "--query-compute-apps=pid",
        "--format=csv,noheader,nounits"
      ),
      stdout = TRUE,
      stderr = TRUE
    ),
    error = identity
  )
  if (inherits(output, "error")) {
    stop(
      "Could not inspect active GPU compute processes: ",
      conditionMessage(output),
      call. = FALSE
    )
  }
  status <- attr(output, "status")
  if (!is.null(status) && status != 0L) {
    stop(
      "Could not inspect active GPU compute processes: ",
      paste(output, collapse = " "),
      call. = FALSE
    )
  }
  benchmark_parse_compute_pids(output)
}

# The compute-only helpers above remain useful for diagnostics, but cannot
# establish retained-performance eligibility: graphics clients are omitted.
benchmark_parse_gpu_inventory <- function(lines) {
  fail <- function(reason) {
    stop("Invalid full GPU process/display inventory: ", reason,
         "; refusing to assume the GPU is idle.", call. = FALSE)
  }
  if (!requireNamespace("xml2", quietly = TRUE)) {
    fail("xml2 is required to parse nvidia-smi XML")
  }
  if (!is.character(lines) || !length(lines) || anyNA(lines)) {
    fail("empty or missing XML")
  }
  document <- tryCatch(
    xml2::read_xml(paste(lines, collapse = "\n"), options = "NONET"),
    error = function(e) fail("malformed nvidia-smi XML")
  )
  root <- xml2::xml_root(document)
  if (!identical(xml2::xml_name(root), "nvidia_smi_log")) {
    fail("unexpected XML root")
  }
  field <- function(node, path) {
    values <- xml2::xml_find_all(node, path)
    if (length(values) != 1L || length(xml2::xml_children(values[[1L]]))) {
      fail(paste("missing or ambiguous", path))
    }
    value <- trimws(xml2::xml_text(values[[1L]]))
    if (!nzchar(value)) fail(paste("empty", path))
    value
  }
  positive_integer <- function(value, label) {
    number <- suppressWarnings(as.numeric(value))
    if (!grepl("^[1-9][0-9]*$", value) || !is.finite(number) ||
        number > .Machine$integer.max) fail(paste("invalid", label))
    as.integer(number)
  }
  count <- positive_integer(field(root, "./attached_gpus"), "GPU count")
  gpu_nodes <- xml2::xml_find_all(root, "./gpu")
  if (length(gpu_nodes) != count) fail("GPU count does not match inventory")
  gpus <- lapply(gpu_nodes, function(gpu) {
    uuid <- field(gpu, "./uuid")
    if (!grepl("^GPU-[[:xdigit:]]{8}-[[:xdigit:]]{4}-[[:xdigit:]]{4}-[[:xdigit:]]{4}-[[:xdigit:]]{12}$", uuid)) {
      fail("invalid GPU UUID")
    }
    display <- field(gpu, "./display_active")
    if (!display %in% c("Enabled", "Disabled")) {
      fail("display activity is unavailable or unrecognized")
    }
    processes <- xml2::xml_find_all(gpu, "./processes")
    if (length(processes) != 1L) fail("missing or ambiguous process inventory")
    # An empty element is a supported, empty process inventory. N/A, error
    # text, and unknown elements must never be interpreted as no processes.
    direct_text <- xml2::xml_text(xml2::xml_find_all(processes, "./text()"))
    if (any(nzchar(trimws(direct_text)))) fail("process inventory is unavailable")
    process_nodes <- xml2::xml_children(processes)
    if (any(xml2::xml_name(process_nodes) != "process_info")) {
      fail("unrecognized process inventory element")
    }
    entries <- lapply(process_nodes, function(process) {
      pid <- positive_integer(field(process, "./pid"), "process PID")
      type <- field(process, "./type")
      if (!type %in% c("C", "G", "C+G", "M", "M+C", "O")) {
        fail("unrecognized process type")
      }
      list(pid = pid, type = type)
    })
    list(uuid = uuid, display_active = display, processes = entries)
  })
  if (anyDuplicated(vapply(gpus, `[[`, character(1L), "uuid"))) {
    fail("duplicate GPU UUID")
  }
  list(schema = "cudaverse-gpu-idle-inventory/1", gpus = gpus)
}

benchmark_gpu_inventory <- function(command = "nvidia-smi") {
  output <- tryCatch(
    suppressWarnings(system2(command, c("-q", "-x"),
                             stdout = TRUE, stderr = TRUE)),
    error = function(e) stop("Could not inspect full GPU process/display inventory: ",
                             conditionMessage(e), call. = FALSE)
  )
  status <- attr(output, "status", exact = TRUE)
  if (!is.null(status) && status != 0L) {
    stop("Could not inspect full GPU process/display inventory: ",
         paste(output, collapse = " "), call. = FALSE)
  }
  benchmark_parse_gpu_inventory(output)
}

benchmark_assert_idle_gpu <- function(
  context,
  inventory = benchmark_gpu_inventory(),
  current_pid = Sys.getpid()
) {
  valid_pid <- function(value) {
    is.numeric(value) && length(value) == 1L && is.finite(value) &&
      value > 0 && value <= .Machine$integer.max && value == floor(value)
  }
  valid_gpu <- function(gpu) {
    is.list(gpu) && is.character(gpu$uuid) && length(gpu$uuid) == 1L &&
      !is.na(gpu$uuid) && nzchar(gpu$uuid) &&
      is.character(gpu$display_active) && length(gpu$display_active) == 1L &&
      gpu$display_active %in% c("Enabled", "Disabled") &&
      is.list(gpu$processes) && all(vapply(gpu$processes, function(process) {
        is.list(process) && valid_pid(process$pid) &&
          is.character(process$type) && length(process$type) == 1L &&
          process$type %in% c("C", "G", "C+G", "M", "M+C", "O")
      }, logical(1L)))
  }
  if (!valid_pid(current_pid) || !is.list(inventory) ||
      !identical(inventory$schema, "cudaverse-gpu-idle-inventory/1") ||
      !is.list(inventory$gpus) || !length(inventory$gpus) ||
      !all(vapply(inventory$gpus, valid_gpu, logical(1L)))) {
    stop("Invalid full GPU process/display inventory; refusing to assume ",
         "the GPU is idle.", call. = FALSE)
  }
  for (gpu in inventory$gpus) {
    if (!identical(gpu$display_active, "Disabled")) {
      stop("Retained benchmark requires an idle GPU; active display context ",
           "on ", gpu$uuid, " during ", context, ".", call. = FALSE)
    }
    competing <- Filter(function(process) {
      process$pid != current_pid || !identical(process$type, "C")
    }, gpu$processes)
    if (length(competing)) {
      labels <- vapply(competing, function(process) {
        paste0(process$pid, " (", process$type, ")")
      }, character(1L))
      stop("Retained benchmark requires an idle GPU; competing or non-compute ",
           "GPU process detected on ", gpu$uuid, " during ", context, ": ",
           paste(labels, collapse = ", "),
           ". Preserve the checkpoint and resume when the GPU is idle.",
           call. = FALSE)
    }
  }
  invisible(TRUE)
}

# Retain the observation before rethrowing a rejected or unavailable inventory.
# Callers decide when to checkpoint; queries and recording stay outside timers.
benchmark_gpu_guard_recorder <- function(
  read_inventory = benchmark_gpu_inventory,
  clock = function() format(Sys.time(), tz = "UTC", usetz = TRUE),
  current_pid = Sys.getpid
) {
  observations <- list()
  check <- function(context) {
    observation <- list(context = context, observed_at_utc = clock(),
                        pid = current_pid(), passed = FALSE)
    failure <- tryCatch({
      observation$inventory <- read_inventory()
      benchmark_assert_idle_gpu(context, inventory = observation$inventory,
                                 current_pid = observation$pid)
      NULL
    }, error = identity, interrupt = identity)
    observation$passed <- is.null(failure)
    if (!is.null(failure)) observation$error <- conditionMessage(failure)
    observations[[length(observations) + 1L]] <<- observation
    if (!is.null(failure)) stop(failure)
    invisible(TRUE)
  }
  prepend <- function(previous) {
    if (is.null(previous)) return(invisible(NULL))
    if (!is.list(previous)) {
      stop("Existing GPU guard observations must be a list.", call. = FALSE)
    }
    observations <<- c(previous, observations)
    invisible(NULL)
  }
  list(check = check, snapshot = function() observations, prepend = prepend)
}
