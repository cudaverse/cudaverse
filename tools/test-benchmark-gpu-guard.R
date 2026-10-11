sys.source(
  file.path("tools", "benchmark-gpu-guard.R"),
  envir = environment()
)
stopifnot(isTRUE(benchmark_validate_idle_gpu_requirement(
  "full", c("base", "native"), TRUE
)))
stopifnot(isTRUE(benchmark_validate_idle_gpu_requirement(
  "smoke", c("base", "native"), FALSE
)))
error <- tryCatch(benchmark_validate_idle_gpu_requirement(
  "full", c("base", "native"), FALSE
), error = identity)
stopifnot(inherits(error, "error"),
          grepl("require idle-GPU inspection", conditionMessage(error),
                fixed = TRUE))

stopifnot(identical(
  benchmark_parse_compute_pids(c(" 42 ", "17", "42")),
  c(17L, 42L)
))
stopifnot(identical(
  benchmark_parse_compute_pids(c("No running processes found", "")),
  integer()
))
stopifnot(identical(benchmark_parse_compute_pids(character()), integer()))
for (invalid in list("N/A", "permission denied", "42 extra", "42.5", "0",
                     "-1", "2147483648", NA_character_,
                     c("42", "unrecognized"),
                     c("42", "No running processes found"))) {
  error <- tryCatch(benchmark_parse_compute_pids(invalid), error = identity)
  stopifnot(inherits(error, "error"),
            grepl("refusing to assume", conditionMessage(error), fixed = TRUE))
}
gpu_uuid <- "GPU-20f1bc2f-fe5d-b05c-a970-3f8ed181777a"
gpu_xml <- function(processes = "", display = "Disabled", uuid = gpu_uuid) {
  paste0('<gpu id="00000000:04:00.0"><uuid>', uuid, '</uuid>',
         '<display_active>', display, '</display_active>',
         '<processes>', processes, '</processes></gpu>')
}
inventory_xml <- function(gpus = gpu_xml(), count = 1L) {
  paste0('<?xml version="1.0"?><!DOCTYPE nvidia_smi_log SYSTEM "nvsmi_device_v12.dtd">',
         '<nvidia_smi_log><attached_gpus>', count, '</attached_gpus>',
         gpus, '</nvidia_smi_log>')
}
process_xml <- function(pid = "42", type = "C") {
  paste0('<process_info><gpu_instance_id>N/A</gpu_instance_id>',
         '<pid>', pid, '</pid><type>', type, '</type>',
         '<process_name>R</process_name><used_memory>N/A</used_memory>',
         '</process_info>')
}
assert_rejected <- function(run, pattern = NULL) {
  error <- tryCatch(run(), error = identity)
  stopifnot(inherits(error, "error"))
  if (!is.null(pattern)) stopifnot(grepl(pattern, conditionMessage(error), fixed = TRUE))
}
parse_inventory <- benchmark_parse_gpu_inventory
empty <- parse_inventory(inventory_xml())
self <- parse_inventory(inventory_xml(gpu_xml(process_xml())))
stopifnot(
  identical(empty$gpus[[1L]]$uuid, gpu_uuid),
  length(empty$gpus[[1L]]$processes) == 0L,
  identical(self$gpus[[1L]]$processes[[1L]], list(pid = 42L, type = "C")),
  isTRUE(benchmark_assert_idle_gpu("self-test", empty, current_pid = 42L)),
  isTRUE(benchmark_assert_idle_gpu("self-test", self, current_pid = 42L))
)
# Complete snapshots from display servers, WDDM C+G clients, MPS, and other
# contexts fail even though a compute-only PID query could be empty.
for (type in c("C", "G", "C+G", "M", "M+C", "O")) {
  inventory <- parse_inventory(inventory_xml(gpu_xml(process_xml("99", type))))
  assert_rejected(function() benchmark_assert_idle_gpu(
    "self-test", inventory, current_pid = 42L), "99")
}
for (type in c("G", "C+G", "M", "M+C", "O")) {
  inventory <- parse_inventory(inventory_xml(gpu_xml(process_xml("42", type))))
  assert_rejected(function() benchmark_assert_idle_gpu(
    "self-test", inventory, current_pid = 42L), "non-compute")
}
display <- parse_inventory(inventory_xml(gpu_xml(display = "Enabled")))
assert_rejected(function() benchmark_assert_idle_gpu(
  "self-test", display, current_pid = 42L), "active display context")
for (invalid in list(NA_integer_, 42.5, -1L, "42", Inf)) {
  assert_rejected(function() benchmark_assert_idle_gpu(
    "self-test", empty, current_pid = invalid), "refusing to assume")
  assert_rejected(function() benchmark_assert_idle_gpu(
    "self-test", invalid, current_pid = 42L), "refusing to assume")
}
for (invalid_xml in list(
  character(), NA_character_, "N/A", "<nvidia_smi_log>",
  inventory_xml(count = 0L), inventory_xml(count = 2L),
  inventory_xml(gpu_xml(display = "N/A")),
  inventory_xml(gpu_xml(display = "")),
  inventory_xml(gpu_xml(processes = "N/A")),
  inventory_xml(gpu_xml(processes = "<unknown/>")),
  inventory_xml(gpu_xml(process_xml(type = "Unknown"))),
  inventory_xml(gpu_xml(process_xml(pid = "42.5"))),
  inventory_xml(gpu_xml(process_xml(pid = "2147483648"))),
  inventory_xml(gpu_xml(process_xml(pid = "0"))),
  inventory_xml(gpu_xml(uuid = "N/A")),
  inventory_xml(paste0(gpu_xml(), gpu_xml()), count = 2L),
  sub("<processes></processes>", "", inventory_xml(), fixed = TRUE),
  sub("<uuid>", "<uuid>duplicate</uuid><uuid>", inventory_xml(), fixed = TRUE),
  sub("<processes>", "<processes/><processes>", inventory_xml(), fixed = TRUE),
  sub("<display_active>", "<display_active/>", inventory_xml(), fixed = TRUE)
)) {
  assert_rejected(function() parse_inventory(invalid_xml), "refusing to assume")
}
# Every reported GPU is inspected, not just the first or the compute device.
second_uuid <- "GPU-08a1fcd0-7811-9bb0-b22e-1c9216ee0cd1"
two_gpu <- parse_inventory(inventory_xml(paste0(gpu_xml(),
  gpu_xml(process_xml("99", "G"), uuid = second_uuid)), count = 2L))
assert_rejected(function() benchmark_assert_idle_gpu(
  "self-test", two_gpu, current_pid = 42L), second_uuid)
# The live command failure path must not be treated as an empty inventory.
assert_rejected(function() benchmark_gpu_inventory(
  "cudaverse-intentionally-missing-nvidia-smi-command"), "Could not inspect")

# Exercise the default call chain used by retained runners with a command
# fixture. No live GPU query or workload is needed for these regressions.
command_fixture <- new.env(parent = baseenv())
sys.source(file.path("tools", "benchmark-gpu-guard.R"), command_fixture)
command_fixture$command_output <- inventory_xml()
command_fixture$command_error <- NULL
command_fixture$command_calls <- list()
command_fixture$system2 <- eval(quote(function(command, args, stdout, stderr) {
  command_calls[[length(command_calls) + 1L]] <<- list(
    command = command, args = args, stdout = stdout, stderr = stderr)
  if (!is.null(command_error)) stop(command_error)
  command_output
}), command_fixture)
stopifnot(isTRUE(command_fixture$benchmark_assert_idle_gpu(
  "default empty inventory", current_pid = 42L)))
stopifnot(identical(command_fixture$command_calls[[1L]], list(
  command = "nvidia-smi", args = c("-q", "-x"), stdout = TRUE, stderr = TRUE)))
for (xml in c(inventory_xml(gpu_xml(process_xml("99", "G"))),
               inventory_xml(gpu_xml(process_xml("42", "C+G"))),
               inventory_xml(gpu_xml(display = "Enabled")),
               inventory_xml(gpu_xml(processes = "N/A")))) {
  command_fixture$command_output <- xml
  assert_rejected(function() command_fixture$benchmark_assert_idle_gpu(
    "default unsafe inventory", current_pid = 42L))
}
command_fixture$command_output <- structure(inventory_xml(), status = 1L)
assert_rejected(function() command_fixture$benchmark_assert_idle_gpu(
  "failed command with plausible output", current_pid = 42L), "Could not inspect")
command_fixture$command_error <- "command unavailable"
assert_rejected(function() command_fixture$benchmark_assert_idle_gpu(
  "unavailable command", current_pid = 42L), "Could not inspect")
command_fixture$command_error <- NULL
command_fixture$command_output <- inventory_xml()
command_fixture$requireNamespace <- function(package, quietly) FALSE
assert_rejected(function() command_fixture$benchmark_assert_idle_gpu(
  "missing XML parser", current_pid = 42L), "xml2 is required")

# The recorder keeps successful, rejected, and unavailable observations before
# propagating a failure, including the context/PID/UTC evidence for each check.
record_inventory <- empty
record_error <- NULL
recorder <- benchmark_gpu_guard_recorder(
  read_inventory = function() {
    if (!is.null(record_error)) stop(record_error)
    record_inventory
  },
  clock = function() "2026-10-10 12:00:00 UTC",
  current_pid = function() 42L
)
stopifnot(isTRUE(recorder$check("startup")))
record_inventory <- display
assert_rejected(function() recorder$check("case-a/native/start"),
                 "active display context")
record_error <- "inventory unavailable"
assert_rejected(function() recorder$check("case-a/native/before sample"),
                 "inventory unavailable")
records <- recorder$snapshot()
stopifnot(length(records) == 3L,
          identical(vapply(records, `[[`, logical(1L), "passed"),
                    c(TRUE, FALSE, FALSE)),
          all(vapply(records, function(record) {
            identical(record$pid, 42L) &&
              identical(record$observed_at_utc, "2026-10-10 12:00:00 UTC")
          }, logical(1L))),
          identical(records[[2L]]$inventory, display),
          is.null(records[[3L]]$inventory),
          identical(records[[3L]]$error, "inventory unavailable"))
prior <- list(context = "prior session", passed = TRUE)
recorder$prepend(list(prior))
stopifnot(identical(recorder$snapshot(), c(list(prior), records)))
assert_rejected(function() recorder$prepend("invalid history"),
                 "must be a list")

# An interrupt in the final diagnostic query can arrive after eligibility was
# set. Exercise the actual driver's handler without running its GPU workload.
driver <- parse(file = file.path("tools", "run-matmul-precision-experiment.R"))
driver_body <- as.list(driver[[1L]][[3L]][[3L]])[-1L]
failure_assignment <- Filter(function(expression) {
  is.call(expression) && identical(expression[[1L]], as.name("<-")) &&
    identical(expression[[2L]], as.name("fail_report"))
}, driver_body)
stopifnot(length(failure_assignment) == 1L)
failure_fixture <- new.env(parent = baseenv())
eval(failure_assignment[[1L]], failure_fixture)
for (condition in list(simpleError("final diagnostic failed"),
                        structure(list(message = "interrupted"),
                                  class = c("interrupt", "condition")))) {
  failure_fixture$report <- list(passed = TRUE, complete = TRUE,
                                 retained_performance_eligible = TRUE)
  failure_fixture$fail_report(condition)
  stopifnot(identical(failure_fixture$report$passed, FALSE),
            identical(failure_fixture$report$complete, FALSE),
            identical(failure_fixture$report$retained_performance_eligible, FALSE),
            identical(failure_fixture$report$runtime_error,
                      conditionMessage(condition)))
}

message("Benchmark GPU guard positive and rejection self-tests passed.")
