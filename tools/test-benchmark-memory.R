sys.source(file.path("tools", "benchmark-memory.R"), envir = environment())

native_snapshots <- list(
  list(allocated_bytes = 10, allocated_peak_bytes = 10, used_bytes = 100),
  list(allocated_bytes = 30, allocated_peak_bytes = 70, used_bytes = 160),
  list(allocated_bytes = 10, allocated_peak_bytes = 70, used_bytes = 104)
)
snapshot_index <- 0L
reset_calls <- 0L
synchronize_calls <- 0L
native <- benchmark_measure_memory(
  "native",
  run = function() double(20L),
  synchronize = function(backend) synchronize_calls <<- synchronize_calls + 1L,
  memory_query = function(backend) {
    snapshot_index <<- snapshot_index + 1L
    native_snapshots[[snapshot_index]]
  },
  reset_native_peak = function() reset_calls <<- reset_calls + 1L
)
stopifnot(
  identical(reset_calls, 1L),
  identical(synchronize_calls, 3L),
  identical(native$backend_allocator_peak_bytes, 60),
  identical(native$tracked_current_post_cleanup_difference_bytes, 0),
  identical(native$whole_device_used_before_bytes, 100),
  identical(native$whole_device_used_with_result_bytes, 160),
  identical(native$whole_device_used_after_cleanup_bytes, 104),
  identical(native$whole_device_post_cleanup_absolute_difference_bytes, 4)
)

torch_snapshots <- list(
  list(allocated_bytes = 512, allocated_peak_bytes = 1024,
       used_bytes = NA_real_),
  list(allocated_bytes = 1024, allocated_peak_bytes = 4096,
       used_bytes = NA_real_),
  list(allocated_bytes = 512, allocated_peak_bytes = 4096,
       used_bytes = NA_real_)
)
snapshot_index <- 0L
torch <- benchmark_measure_memory(
  "torch",
  run = function() TRUE,
  synchronize = function(backend) invisible(TRUE),
  memory_query = function(backend) {
    snapshot_index <<- snapshot_index + 1L
    torch_snapshots[[snapshot_index]]
  },
  reset_native_peak = function() stop("torch must not reset native peak")
)
stopifnot(
  identical(torch$backend_allocator_peak_bytes, 3584),
  is.na(torch$tracked_current_post_cleanup_difference_bytes),
  is.na(torch$whole_device_post_cleanup_absolute_difference_bytes)
)

base_snapshots <- replicate(
  3L,
  list(allocated_bytes = NA_real_, allocated_peak_bytes = NA_real_,
       used_bytes = NA_real_),
  simplify = FALSE
)
snapshot_index <- 0L
base <- benchmark_measure_memory(
  "base",
  run = function() TRUE,
  synchronize = function(backend) invisible(TRUE),
  memory_query = function(backend) {
    snapshot_index <<- snapshot_index + 1L
    base_snapshots[[snapshot_index]]
  },
  reset_native_peak = function() stop("base must not reset native peak")
)
stopifnot(
  identical(base$backend_allocator_peak_bytes, 0),
  is.na(base$tracked_current_post_cleanup_difference_bytes)
)

message("Benchmark memory-contract self-tests passed.")
