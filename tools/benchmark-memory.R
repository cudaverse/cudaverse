benchmark_measure_memory <- function(backend, run, synchronize,
                                     memory_query, reset_native_peak) {
  synchronize(backend)
  invisible(gc())
  if (identical(backend, "native")) reset_native_peak()
  before <- memory_query(backend)

  value <- run()
  synchronize(backend)
  during <- memory_query(backend)
  value <- NULL
  invisible(gc())
  synchronize(backend)
  after <- memory_query(backend)

  allocator_peak <- if (identical(backend, "base")) {
    0
  } else {
    during$allocated_peak_bytes - before$allocated_bytes
  }
  current_difference <- if (identical(backend, "native")) {
    after$allocated_bytes - before$allocated_bytes
  } else {
    NA_real_
  }
  list(
    backend_allocator_peak_bytes = allocator_peak,
    backend_allocator_peak_source = if (identical(backend, "native")) {
      "cuda_memory_info native cudaverse-owned allocation tracker"
    } else if (identical(backend, "torch")) {
      "cuda_memory_info torch allocator session high-water mark"
    } else {
      "not applicable (CPU)"
    },
    tracked_current_post_cleanup_difference_bytes = current_difference,
    whole_device_used_before_bytes = before$used_bytes,
    whole_device_used_with_result_bytes = during$used_bytes,
    whole_device_used_after_cleanup_bytes = after$used_bytes,
    whole_device_post_cleanup_absolute_difference_bytes =
      abs(after$used_bytes - before$used_bytes)
  )
}
