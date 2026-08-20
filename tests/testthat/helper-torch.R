torch_cpu_runtime_available <- function() {
  if (!requireNamespace("torch", quietly = TRUE) ||
      !cudaverse:::.torch_stable_sort_available()) {
    return(FALSE)
  }
  isTRUE(tryCatch({
    torch::torch_zeros(1L, device = "cpu")
    TRUE
  }, error = function(error) FALSE))
}
