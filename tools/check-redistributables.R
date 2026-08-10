check_redistributables <- function(root) {
  root <- normalizePath(root)
  package_files <- list.files(
    root, recursive = TRUE, full.names = TRUE, all.files = TRUE, no.. = TRUE
  )
  binary_pattern <- paste0(
    "(nvcuda|libcuda|cublas|cudart|cusparse|cusolver|cudnn|nvrtc|",
    "nvjitlink|torch|libtorch).*(\\.dll|\\.so(\\.[0-9]+)*|\\.dylib|",
    "\\.cubin|\\.fatbin)$"
  )
  bundled <- package_files[grepl(
    binary_pattern,
    basename(package_files),
    ignore.case = TRUE,
    perl = TRUE
  )]
  if (length(bundled)) {
    stop(
      "Unapproved redistributable candidate(s):\n",
      paste0("- ", bundled, collapse = "\n"),
      call. = FALSE
    )
  }

  device_code <- package_files[grepl(
    "\\.(ptx|cubin|fatbin)$", package_files, ignore.case = TRUE, perl = TRUE
  )]
  allowed_relative_paths <- c(
    file.path("inst", "kernels", "cudaverse_dense_kernels.ptx"),
    file.path("kernels", "cudaverse_dense_kernels.ptx"),
    file.path("cudaverse", "inst", "kernels",
              "cudaverse_dense_kernels.ptx"),
    file.path("cudaverse", "kernels", "cudaverse_dense_kernels.ptx")
  )
  allowed_ptx <- normalizePath(
    file.path(root, allowed_relative_paths), mustWork = FALSE
  )
  unexpected_device_code <- setdiff(
    normalizePath(device_code, mustWork = FALSE), allowed_ptx
  )
  if (length(unexpected_device_code)) {
    stop(
      "Unexpected CUDA device-code payload(s):\n",
      paste0("- ", unexpected_device_code, collapse = "\n"),
      call. = FALSE
    )
  }
  if (length(device_code) != 1L) {
    stop(
      "Expected exactly one package-owned CUDA-kernel PTX; found ",
      length(device_code), ".",
      call. = FALSE
    )
  }
  message(paste(
    "Redistribution gate passed: no NVIDIA or LibTorch runtime binary is",
    "bundled; only the package-owned, checksum-pinned CUDA-kernel PTX is present."
  ))
  invisible(TRUE)
}

if (sys.nframe() == 0L) {
  arguments <- commandArgs(trailingOnly = TRUE)
  check_redistributables(if (length(arguments)) arguments[[1L]] else getwd())
}
