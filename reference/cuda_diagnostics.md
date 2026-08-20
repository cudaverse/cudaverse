# Diagnose the optional CUDA runtime

Inspecting the runtime is non-destructive and never installs or
downloads torch. The returned `reason` is suitable for logs and
provenance.

## Usage

``` r
cuda_diagnostics()
```

## Value

A named list containing the legacy fields `torch_installed`,
`torch_version`, `cuda_available`, `cuda_device_count`, `reason`, and
`detection_error`, plus `available_backends`, `auto_eligible_backends`,
`auto_selection_reason`, `selected_backend`, and per-backend diagnostic
details. The additive `status`, `summary`, `next_steps`, and
`backend_status` fields provide a user-facing health result without
removing the machine-readable details. Each backend detail distinguishes
advertised `capabilities` from callable internal `operations`. Native
automatic eligibility requires a compatible backend contract, the
complete tensor/algorithm capability set, all runtime components, and a
passing cached self-test. The legacy fields are retained throughout the
0.4 compatibility cycle.

## Examples

``` r
cuda_diagnostics()
#> <cuda_diagnostics status=cpu_only available=FALSE devices=NA selected=base torch=0.17.0 reason=backend_error>
#> CUDA is unavailable; automatic requests will use the base CPU backend (backend_error).
#> Next steps:
#> - Inspect backend_status$error and backend_diagnostics for the failing runtime probe before retrying CUDA.
```
