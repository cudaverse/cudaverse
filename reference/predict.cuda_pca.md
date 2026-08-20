# Project observations with a fitted CUDA-aware PCA model

`predict.cuda_pca()` applies the fitted centring, scaling, and loadings
to new observations. Named features may be supplied in any order and are
aligned safely before projection. If the fitted model has feature names,
unnamed or mismatched columns are rejected instead of being used in the
wrong order.

## Usage

``` r
# S3 method for class 'cuda_pca'
predict(object, newdata, device = c("model", "auto", "cuda", "cpu"), ...)
```

## Arguments

- object:

  A fitted `cuda_pca` object.

- newdata:

  A finite numeric matrix or data frame with observations in rows and
  the model features in columns. When omitted, the training scores in
  `object$x` are returned.

- device:

  Where to compute the projection. `"model"` reuses the actual device of
  the fitted model; `"auto"`, `"cuda"`, and `"cpu"` follow the usual
  cudaverse device-selection rules.

- ...:

  Must be empty.

## Value

A numeric matrix of component scores. New observation names and stable
component names are retained. A recomputed prediction includes
stage-level provenance and is materialized as an R matrix on the CPU.
The native backend also retains shared device storage so a subsequent
native distance or kNN operation can reuse the scores without uploading
them. Omitting `newdata` returns the validated stored training scores
unchanged; that retrieval does not create a prediction stage.

## See also

[`cuda_pca()`](https://cudaverse.github.io/cudaverse/reference/cuda_pca.md)

## Examples

``` r
train <- as.matrix(iris[1:100, 1:4])
fit <- cuda_pca(train, n_components = 2, device = "cpu")
predict(fit, as.matrix(iris[101:105, 1:4]), device = "cpu")
#>          PC1        PC2
#> 101 3.532286 -0.3768000
#> 102 2.491451  0.3064927
#> 103 3.622220 -0.6979323
#> 104 3.020128 -0.1303527
#> 105 3.374626 -0.3093205
#> attr(,"device")
#> [1] "cpu"
#> attr(,"provenance_schema")
#> [1] "cudaverse-stage/1"
#> attr(,"requested_device")
#> [1] "cpu"
#> attr(,"compute_device")
#> [1] "cpu"
#> attr(,"compute_stages")
#> attr(,"compute_stages")$projection
#> $requested_device
#> [1] "cpu"
#> 
#> $device
#> [1] "cpu"
#> 
#> $backend
#> [1] "base"
#> 
#> $selection_reason
#> [1] "explicit_cpu"
#> 
#> $fallback
#> [1] FALSE
#> 
#> $output_device
#> [1] "cpu"
#> 
#> attr(,"class")
#> [1] "cuda_stage"
#> 
#> attr(,"backend")
#> [1] "base"
#> attr(,"parameters")
#> attr(,"parameters")$n_components
#> [1] 2
#> 
#> attr(,"source_device")
#> [1] "cpu"
#> attr(,"source_class")
#> [1] "matrix"
```
