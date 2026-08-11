# Subset and replace tensor values

Tensor indices follow ordinary one-based R array semantics. Subsetting
returns a `cudatensor`, including when a single value is selected.
Replacement preserves the tensor dtype; fractional values therefore
cannot be assigned to an integer tensor.

## Usage

``` r
# S3 method for class 'cudatensor'
x[..., drop = TRUE]

# S3 method for class 'cudatensor'
x[...] <- value
```

## Arguments

- x:

  A `cudatensor`.

- ...:

  One-based R array indices.

- drop:

  Whether dimensions of length one are dropped.

- value:

  Numeric replacement values or another `cudatensor`.

## Value

A `cudatensor` on the same device as `x`.

## Details

Backends may implement value gathering and replacement directly. The
native CUDA backend evaluates only R index metadata on the host and
keeps tensor values on the device. Compatibility backends without
indexing operations use a recorded CPU round trip. Subscripts containing
`NA` currently use the compatibility path.
