#include <float.h>
#include <limits.h>
#include <math.h>

#define CUDAVERSE_MAX_RANK 8
#define CUDAVERSE_FAST_TOPK 32

struct ReductionMeta {
  int rank;
  int output_rank;
  int keepdim;
  int operation;
  int shape[CUDAVERSE_MAX_RANK];
  int reduced[CUDAVERSE_MAX_RANK];
  int output_shape[CUDAVERSE_MAX_RANK];
  int output_to_input[CUDAVERSE_MAX_RANK];
};

struct BroadcastMeta {
  int rank;
  int source_shape[CUDAVERSE_MAX_RANK];
  int target_shape[CUDAVERSE_MAX_RANK];
  unsigned long long source_stride[CUDAVERSE_MAX_RANK];
};

template <typename T, typename Accumulator>
__device__ void reduce_value(const T* input, T* output,
                             const ReductionMeta& meta,
                             unsigned long long output_index) {
  int coordinate[CUDAVERSE_MAX_RANK] = {0};
  unsigned long long remainder = output_index;
  for (int dimension = 0; dimension < meta.output_rank; ++dimension) {
    int extent = meta.output_shape[dimension];
    int value = static_cast<int>(remainder % extent);
    remainder /= extent;
    int input_dimension = meta.output_to_input[dimension];
    if (input_dimension >= 0) coordinate[input_dimension] = value;
  }

  unsigned long long combinations = 1;
  for (int dimension = 0; dimension < meta.rank; ++dimension) {
    if (meta.reduced[dimension]) combinations *= meta.shape[dimension];
  }

  Accumulator total = static_cast<Accumulator>(0);
  for (unsigned long long reduction_index = 0;
       reduction_index < combinations; ++reduction_index) {
    unsigned long long reduced_remainder = reduction_index;
    for (int dimension = 0; dimension < meta.rank; ++dimension) {
      if (meta.reduced[dimension]) {
        coordinate[dimension] = static_cast<int>(
            reduced_remainder % meta.shape[dimension]);
        reduced_remainder /= meta.shape[dimension];
      }
    }
    unsigned long long input_index = 0;
    unsigned long long stride = 1;
    for (int dimension = 0; dimension < meta.rank; ++dimension) {
      input_index += static_cast<unsigned long long>(coordinate[dimension]) *
                     stride;
      stride *= meta.shape[dimension];
    }
    total += static_cast<Accumulator>(input[input_index]);
  }
  if (meta.operation == 1) {
    total /= static_cast<Accumulator>(combinations);
  }
  output[output_index] = static_cast<T>(total);
}

extern "C" __global__ void cudaverse_reduce_f64(
    const double* input, double* output, ReductionMeta meta,
    unsigned long long output_elements) {
  unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < output_elements) {
    reduce_value<double, double>(input, output, meta, index);
  }
}

extern "C" __global__ void cudaverse_reduce_f32(
    const float* input, float* output, ReductionMeta meta,
    unsigned long long output_elements) {
  unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < output_elements) {
    reduce_value<float, double>(input, output, meta, index);
  }
}

extern "C" __global__ void cudaverse_reduce_i32(
    const int* input, int* output, ReductionMeta meta,
    unsigned long long output_elements) {
  unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < output_elements) {
    reduce_value<int, long long>(input, output, meta, index);
  }
}

extern "C" __global__ void cudaverse_cast_i32_f64(
    const int* input, double* output, unsigned long long elements) {
  unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < elements) output[index] = static_cast<double>(input[index]);
}

extern "C" __global__ void cudaverse_cast_f32_f64(
    const float* input, double* output, unsigned long long elements) {
  unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < elements) output[index] = static_cast<double>(input[index]);
}

extern "C" __global__ void cudaverse_cast_f64_f32(
    const double* input, float* output, unsigned long long elements) {
  unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < elements) output[index] = static_cast<float>(input[index]);
}

extern "C" __global__ void cudaverse_cast_i32_f32(
    const int* input, float* output, unsigned long long elements) {
  unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < elements) output[index] = static_cast<float>(input[index]);
}

extern "C" __global__ void cudaverse_cast_f64_i32(
    const double* input, int* output, unsigned long long elements) {
  unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < elements) output[index] = static_cast<int>(input[index]);
}

extern "C" __global__ void cudaverse_cast_f32_i32(
    const float* input, int* output, unsigned long long elements) {
  unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < elements) output[index] = static_cast<int>(input[index]);
}

template <typename T>
__device__ T binary_value(T left, T right, int operation) {
  if (operation == 0) return left + right;
  if (operation == 1) return left - right;
  if (operation == 2) return left * right;
  if (operation == 3) return left / right;
  return static_cast<T>(pow(left, right));
}

template <typename T>
__device__ void binary_kernel(const T* left, const T* right, T* output,
                              int operation, unsigned long long elements) {
  unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < elements) {
    output[index] = binary_value(left[index], right[index], operation);
  }
}

extern "C" __global__ void cudaverse_binary_f64(
    const double* left, const double* right, double* output,
    int operation, unsigned long long elements) {
  binary_kernel(left, right, output, operation, elements);
}

extern "C" __global__ void cudaverse_binary_f32(
    const float* left, const float* right, float* output,
    int operation, unsigned long long elements) {
  binary_kernel(left, right, output, operation, elements);
}

template <typename T>
__device__ void broadcast_kernel(const T* input, T* output,
                                 const BroadcastMeta& meta,
                                 unsigned long long elements) {
  unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= elements) return;
  unsigned long long remainder = index;
  unsigned long long source_index = 0;
  for (int dimension = 0; dimension < meta.rank; ++dimension) {
    int coordinate = static_cast<int>(
        remainder % meta.target_shape[dimension]);
    remainder /= meta.target_shape[dimension];
    if (meta.source_shape[dimension] != 1) {
      source_index += static_cast<unsigned long long>(coordinate) *
                      meta.source_stride[dimension];
    }
  }
  output[index] = input[source_index];
}

extern "C" __global__ void cudaverse_broadcast_f64(
    const double* input, double* output, BroadcastMeta meta,
    unsigned long long elements) {
  broadcast_kernel(input, output, meta, elements);
}

extern "C" __global__ void cudaverse_broadcast_f32(
    const float* input, float* output, BroadcastMeta meta,
    unsigned long long elements) {
  broadcast_kernel(input, output, meta, elements);
}

extern "C" __global__ void cudaverse_broadcast_i32(
    const int* input, int* output, BroadcastMeta meta,
    unsigned long long elements) {
  broadcast_kernel(input, output, meta, elements);
}

extern "C" __global__ void cudaverse_column_stats_f64(
    const double* input, double* center, double* scale, int rows, int columns,
    int use_center, int use_scale) {
  int column = blockIdx.x * blockDim.x + threadIdx.x;
  if (column >= columns) return;

  double mean = 0.0;
  double m2 = 0.0;
  double sum_squares = 0.0;
  const double* values = input + static_cast<unsigned long long>(column) * rows;
  for (int row = 0; row < rows; ++row) {
    double value = values[row];
    double delta = value - mean;
    mean += delta / static_cast<double>(row + 1);
    m2 += delta * (value - mean);
    sum_squares += value * value;
  }
  center[column] = use_center ? mean : 0.0;
  if (use_scale) {
    double numerator = use_center ? m2 : sum_squares;
    scale[column] = sqrt(numerator / static_cast<double>(rows - 1));
  } else {
    scale[column] = 1.0;
  }
}

extern "C" __global__ void cudaverse_center_scale_f64(
    const double* input, double* output, const double* center,
    const double* scale, int rows, int columns) {
  unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long long elements =
      static_cast<unsigned long long>(rows) * columns;
  if (index >= elements) return;
  int column = static_cast<int>(index / rows);
  output[index] = (input[index] - center[column]) / scale[column];
}

extern "C" __global__ void cudaverse_scale_columns_f64(
    const double* input, double* output, const double* scale,
    int rows, int columns) {
  unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long long elements =
      static_cast<unsigned long long>(rows) * columns;
  if (index >= elements) return;
  int column = static_cast<int>(index / rows);
  output[index] = input[index] * scale[column];
}

template <typename T>
__device__ void transpose_kernel(const T* input, T* output,
                                 int rows, int columns) {
  unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long long elements =
      static_cast<unsigned long long>(rows) * columns;
  if (index >= elements) return;
  int row = static_cast<int>(index % rows);
  int column = static_cast<int>(index / rows);
  output[column + static_cast<unsigned long long>(row) * columns] =
      input[index];
}

extern "C" __global__ void cudaverse_transpose_f64(
    const double* input, double* output, int rows, int columns) {
  transpose_kernel(input, output, rows, columns);
}

extern "C" __global__ void cudaverse_transpose_f32(
    const float* input, float* output, int rows, int columns) {
  transpose_kernel(input, output, rows, columns);
}

extern "C" __global__ void cudaverse_transpose_i32(
    const int* input, int* output, int rows, int columns) {
  transpose_kernel(input, output, rows, columns);
}

extern "C" __global__ void cudaverse_gather_rows_f64(
    const double* input, double* output, int rows, int columns,
    int first_row, int selected_rows) {
  unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long long elements =
      static_cast<unsigned long long>(selected_rows) * columns;
  if (index >= elements) return;
  int row = static_cast<int>(index % selected_rows);
  int column = static_cast<int>(index / selected_rows);
  output[index] = input[first_row + row +
                        static_cast<unsigned long long>(column) * rows];
}

extern "C" __global__ void cudaverse_normalize_rows_f64(
    const double* input, double* output, int rows, int columns) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) return;
  double scale = 0.0;
  for (int column = 0; column < columns; ++column) {
    double value = fabs(input[row + static_cast<unsigned long long>(column) * rows]);
    if (value > scale) scale = value;
  }
  double sum = 0.0;
  if (scale > 0.0) {
    for (int column = 0; column < columns; ++column) {
      double value = input[row + static_cast<unsigned long long>(column) * rows] /
                     scale;
      sum += value * value;
    }
  }
  double norm = scale * sqrt(sum);
  for (int column = 0; column < columns; ++column) {
    unsigned long long index = row +
        static_cast<unsigned long long>(column) * rows;
    output[index] = input[index] / norm;
  }
}

extern "C" __global__ void cudaverse_row_norms_squared_f64(
    const double* input, double* output, int rows, int columns) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) return;
  double total = 0.0;
  for (int column = 0; column < columns; ++column) {
    double value = input[row + static_cast<unsigned long long>(column) * rows];
    total += value * value;
  }
  output[row] = total;
}

extern "C" __global__ void cudaverse_distance_from_gram_f64(
    const double* gram, const double* query, const double* reference,
    const double* query_norms, const double* reference_norms, double* distance,
    int query_rows, int reference_rows, int columns, int metric,
    int query_offset, int self) {
  unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long long elements =
      static_cast<unsigned long long>(query_rows) * reference_rows;
  if (index >= elements) return;
  int query_row = static_cast<int>(index % query_rows);
  int reference_row = static_cast<int>(index / query_rows);
  if (self && query_offset + query_row == reference_row) {
    distance[index] = 0.0;
    return;
  }
  double cross = gram[index];
  if (metric == 1) {
    double value = 1.0 - cross;
    distance[index] = value < 0.0 ? 0.0 : (value > 2.0 ? 2.0 : value);
    return;
  }

  double magnitude = query_norms[query_row] + reference_norms[reference_row];
  double squared = magnitude - 2.0 * cross;
  bool risky = !isfinite(squared) || squared < 0.0 ||
      squared <= 1e-12 * magnitude;
  if (!risky) {
    distance[index] = sqrt(squared);
    return;
  }

  double direct_scale = 0.0;
  for (int column = 0; column < columns; ++column) {
    double difference = query[query_row +
        static_cast<unsigned long long>(column) * query_rows] -
        reference[reference_row +
        static_cast<unsigned long long>(column) * reference_rows];
    double absolute = fabs(difference);
    if (absolute > direct_scale) direct_scale = absolute;
  }
  if (direct_scale == 0.0) {
    distance[index] = 0.0;
    return;
  }
  if (isinf(direct_scale)) {
    distance[index] = INFINITY;
    return;
  }
  double direct_sum = 0.0;
  for (int column = 0; column < columns; ++column) {
    double difference = query[query_row +
        static_cast<unsigned long long>(column) * query_rows] -
        reference[reference_row +
        static_cast<unsigned long long>(column) * reference_rows];
    double scaled = difference / direct_scale;
    direct_sum += scaled * scaled;
  }
  distance[index] = direct_scale * sqrt(direct_sum);
}

__device__ bool cudaverse_better(double left_distance, int left_index,
                                 double right_distance, int right_index) {
  return left_distance < right_distance ||
      (left_distance == right_distance && left_index < right_index);
}

__device__ void cudaverse_insert(double value, int index, double* distances,
                                 int* indices, int k) {
  if (!cudaverse_better(value, index, distances[k - 1], indices[k - 1])) return;
  int position = k - 1;
  while (position > 0 &&
         cudaverse_better(value, index, distances[position - 1],
                          indices[position - 1])) {
    distances[position] = distances[position - 1];
    indices[position] = indices[position - 1];
    --position;
  }
  distances[position] = value;
  indices[position] = index;
}

extern "C" __global__ void cudaverse_topk_stable_f64(
    const double* distance, int* output_index, double* output_distance,
    int query_rows, int reference_rows, int k, int query_offset, int self) {
  int query_row = blockIdx.x;
  int thread = threadIdx.x;
  if (query_row >= query_rows || k > CUDAVERSE_FAST_TOPK) return;

  double local_distance[CUDAVERSE_FAST_TOPK];
  int local_index[CUDAVERSE_FAST_TOPK];
  for (int rank = 0; rank < k; ++rank) {
    local_distance[rank] = DBL_MAX;
    local_index[rank] = INT_MAX;
  }
  for (int candidate = thread; candidate < reference_rows;
       candidate += blockDim.x) {
    if (self && query_offset + query_row == candidate) continue;
    double value = distance[query_row +
        static_cast<unsigned long long>(candidate) * query_rows];
    cudaverse_insert(value, candidate, local_distance, local_index, k);
  }

  extern __shared__ unsigned char shared_bytes[];
  double* shared_distance = reinterpret_cast<double*>(shared_bytes);
  int* shared_index = reinterpret_cast<int*>(
      shared_distance + static_cast<unsigned long long>(blockDim.x) * k);
  for (int rank = 0; rank < k; ++rank) {
    int slot = thread * k + rank;
    shared_distance[slot] = local_distance[rank];
    shared_index[slot] = local_index[rank];
  }
  __syncthreads();

  if (thread == 0) {
    double best_distance[CUDAVERSE_FAST_TOPK];
    int best_index[CUDAVERSE_FAST_TOPK];
    for (int rank = 0; rank < k; ++rank) {
      best_distance[rank] = DBL_MAX;
      best_index[rank] = INT_MAX;
    }
    for (int candidate = 0; candidate < blockDim.x * k; ++candidate) {
      cudaverse_insert(shared_distance[candidate], shared_index[candidate],
                       best_distance, best_index, k);
    }
    for (int rank = 0; rank < k; ++rank) {
      unsigned long long output = query_row +
          static_cast<unsigned long long>(rank) * query_rows;
      output_index[output] = best_index[rank] + 1;
      output_distance[output] = best_distance[rank];
    }
  }
}

extern "C" __global__ void cudaverse_topk_stable_general_f64(
    const double* distance, int* output_index, double* output_distance,
    int query_rows, int reference_rows, int k, int query_offset, int self) {
  int query_row = blockIdx.x * blockDim.x + threadIdx.x;
  if (query_row >= query_rows) return;
  for (int rank = 0; rank < k; ++rank) {
    unsigned long long output = query_row +
        static_cast<unsigned long long>(rank) * query_rows;
    output_index[output] = INT_MAX;
    output_distance[output] = DBL_MAX;
  }
  for (int candidate = 0; candidate < reference_rows; ++candidate) {
    if (self && query_offset + query_row == candidate) continue;
    double value = distance[query_row +
        static_cast<unsigned long long>(candidate) * query_rows];
    int candidate_index = candidate + 1;
    int position = k - 1;
    unsigned long long last = query_row +
        static_cast<unsigned long long>(position) * query_rows;
    if (!cudaverse_better(value, candidate_index, output_distance[last],
                          output_index[last])) continue;
    while (position > 0) {
      unsigned long long previous = query_row +
          static_cast<unsigned long long>(position - 1) * query_rows;
      if (!cudaverse_better(value, candidate_index,
                            output_distance[previous],
                            output_index[previous])) break;
      unsigned long long current = query_row +
          static_cast<unsigned long long>(position) * query_rows;
      output_distance[current] = output_distance[previous];
      output_index[current] = output_index[previous];
      --position;
    }
    unsigned long long output = query_row +
        static_cast<unsigned long long>(position) * query_rows;
    output_distance[output] = value;
    output_index[output] = candidate_index;
  }
}

extern "C" __global__ void cudaverse_fill_f64(
    double* output, double value, unsigned long long elements) {
  unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < elements) output[index] = value;
}

extern "C" __global__ void cudaverse_sparse_row_sums_f64(
    const int* row_ptr, const double* values, double* output, int rows) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) return;
  double total = 0.0;
  for (int position = row_ptr[row]; position < row_ptr[row + 1]; ++position) {
    total += values[position];
  }
  output[row] = total;
}

extern "C" __global__ void cudaverse_sparse_col_sums_f64(
    const int* col_index, const double* values, double* output, int nnz) {
  int position = blockIdx.x * blockDim.x + threadIdx.x;
  if (position < nnz) atomicAdd(output + col_index[position], values[position]);
}

extern "C" __global__ void cudaverse_sparse_normalize_f64(
    const int* row_index, const int* col_index, const double* values,
    const double* sums, double* output, int nnz, int margin,
    double scale_factor, int use_log1p) {
  int position = blockIdx.x * blockDim.x + threadIdx.x;
  if (position >= nnz) return;
  int group = margin == 0 ? row_index[position] : col_index[position];
  double value = values[position] * scale_factor / sums[group];
  output[position] = use_log1p ? log1p(value) : value;
}

extern "C" __global__ void cudaverse_csr_spmm_f64(
    const int* row_ptr, const int* col_index, const double* values,
    const double* dense, double* output, int rows, int dense_rows,
    int dense_columns) {
  unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long long elements =
      static_cast<unsigned long long>(rows) * dense_columns;
  if (index >= elements) return;
  int row = static_cast<int>(index % rows);
  int column = static_cast<int>(index / rows);
  double total = 0.0;
  for (int position = row_ptr[row]; position < row_ptr[row + 1]; ++position) {
    int inner = col_index[position];
    total += values[position] *
        dense[inner + static_cast<unsigned long long>(column) * dense_rows];
  }
  output[index] = total;
}

extern "C" __global__ void cudaverse_csr_to_dense_f64(
    const int* row_index, const int* col_index, const double* values,
    double* output, int rows, int nnz) {
  int position = blockIdx.x * blockDim.x + threadIdx.x;
  if (position >= nnz) return;
  int row = row_index[position];
  int column = col_index[position];
  output[row + static_cast<unsigned long long>(column) * rows] =
      values[position];
}
