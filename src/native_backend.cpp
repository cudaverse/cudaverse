#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Include the C++ standard library before R headers. Rinternals.h defines a
// legacy length() macro that otherwise collides with libc++ locale methods on
// macOS.
#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>
#include <R_ext/Utils.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace {

using CUresult = int;
using CUdevice = int;
using CUcontext = void*;
using CUdeviceptr = std::uint64_t;
using CUmodule = void*;
using CUfunction = void*;
using CUstream = void*;
using cublasHandle_t = void*;
using cublasStatus_t = int;
using cusolverDnHandle_t = void*;
using cusolverStatus_t = int;

constexpr CUresult CUDA_SUCCESS = 0;
constexpr cublasStatus_t CUBLAS_STATUS_SUCCESS = 0;
constexpr cusolverStatus_t CUSOLVER_STATUS_SUCCESS = 0;
constexpr int CUBLAS_OP_N = 0;
constexpr int CUBLAS_OP_T = 1;

using cuInit_t = CUresult (*)(unsigned int);
using cuDriverGetVersion_t = CUresult (*)(int*);
using cuDeviceGetCount_t = CUresult (*)(int*);
using cuDeviceGet_t = CUresult (*)(CUdevice*, int);
using cuDevicePrimaryCtxRetain_t = CUresult (*)(CUcontext*, CUdevice);
using cuCtxSetCurrent_t = CUresult (*)(CUcontext);
using cuMemAlloc_t = CUresult (*)(CUdeviceptr*, std::size_t);
using cuMemFree_t = CUresult (*)(CUdeviceptr);
using cuMemcpyHtoD_t = CUresult (*)(CUdeviceptr, const void*, std::size_t);
using cuMemcpyDtoH_t = CUresult (*)(void*, CUdeviceptr, std::size_t);
using cuMemcpyDtoD_t = CUresult (*)(CUdeviceptr, CUdeviceptr, std::size_t);
using cuCtxSynchronize_t = CUresult (*)();
using cuMemGetInfo_t = CUresult (*)(std::size_t*, std::size_t*);
using cuGetErrorName_t = CUresult (*)(CUresult, const char**);
using cuGetErrorString_t = CUresult (*)(CUresult, const char**);
using cuModuleLoad_t = CUresult (*)(CUmodule*, const char*);
using cuModuleUnload_t = CUresult (*)(CUmodule);
using cuModuleGetFunction_t = CUresult (*)(CUfunction*, CUmodule, const char*);
using cuLaunchKernel_t = CUresult (*)(
    CUfunction, unsigned int, unsigned int, unsigned int,
    unsigned int, unsigned int, unsigned int, unsigned int, CUstream,
    void**, void**);

using cublasCreate_t = cublasStatus_t (*)(cublasHandle_t*);
using cublasDgemm_t = cublasStatus_t (*)(
    cublasHandle_t, int, int, int, int, int, const double*,
    const double*, int, const double*, int, const double*, double*, int);
using cublasSgemm_t = cublasStatus_t (*)(
    cublasHandle_t, int, int, int, int, int, const float*,
    const float*, int, const float*, int, const float*, float*, int);
using cusolverDnCreate_t = cusolverStatus_t (*)(cusolverDnHandle_t*);
using cusolverDnDgesvd_bufferSize_t = cusolverStatus_t (*)(
    cusolverDnHandle_t, int, int, int*);
using cusolverDnDgesvd_t = cusolverStatus_t (*)(
    cusolverDnHandle_t, signed char, signed char, int, int, double*, int,
    double*, double*, int, double*, int, double*, int, double*, int*);

struct BackendApi {
  void* driver = nullptr;
  void* cublas = nullptr;
  void* cusolver = nullptr;
  CUcontext context = nullptr;
  CUmodule kernels = nullptr;
  cublasHandle_t cublas_handle = nullptr;
  cusolverDnHandle_t cusolver_handle = nullptr;
  int driver_version = 0;
  int device_count = 0;

  cuInit_t cuInit = nullptr;
  cuDriverGetVersion_t cuDriverGetVersion = nullptr;
  cuDeviceGetCount_t cuDeviceGetCount = nullptr;
  cuDeviceGet_t cuDeviceGet = nullptr;
  cuDevicePrimaryCtxRetain_t cuDevicePrimaryCtxRetain = nullptr;
  cuCtxSetCurrent_t cuCtxSetCurrent = nullptr;
  cuMemAlloc_t cuMemAlloc = nullptr;
  cuMemFree_t cuMemFree = nullptr;
  cuMemcpyHtoD_t cuMemcpyHtoD = nullptr;
  cuMemcpyDtoH_t cuMemcpyDtoH = nullptr;
  cuMemcpyDtoD_t cuMemcpyDtoD = nullptr;
  cuCtxSynchronize_t cuCtxSynchronize = nullptr;
  cuMemGetInfo_t cuMemGetInfo = nullptr;
  cuGetErrorName_t cuGetErrorName = nullptr;
  cuGetErrorString_t cuGetErrorString = nullptr;
  cuModuleLoad_t cuModuleLoad = nullptr;
  cuModuleUnload_t cuModuleUnload = nullptr;
  cuModuleGetFunction_t cuModuleGetFunction = nullptr;
  cuLaunchKernel_t cuLaunchKernel = nullptr;
  cublasCreate_t cublasCreate = nullptr;
  cublasDgemm_t cublasDgemm = nullptr;
  cublasSgemm_t cublasSgemm = nullptr;
  cusolverDnCreate_t cusolverDnCreate = nullptr;
  cusolverDnDgesvd_bufferSize_t cusolverDnDgesvd_bufferSize = nullptr;
  cusolverDnDgesvd_t cusolverDnDgesvd = nullptr;
};

BackendApi api;

std::atomic<std::size_t> tracked_device_bytes{0};
std::atomic<std::size_t> peak_tracked_device_bytes{0};

void track_device_allocation(std::size_t bytes) {
  std::size_t current = tracked_device_bytes.fetch_add(bytes) + bytes;
  std::size_t peak = peak_tracked_device_bytes.load();
  while (current > peak &&
         !peak_tracked_device_bytes.compare_exchange_weak(peak, current)) {
  }
}

void track_device_release(std::size_t bytes) {
  tracked_device_bytes.fetch_sub(bytes);
}

void release_tracked_device_memory(CUdeviceptr pointer,
                                   std::size_t bytes) noexcept {
  if (pointer == 0 || api.cuMemFree == nullptr) return;
  if (api.cuMemFree(pointer) == CUDA_SUCCESS) track_device_release(bytes);
}

enum class DType { Float64, Float32, Integer };

constexpr int CUDAVERSE_MAX_RANK = 8;

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

struct SharedAllocation {
  CUdeviceptr pointer;
  std::size_t bytes;
  std::atomic<unsigned int> references;
};

struct SharedBuffer {
  CUdeviceptr pointer;
  std::size_t bytes;
  std::size_t elements;
  DType dtype;
  std::vector<int> shape;
  std::atomic<unsigned int> references;
  SharedAllocation* allocation;
};

struct SharedSparseBuffer {
  CUdeviceptr row_index;
  CUdeviceptr row_ptr;
  CUdeviceptr col_index;
  CUdeviceptr values;
  std::size_t row_index_bytes;
  std::size_t row_ptr_bytes;
  std::size_t col_index_bytes;
  std::size_t value_bytes;
  int rows;
  int columns;
  int nnz;
  std::atomic<unsigned int> references;
};

void* open_library(const char* environment_name,
                   const std::vector<const char*>& candidates) {
  const char* explicit_path = std::getenv(environment_name);
#ifdef _WIN32
  if (explicit_path != nullptr && std::strlen(explicit_path) > 0) {
    HMODULE handle = LoadLibraryExA(
        explicit_path,
        nullptr,
        LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    if (handle == nullptr) handle = LoadLibraryA(explicit_path);
    if (handle != nullptr) return reinterpret_cast<void*>(handle);
  }
  for (const char* candidate : candidates) {
    HMODULE handle = LoadLibraryA(candidate);
    if (handle != nullptr) return reinterpret_cast<void*>(handle);
  }
#else
  if (explicit_path != nullptr && std::strlen(explicit_path) > 0) {
    void* handle = dlopen(explicit_path, RTLD_NOW | RTLD_LOCAL);
    if (handle != nullptr) return handle;
  }
  for (const char* candidate : candidates) {
    void* handle = dlopen(candidate, RTLD_NOW | RTLD_LOCAL);
    if (handle != nullptr) return handle;
  }
#endif
  return nullptr;
}

void* lookup(void* library, const char* name) {
  if (library == nullptr) return nullptr;
#ifdef _WIN32
  return reinterpret_cast<void*>(GetProcAddress(
      reinterpret_cast<HMODULE>(library), name));
#else
  return dlsym(library, name);
#endif
}

template <typename T>
bool bind_symbol(T& target, void* library, const char* name) {
  target = reinterpret_cast<T>(lookup(library, name));
  return target != nullptr;
}

std::string cuda_error(CUresult status) {
  const char* name = nullptr;
  const char* description = nullptr;
  if (api.cuGetErrorName != nullptr) api.cuGetErrorName(status, &name);
  if (api.cuGetErrorString != nullptr) api.cuGetErrorString(status, &description);
  std::string result = "CUDA error " + std::to_string(status);
  if (name != nullptr) result += " (" + std::string(name) + ")";
  if (description != nullptr) result += ": " + std::string(description);
  return result;
}

std::string cublas_error(cublasStatus_t status) {
  return "cuBLAS error " + std::to_string(status);
}

std::string cusolver_error(cusolverStatus_t status) {
  return "cuSOLVER error " + std::to_string(status);
}

bool load_driver(std::string& reason) {
#ifdef __APPLE__
  reason = "Native CUDA is not supported on macOS.";
  return false;
#else
  if (api.driver == nullptr) {
#ifdef _WIN32
    api.driver = open_library("CUDAVERSE_CUDA_DRIVER_PATH", {"nvcuda.dll"});
#else
    api.driver = open_library("CUDAVERSE_CUDA_DRIVER_PATH",
                              {"libcuda.so.1", "libcuda.so"});
#endif
  }
  if (api.driver == nullptr) {
    reason = "The NVIDIA CUDA Driver library could not be loaded.";
    return false;
  }
  bool bound =
      bind_symbol(api.cuInit, api.driver, "cuInit") &&
      bind_symbol(api.cuDriverGetVersion, api.driver, "cuDriverGetVersion") &&
      bind_symbol(api.cuDeviceGetCount, api.driver, "cuDeviceGetCount") &&
      bind_symbol(api.cuDeviceGet, api.driver, "cuDeviceGet") &&
      bind_symbol(api.cuDevicePrimaryCtxRetain, api.driver,
                  "cuDevicePrimaryCtxRetain") &&
      bind_symbol(api.cuCtxSetCurrent, api.driver, "cuCtxSetCurrent") &&
      bind_symbol(api.cuMemAlloc, api.driver, "cuMemAlloc_v2") &&
      bind_symbol(api.cuMemFree, api.driver, "cuMemFree_v2") &&
      bind_symbol(api.cuMemcpyHtoD, api.driver, "cuMemcpyHtoD_v2") &&
      bind_symbol(api.cuMemcpyDtoH, api.driver, "cuMemcpyDtoH_v2") &&
      bind_symbol(api.cuMemcpyDtoD, api.driver, "cuMemcpyDtoD_v2") &&
      bind_symbol(api.cuCtxSynchronize, api.driver, "cuCtxSynchronize") &&
      bind_symbol(api.cuMemGetInfo, api.driver, "cuMemGetInfo_v2") &&
      bind_symbol(api.cuGetErrorName, api.driver, "cuGetErrorName") &&
      bind_symbol(api.cuGetErrorString, api.driver, "cuGetErrorString") &&
      bind_symbol(api.cuModuleLoad, api.driver, "cuModuleLoad") &&
      bind_symbol(api.cuModuleUnload, api.driver, "cuModuleUnload") &&
      bind_symbol(api.cuModuleGetFunction, api.driver,
                  "cuModuleGetFunction") &&
      bind_symbol(api.cuLaunchKernel, api.driver, "cuLaunchKernel");
  if (!bound) {
    reason = "The CUDA Driver library is missing required symbols.";
    return false;
  }
  CUresult status = api.cuInit(0);
  if (status != CUDA_SUCCESS) {
    reason = cuda_error(status);
    return false;
  }
  status = api.cuDriverGetVersion(&api.driver_version);
  if (status != CUDA_SUCCESS) {
    reason = cuda_error(status);
    return false;
  }
  status = api.cuDeviceGetCount(&api.device_count);
  if (status != CUDA_SUCCESS) {
    reason = cuda_error(status);
    return false;
  }
  if (api.device_count < 1) {
    reason = "The CUDA driver reported no devices.";
    return false;
  }
  if (api.context == nullptr) {
    CUdevice device = 0;
    status = api.cuDeviceGet(&device, 0);
    if (status == CUDA_SUCCESS) {
      status = api.cuDevicePrimaryCtxRetain(&api.context, device);
    }
    if (status != CUDA_SUCCESS) {
      reason = cuda_error(status);
      return false;
    }
  }
  status = api.cuCtxSetCurrent(api.context);
  if (status != CUDA_SUCCESS) {
    reason = cuda_error(status);
    return false;
  }
  return true;
#endif
}

bool load_cublas(std::string& reason) {
  if (api.cublas == nullptr) {
#ifdef _WIN32
    api.cublas = open_library("CUDAVERSE_CUBLAS_PATH", {"cublas64_12.dll"});
#else
    api.cublas = open_library("CUDAVERSE_CUBLAS_PATH",
                              {"libcublas.so.12", "libcublas.so"});
#endif
  }
  if (api.cublas == nullptr) {
    reason = "The NVIDIA cuBLAS 12 library could not be loaded.";
    return false;
  }
  bool bound = bind_symbol(api.cublasCreate, api.cublas, "cublasCreate_v2") &&
               bind_symbol(api.cublasDgemm, api.cublas, "cublasDgemm_v2") &&
               bind_symbol(api.cublasSgemm, api.cublas, "cublasSgemm_v2");
  if (!bound) {
    reason = "The cuBLAS library is missing required symbols.";
    return false;
  }
  if (api.cublas_handle == nullptr) {
    cublasStatus_t status = api.cublasCreate(&api.cublas_handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
      reason = cublas_error(status);
      return false;
    }
  }
  return true;
}

bool ensure_backend(std::string& reason) {
  return load_driver(reason) && load_cublas(reason);
}

void require_backend() {
  std::string reason;
  if (!ensure_backend(reason)) Rf_error("%s", reason.c_str());
}

void check_cuda(CUresult status, const char* operation) {
  if (status != CUDA_SUCCESS) {
    std::string message = std::string(operation) + ": " + cuda_error(status);
    Rf_error("%s", message.c_str());
  }
}

void check_cublas(cublasStatus_t status, const char* operation) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::string message = std::string(operation) + ": " + cublas_error(status);
    Rf_error("%s", message.c_str());
  }
}

bool load_cusolver(std::string& reason) {
  if (!load_driver(reason)) return false;
  if (api.cusolver == nullptr) {
#ifdef _WIN32
    api.cusolver = open_library(
        "CUDAVERSE_CUSOLVER_PATH", {"cusolver64_11.dll"});
#elif defined(__APPLE__)
    reason = "Native CUDA is not supported on macOS.";
    return false;
#else
    api.cusolver = open_library(
        "CUDAVERSE_CUSOLVER_PATH",
        {"libcusolver.so.11", "libcusolver.so"});
#endif
  }
  if (api.cusolver == nullptr) {
    reason = "The NVIDIA cuSOLVER 11 library could not be loaded.";
    return false;
  }
  bool bound =
      bind_symbol(api.cusolverDnCreate, api.cusolver,
                  "cusolverDnCreate") &&
      bind_symbol(api.cusolverDnDgesvd_bufferSize, api.cusolver,
                  "cusolverDnDgesvd_bufferSize") &&
      bind_symbol(api.cusolverDnDgesvd, api.cusolver,
                  "cusolverDnDgesvd");
  if (!bound) {
    reason = "The cuSOLVER library is missing required SVD symbols.";
    return false;
  }
  if (api.cusolver_handle == nullptr) {
    cusolverStatus_t status = api.cusolverDnCreate(&api.cusolver_handle);
    if (status != CUSOLVER_STATUS_SUCCESS) {
      reason = cusolver_error(status);
      return false;
    }
  }
  return true;
}

void require_kernels() {
  if (api.kernels == nullptr) {
    Rf_error("The cudaverse PTX module has not been loaded.");
  }
}

CUfunction get_kernel(const char* name) {
  require_kernels();
  CUfunction function = nullptr;
  check_cuda(api.cuModuleGetFunction(&function, api.kernels, name),
             "cuModuleGetFunction");
  return function;
}

CUresult launch_1d(CUfunction function, std::size_t elements,
                   void** parameters) {
  if (elements == 0) return CUDA_SUCCESS;
  constexpr unsigned int threads = 256;
  unsigned long long blocks =
      (static_cast<unsigned long long>(elements) + threads - 1) / threads;
  if (blocks > std::numeric_limits<unsigned int>::max()) {
    Rf_error("Native CUDA kernel launch is too large.");
  }
  return api.cuLaunchKernel(
      function, static_cast<unsigned int>(blocks), 1, 1,
      threads, 1, 1, 0, nullptr, parameters, nullptr);
}

SEXP buffer_tag() {
  static SEXP tag = R_NilValue;
  if (tag == R_NilValue) tag = Rf_install("cudaverse_native_buffer");
  return tag;
}

SEXP sparse_buffer_tag() {
  static SEXP tag = R_NilValue;
  if (tag == R_NilValue) tag = Rf_install("cudaverse_native_sparse_buffer");
  return tag;
}

SharedBuffer* get_buffer(SEXP pointer) {
  if (TYPEOF(pointer) != EXTPTRSXP ||
      R_ExternalPtrTag(pointer) != buffer_tag()) {
    Rf_error("Expected a live cudaverse native external pointer.");
  }
  auto* buffer = static_cast<SharedBuffer*>(R_ExternalPtrAddr(pointer));
  if (buffer == nullptr) Rf_error("The native CUDA allocation was released.");
  return buffer;
}

void destroy_buffer(SharedBuffer* buffer) {
  if (buffer == nullptr) return;
  SharedAllocation* allocation = buffer->allocation;
  if (allocation != nullptr && allocation->references.fetch_sub(1) == 1) {
    if (allocation->pointer != 0 && api.cuMemFree != nullptr) {
      if (api.context != nullptr && api.cuCtxSetCurrent != nullptr) {
        api.cuCtxSetCurrent(api.context);
      }
      release_tracked_device_memory(allocation->pointer, allocation->bytes);
    }
    delete allocation;
  }
  delete buffer;
}

void release_reference(SEXP pointer) {
  auto* buffer = static_cast<SharedBuffer*>(R_ExternalPtrAddr(pointer));
  if (buffer == nullptr) return;
  R_ClearExternalPtr(pointer);
  if (buffer->references.fetch_sub(1) == 1) destroy_buffer(buffer);
}

void buffer_finalizer(SEXP pointer) {
  release_reference(pointer);
}

SEXP make_pointer(SharedBuffer* buffer, bool add_reference) {
  if (add_reference) buffer->references.fetch_add(1);
  SEXP pointer = PROTECT(R_MakeExternalPtr(buffer, buffer_tag(), R_NilValue));
  R_RegisterCFinalizerEx(pointer, buffer_finalizer, static_cast<Rboolean>(1));
  UNPROTECT(1);
  return pointer;
}

SharedSparseBuffer* get_sparse_buffer(SEXP pointer) {
  if (TYPEOF(pointer) != EXTPTRSXP ||
      R_ExternalPtrTag(pointer) != sparse_buffer_tag()) {
    Rf_error("Expected a live cudaverse native sparse external pointer.");
  }
  auto* buffer = static_cast<SharedSparseBuffer*>(R_ExternalPtrAddr(pointer));
  if (buffer == nullptr) {
    Rf_error("The native CUDA sparse allocation was released.");
  }
  return buffer;
}

void release_sparse_reference(SEXP pointer) {
  auto* buffer = static_cast<SharedSparseBuffer*>(R_ExternalPtrAddr(pointer));
  if (buffer == nullptr) return;
  R_ClearExternalPtr(pointer);
  if (buffer->references.fetch_sub(1) == 1) {
    if (api.context != nullptr && api.cuCtxSetCurrent != nullptr) {
      api.cuCtxSetCurrent(api.context);
    }
    release_tracked_device_memory(buffer->row_index,
                                  buffer->row_index_bytes);
    release_tracked_device_memory(buffer->row_ptr, buffer->row_ptr_bytes);
    release_tracked_device_memory(buffer->col_index,
                                  buffer->col_index_bytes);
    release_tracked_device_memory(buffer->values, buffer->value_bytes);
    delete buffer;
  }
}

void sparse_buffer_finalizer(SEXP pointer) {
  release_sparse_reference(pointer);
}

SEXP make_sparse_pointer(SharedSparseBuffer* buffer, bool add_reference) {
  if (add_reference) buffer->references.fetch_add(1);
  SEXP pointer = PROTECT(
      R_MakeExternalPtr(buffer, sparse_buffer_tag(), R_NilValue));
  R_RegisterCFinalizerEx(
      pointer, sparse_buffer_finalizer, static_cast<Rboolean>(1));
  UNPROTECT(1);
  return pointer;
}

DType parse_dtype(SEXP dtype) {
  if (TYPEOF(dtype) != STRSXP || XLENGTH(dtype) != 1) {
    Rf_error("`dtype` must be one character string.");
  }
  std::string value = CHAR(STRING_ELT(dtype, 0));
  if (value == "float64") return DType::Float64;
  if (value == "float32") return DType::Float32;
  if (value == "integer") return DType::Integer;
  Rf_error("Unsupported native CUDA dtype `%s`.", value.c_str());
  return DType::Float64;
}

std::size_t dtype_size(DType dtype) {
  switch (dtype) {
    case DType::Float64: return sizeof(double);
    case DType::Float32: return sizeof(float);
    case DType::Integer: return sizeof(int);
  }
  return 0;
}

std::vector<int> parse_shape(SEXP shape, std::size_t& elements) {
  if (TYPEOF(shape) != INTSXP || XLENGTH(shape) < 1) {
    Rf_error("`shape` must be a non-empty integer vector.");
  }
  std::vector<int> result(static_cast<std::size_t>(XLENGTH(shape)));
  elements = 1;
  for (R_xlen_t index = 0; index < XLENGTH(shape); ++index) {
    int extent = INTEGER(shape)[index];
    if (extent < 1) Rf_error("Native CUDA shapes must be positive.");
    if (elements > std::numeric_limits<std::size_t>::max() /
                       static_cast<std::size_t>(extent)) {
      Rf_error("Native CUDA shape is too large.");
    }
    elements *= static_cast<std::size_t>(extent);
    result[static_cast<std::size_t>(index)] = extent;
  }
  return result;
}

std::vector<int> parse_indices(SEXP indices, std::size_t upper_bound,
                               const char* argument) {
  if (TYPEOF(indices) != INTSXP || XLENGTH(indices) < 1) {
    Rf_error("`%s` must be a non-empty integer vector.", argument);
  }
  std::vector<int> result(static_cast<std::size_t>(XLENGTH(indices)));
  for (R_xlen_t index = 0; index < XLENGTH(indices); ++index) {
    int value = INTEGER(indices)[index];
    if (value == NA_INTEGER || value < 1 ||
        static_cast<std::size_t>(value) > upper_bound) {
      Rf_error("`%s` contains an out-of-range or missing index.", argument);
    }
    result[static_cast<std::size_t>(index)] = value - 1;
  }
  return result;
}

SharedBuffer* buffer_from_device(CUdeviceptr pointer, std::size_t bytes,
                                 std::size_t elements, DType dtype,
                                 const std::vector<int>& shape) {
  SharedAllocation* allocation = nullptr;
  try {
    allocation = new SharedAllocation{pointer, bytes, 1};
    return new SharedBuffer{
        pointer, bytes, elements, dtype, shape, 1, allocation};
  } catch (...) {
    delete allocation;
    release_tracked_device_memory(pointer, bytes);
    throw;
  }
}

SharedBuffer* buffer_view(SharedBuffer* source,
                          const std::vector<int>& shape) {
  auto* view = new SharedBuffer{
      source->pointer, source->bytes, source->elements, source->dtype,
      shape, 1, source->allocation};
  source->allocation->references.fetch_add(1);
  return view;
}

SharedBuffer* contiguous_buffer_view(SharedBuffer* source,
                                     const std::vector<int>& shape,
                                     std::size_t offset_elements,
                                     std::size_t elements) {
  if (offset_elements > source->elements ||
      elements > source->elements - offset_elements) {
    Rf_error("Native CUDA view is outside its source allocation.");
  }
  std::size_t element_bytes = dtype_size(source->dtype);
  std::size_t offset_bytes = offset_elements * element_bytes;
  std::size_t view_bytes = elements * element_bytes;
  auto* view = new SharedBuffer{
      source->pointer + static_cast<CUdeviceptr>(offset_bytes),
      view_bytes, elements, source->dtype, shape, 1, source->allocation};
  source->allocation->references.fetch_add(1);
  return view;
}

SharedBuffer* allocate_buffer(DType dtype, const std::vector<int>& shape,
                              std::size_t elements) {
  std::size_t bytes = elements * dtype_size(dtype);
  CUdeviceptr pointer = 0;
  CUresult status = api.cuMemAlloc(&pointer, bytes);
  if (status != CUDA_SUCCESS) {
    check_cuda(status, "cuMemAlloc");
  }
  track_device_allocation(bytes);
  return buffer_from_device(pointer, bytes, elements, dtype, shape);
}

class DeviceMemory {
 public:
  DeviceMemory() = default;
  explicit DeviceMemory(std::size_t bytes) : bytes_(bytes) {
    if (bytes_ == 0) return;
    CUresult status = api.cuMemAlloc(&pointer_, bytes_);
    if (status != CUDA_SUCCESS) {
      throw std::runtime_error("cuMemAlloc: " + cuda_error(status));
    }
    track_device_allocation(bytes_);
  }
  ~DeviceMemory() {
    release_tracked_device_memory(pointer_, bytes_);
  }
  DeviceMemory(const DeviceMemory&) = delete;
  DeviceMemory& operator=(const DeviceMemory&) = delete;
  DeviceMemory(DeviceMemory&& other) noexcept
      : pointer_(other.pointer_), bytes_(other.bytes_) {
    other.pointer_ = 0;
    other.bytes_ = 0;
  }
  DeviceMemory& operator=(DeviceMemory&& other) noexcept {
    if (this != &other) {
      release_tracked_device_memory(pointer_, bytes_);
      pointer_ = other.pointer_;
      bytes_ = other.bytes_;
      other.pointer_ = 0;
      other.bytes_ = 0;
    }
    return *this;
  }
  CUdeviceptr pointer() const { return pointer_; }
  std::size_t bytes() const { return bytes_; }
  CUdeviceptr release() {
    CUdeviceptr result = pointer_;
    pointer_ = 0;
    bytes_ = 0;
    return result;
  }

 private:
  CUdeviceptr pointer_ = 0;
  std::size_t bytes_ = 0;
};

void cuda_or_throw(CUresult status, const char* operation) {
  if (status != CUDA_SUCCESS) {
    throw std::runtime_error(
        std::string(operation) + ": " + cuda_error(status));
  }
}

void cusolver_or_throw(cusolverStatus_t status, const char* operation) {
  if (status != CUSOLVER_STATUS_SUCCESS) {
    throw std::runtime_error(
        std::string(operation) + ": " + cusolver_error(status));
  }
}

void launch_or_throw(CUfunction function, const char* name,
                     std::size_t elements, void** parameters) {
  cuda_or_throw(launch_1d(function, elements, parameters), name);
}

DeviceMemory copy_host_to_device(const void* source, std::size_t bytes,
                                 const char* operation) {
  DeviceMemory result(bytes);
  if (bytes > 0) {
    cuda_or_throw(api.cuMemcpyHtoD(result.pointer(), source, bytes), operation);
  }
  return result;
}

DeviceMemory copy_device_to_device(CUdeviceptr source, std::size_t bytes,
                                   const char* operation) {
  DeviceMemory result(bytes);
  if (bytes > 0) {
    cuda_or_throw(api.cuMemcpyDtoD(result.pointer(), source, bytes), operation);
  }
  return result;
}

SharedBuffer* shared_buffer_from_device(DeviceMemory&& memory, DType dtype,
                                         const std::vector<int>& shape,
                                         std::size_t elements) {
  std::size_t bytes = memory.bytes();
  return buffer_from_device(
      memory.release(), bytes, elements, dtype, shape);
}

SharedSparseBuffer* shared_sparse_from_devices(
    DeviceMemory&& row_index, DeviceMemory&& row_ptr,
    DeviceMemory&& col_index, DeviceMemory&& values,
    int rows, int columns, int nnz) {
  std::size_t row_index_bytes = row_index.bytes();
  std::size_t row_ptr_bytes = row_ptr.bytes();
  std::size_t col_index_bytes = col_index.bytes();
  std::size_t value_bytes = values.bytes();
  return new SharedSparseBuffer{
      row_index.release(), row_ptr.release(), col_index.release(),
      values.release(), row_index_bytes, row_ptr_bytes, col_index_bytes,
      value_bytes, rows, columns, nnz, 1};
}

struct DeviceSvd {
  int rank = 0;
  DeviceMemory values;
  DeviceMemory left;
  DeviceMemory right;
};

DeviceSvd run_device_svd(CUdeviceptr input, int rows, int columns) {
  std::string reason;
  if (!load_cusolver(reason)) throw std::runtime_error(reason);
  require_kernels();
  CUfunction transpose = get_kernel("cudaverse_transpose_f64");

  bool transposed = rows < columns;
  int solver_rows = transposed ? columns : rows;
  int solver_columns = transposed ? rows : columns;
  int rank = solver_columns;
  std::size_t matrix_elements =
      static_cast<std::size_t>(rows) * columns;
  DeviceMemory matrix(matrix_elements * sizeof(double));
  if (transposed) {
    CUdeviceptr output = matrix.pointer();
    void* parameters[] = {&input, &output, &rows, &columns};
    launch_or_throw(transpose, "cudaverse_transpose_f64",
                    matrix_elements, parameters);
  } else {
    cuda_or_throw(api.cuMemcpyDtoD(
                      matrix.pointer(), input,
                      matrix_elements * sizeof(double)),
                  "cuMemcpyDtoD");
  }

  int workspace_elements = 0;
  cusolver_or_throw(
      api.cusolverDnDgesvd_bufferSize(
          api.cusolver_handle, solver_rows, solver_columns,
          &workspace_elements),
      "cusolverDnDgesvd_bufferSize");
  if (workspace_elements < 1) {
    throw std::runtime_error("cuSOLVER returned an invalid SVD workspace.");
  }

  DeviceMemory singular_values(static_cast<std::size_t>(rank) *
                               sizeof(double));
  DeviceMemory solver_left(
      static_cast<std::size_t>(solver_rows) * rank * sizeof(double));
  DeviceMemory solver_right_transposed(
      static_cast<std::size_t>(rank) * solver_columns * sizeof(double));
  DeviceMemory workspace(
      static_cast<std::size_t>(workspace_elements) * sizeof(double));
  DeviceMemory rwork(
      static_cast<std::size_t>(rank > 1 ? rank - 1 : 1) * sizeof(double));
  DeviceMemory device_info(sizeof(int));

  cusolver_or_throw(
      api.cusolverDnDgesvd(
          api.cusolver_handle, static_cast<signed char>('S'),
          static_cast<signed char>('S'), solver_rows, solver_columns,
          reinterpret_cast<double*>(matrix.pointer()), solver_rows,
          reinterpret_cast<double*>(singular_values.pointer()),
          reinterpret_cast<double*>(solver_left.pointer()), solver_rows,
          reinterpret_cast<double*>(solver_right_transposed.pointer()), rank,
          reinterpret_cast<double*>(workspace.pointer()), workspace_elements,
          reinterpret_cast<double*>(rwork.pointer()),
          reinterpret_cast<int*>(device_info.pointer())),
      "cusolverDnDgesvd");
  int info = 0;
  cuda_or_throw(api.cuMemcpyDtoH(&info, device_info.pointer(), sizeof(int)),
                "cuMemcpyDtoH(SVD info)");
  if (info != 0) {
    throw std::runtime_error(
        "cuSOLVER SVD failed to converge (info=" +
        std::to_string(info) + ").");
  }

  DeviceMemory original_left;
  DeviceMemory original_right;
  if (!transposed) {
    original_left = std::move(solver_left);
    original_right = DeviceMemory(
        static_cast<std::size_t>(columns) * rank * sizeof(double));
    CUdeviceptr source = solver_right_transposed.pointer();
    CUdeviceptr destination = original_right.pointer();
    int transpose_rows = rank;
    int transpose_columns = columns;
    void* parameters[] = {
        &source, &destination, &transpose_rows, &transpose_columns};
    launch_or_throw(
        transpose, "cudaverse_transpose_f64",
        static_cast<std::size_t>(rank) * columns, parameters);
  } else {
    original_right = std::move(solver_left);
    original_left = DeviceMemory(
        static_cast<std::size_t>(rows) * rank * sizeof(double));
    CUdeviceptr source = solver_right_transposed.pointer();
    CUdeviceptr destination = original_left.pointer();
    int transpose_rows = rank;
    int transpose_columns = rows;
    void* parameters[] = {
        &source, &destination, &transpose_rows, &transpose_columns};
    launch_or_throw(
        transpose, "cudaverse_transpose_f64",
        static_cast<std::size_t>(rank) * rows, parameters);
  }

  DeviceSvd result;
  result.rank = rank;
  result.values = std::move(singular_values);
  result.left = std::move(original_left);
  result.right = std::move(original_right);
  return result;
}

std::vector<double> copy_double_to_host(CUdeviceptr pointer,
                                        std::size_t elements,
                                        const char* operation) {
  std::vector<double> result(elements);
  cuda_or_throw(api.cuMemcpyDtoH(
                    result.data(), pointer, elements * sizeof(double)),
                operation);
  return result;
}

SEXP numeric_vector(const std::vector<double>& values) {
  SEXP result = PROTECT(Rf_allocVector(REALSXP, values.size()));
  if (!values.empty()) {
    std::memcpy(REAL(result), values.data(), values.size() * sizeof(double));
  }
  UNPROTECT(1);
  return result;
}

void check_user_interrupt(void*) {
  R_CheckUserInterrupt();
}

void interrupt_or_throw() {
  if (!R_ToplevelExec(check_user_interrupt, nullptr)) {
    throw std::runtime_error("Native CUDA k-means was interrupted.");
  }
}

DeviceMemory row_norms_squared(CUdeviceptr input, int rows, int columns) {
  CUfunction function = get_kernel("cudaverse_row_norms_squared_f64");
  DeviceMemory output(static_cast<std::size_t>(rows) * sizeof(double));
  CUdeviceptr output_pointer = output.pointer();
  void* parameters[] = {&input, &output_pointer, &rows, &columns};
  launch_or_throw(function, "cudaverse_row_norms_squared_f64", rows,
                  parameters);
  return output;
}

DeviceMemory distance_device(CUdeviceptr query, int query_rows,
                             CUdeviceptr reference, int reference_rows,
                             int columns, int metric, int query_offset,
                             int self, CUdeviceptr cached_reference_norms = 0) {
  CUfunction finalize = get_kernel("cudaverse_distance_from_gram_f64");
  DeviceMemory query_norms;
  DeviceMemory reference_norms;
  CUdeviceptr query_norm_pointer = 0;
  CUdeviceptr reference_norm_pointer = cached_reference_norms;
  if (metric == 0) {
    query_norms = row_norms_squared(query, query_rows, columns);
    query_norm_pointer = query_norms.pointer();
    if (reference_norm_pointer == 0) {
      reference_norms = row_norms_squared(
          reference, reference_rows, columns);
      reference_norm_pointer = reference_norms.pointer();
    }
  }

  std::size_t output_elements =
      static_cast<std::size_t>(query_rows) * reference_rows;
  DeviceMemory gram(output_elements * sizeof(double));
  const double alpha = 1.0;
  const double beta = 0.0;
  cublasStatus_t blas_status = api.cublasDgemm(
      api.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
      query_rows, reference_rows, columns, &alpha,
      reinterpret_cast<const double*>(query), query_rows,
      reinterpret_cast<const double*>(reference), reference_rows, &beta,
      reinterpret_cast<double*>(gram.pointer()), query_rows);
  if (blas_status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error("cublasDgemm(distance): " +
                             cublas_error(blas_status));
  }

  DeviceMemory output(output_elements * sizeof(double));
  CUdeviceptr gram_pointer = gram.pointer();
  CUdeviceptr output_pointer = output.pointer();
  void* parameters[] = {
      &gram_pointer, &query, &reference, &query_norm_pointer,
      &reference_norm_pointer, &output_pointer, &query_rows,
      &reference_rows, &columns, &metric, &query_offset, &self};
  launch_or_throw(finalize, "cudaverse_distance_from_gram_f64",
                  output_elements, parameters);
  return output;
}

DeviceMemory sparse_sums_device(SharedSparseBuffer* sparse, int margin) {
  int groups = margin == 0 ? sparse->rows : sparse->columns;
  DeviceMemory output(static_cast<std::size_t>(groups) * sizeof(double));
  CUdeviceptr output_pointer = output.pointer();
  if (margin == 0) {
    CUfunction function = get_kernel("cudaverse_sparse_row_sums_f64");
    void* parameters[] = {
        &sparse->row_ptr, &sparse->values, &output_pointer, &sparse->rows};
    launch_or_throw(function, "cudaverse_sparse_row_sums_f64",
                    sparse->rows, parameters);
  } else {
    CUfunction fill = get_kernel("cudaverse_fill_f64");
    double zero = 0.0;
    unsigned long long elements = static_cast<unsigned long long>(groups);
    void* fill_parameters[] = {&output_pointer, &zero, &elements};
    launch_or_throw(fill, "cudaverse_fill_f64", groups, fill_parameters);
    CUfunction function = get_kernel("cudaverse_sparse_col_sums_f64");
    void* parameters[] = {
        &sparse->col_index, &sparse->values, &output_pointer, &sparse->nnz};
    launch_or_throw(function, "cudaverse_sparse_col_sums_f64",
                    sparse->nnz, parameters);
  }
  return output;
}

DeviceMemory sparse_to_dense_device(SharedSparseBuffer* sparse) {
  std::size_t elements =
      static_cast<std::size_t>(sparse->rows) * sparse->columns;
  DeviceMemory output(elements * sizeof(double));
  CUdeviceptr output_pointer = output.pointer();
  CUfunction fill = get_kernel("cudaverse_fill_f64");
  double zero = 0.0;
  unsigned long long fill_elements =
      static_cast<unsigned long long>(elements);
  void* fill_parameters[] = {&output_pointer, &zero, &fill_elements};
  launch_or_throw(fill, "cudaverse_fill_f64", elements, fill_parameters);
  CUfunction scatter = get_kernel("cudaverse_csr_to_dense_f64");
  void* scatter_parameters[] = {
      &sparse->row_index, &sparse->col_index, &sparse->values,
      &output_pointer, &sparse->rows, &sparse->nnz};
  launch_or_throw(scatter, "cudaverse_csr_to_dense_f64",
                  sparse->nnz, scatter_parameters);
  return output;
}

SEXP scalar_string_or_na(const std::string& value) {
  return value.empty() ? Rf_ScalarString(NA_STRING) : Rf_mkString(value.c_str());
}

SEXP named_list(const std::vector<const char*>& names) {
  SEXP result = PROTECT(Rf_allocVector(VECSXP, names.size()));
  SEXP result_names = PROTECT(Rf_allocVector(STRSXP, names.size()));
  for (std::size_t index = 0; index < names.size(); ++index) {
    SET_STRING_ELT(result_names, index, Rf_mkChar(names[index]));
  }
  Rf_setAttrib(result, R_NamesSymbol, result_names);
  UNPROTECT(2);
  return result;
}

}  // namespace

extern "C" SEXP C_cudaverse_cuda_load_kernels(SEXP path_sexp) {
  if (TYPEOF(path_sexp) != STRSXP || XLENGTH(path_sexp) != 1 ||
      STRING_ELT(path_sexp, 0) == NA_STRING) {
    Rf_error("`path` must be one PTX file path.");
  }
  std::string reason;
  if (!load_driver(reason)) Rf_error("%s", reason.c_str());
  if (api.kernels != nullptr) return Rf_ScalarLogical(1);
  const char* path = Rf_translateCharUTF8(STRING_ELT(path_sexp, 0));
  check_cuda(api.cuModuleLoad(&api.kernels, path), "cuModuleLoad");
  return Rf_ScalarLogical(1);
}

extern "C" SEXP C_cudaverse_cuda_diagnostics() {
  std::string reason;
  bool available = ensure_backend(reason);
  std::string solver_reason;
  bool solver_available = available && load_cusolver(solver_reason);
  SEXP result = PROTECT(named_list({
      "installed", "available", "device_count", "version", "reason",
      "detection_error", "driver_version", "cublas_loaded",
      "cusolver_loaded", "cusolver_error"}));
  SET_VECTOR_ELT(result, 0, Rf_ScalarLogical(1));
  SET_VECTOR_ELT(result, 1, Rf_ScalarLogical(available));
  SET_VECTOR_ELT(result, 2, Rf_ScalarInteger(api.device_count));
  std::string version = api.driver_version > 0
      ? std::to_string(api.driver_version)
      : std::string();
  SET_VECTOR_ELT(result, 3, scalar_string_or_na(version));
  SET_VECTOR_ELT(result, 4, Rf_mkString(available ? "cuda_available" :
#ifdef __APPLE__
                                       "unsupported_platform"));
#else
                                       (api.driver == nullptr
                                            ? "driver_unavailable"
                                            : "backend_error")));
#endif
  SET_VECTOR_ELT(result, 5,
                 available ? R_NilValue : Rf_mkString(reason.c_str()));
  SET_VECTOR_ELT(result, 6, Rf_ScalarInteger(api.driver_version));
  SET_VECTOR_ELT(result, 7, Rf_ScalarLogical(api.cublas_handle != nullptr));
  SET_VECTOR_ELT(result, 8, Rf_ScalarLogical(solver_available));
  SET_VECTOR_ELT(result, 9, solver_available || !available
      ? R_NilValue
      : Rf_mkString(solver_reason.c_str()));
  UNPROTECT(1);
  return result;
}

extern "C" SEXP C_cudaverse_cuda_from_host(SEXP x, SEXP dtype_sexp,
                                             SEXP shape_sexp) {
  require_backend();
  DType dtype = parse_dtype(dtype_sexp);
  std::size_t elements = 0;
  std::vector<int> shape = parse_shape(shape_sexp, elements);
  if (static_cast<std::size_t>(XLENGTH(x)) != elements) {
    Rf_error("Host data length does not match `shape`.");
  }
  SharedBuffer* buffer = allocate_buffer(dtype, shape, elements);
  CUresult status = CUDA_SUCCESS;
  if (dtype == DType::Float64) {
    SEXP values = PROTECT(Rf_coerceVector(x, REALSXP));
    status = api.cuMemcpyHtoD(buffer->pointer, REAL(values), buffer->bytes);
    UNPROTECT(1);
  } else if (dtype == DType::Float32) {
    SEXP values = PROTECT(Rf_coerceVector(x, REALSXP));
    std::vector<float> converted(elements);
    for (std::size_t index = 0; index < elements; ++index) {
      converted[index] = static_cast<float>(REAL(values)[index]);
    }
    status = api.cuMemcpyHtoD(buffer->pointer, converted.data(), buffer->bytes);
    UNPROTECT(1);
  } else {
    SEXP values = PROTECT(Rf_coerceVector(x, INTSXP));
    status = api.cuMemcpyHtoD(buffer->pointer, INTEGER(values), buffer->bytes);
    UNPROTECT(1);
  }
  if (status != CUDA_SUCCESS) {
    destroy_buffer(buffer);
    check_cuda(status, "cuMemcpyHtoD");
  }
  return make_pointer(buffer, false);
}

extern "C" SEXP C_cudaverse_cuda_to_host(SEXP pointer) {
  require_backend();
  SharedBuffer* buffer = get_buffer(pointer);
  if (buffer->dtype == DType::Float64) {
    SEXP result = PROTECT(Rf_allocVector(REALSXP, buffer->elements));
    check_cuda(api.cuMemcpyDtoH(REAL(result), buffer->pointer, buffer->bytes),
               "cuMemcpyDtoH");
    UNPROTECT(1);
    return result;
  }
  if (buffer->dtype == DType::Float32) {
    std::vector<float> values(buffer->elements);
    check_cuda(api.cuMemcpyDtoH(values.data(), buffer->pointer, buffer->bytes),
               "cuMemcpyDtoH");
    SEXP result = PROTECT(Rf_allocVector(REALSXP, buffer->elements));
    for (std::size_t index = 0; index < buffer->elements; ++index) {
      REAL(result)[index] = static_cast<double>(values[index]);
    }
    UNPROTECT(1);
    return result;
  }
  SEXP result = PROTECT(Rf_allocVector(INTSXP, buffer->elements));
  check_cuda(api.cuMemcpyDtoH(INTEGER(result), buffer->pointer, buffer->bytes),
             "cuMemcpyDtoH");
  UNPROTECT(1);
  return result;
}

extern "C" SEXP C_cudaverse_cuda_cast(SEXP pointer, SEXP dtype_sexp) {
  require_backend();
  require_kernels();
  SharedBuffer* input = get_buffer(pointer);
  DType target = parse_dtype(dtype_sexp);
  if (input->dtype == target) return make_pointer(input, true);

  const char* kernel = nullptr;
  if (input->dtype == DType::Integer && target == DType::Float64) {
    kernel = "cudaverse_cast_i32_f64";
  } else if (input->dtype == DType::Float32 && target == DType::Float64) {
    kernel = "cudaverse_cast_f32_f64";
  } else if (input->dtype == DType::Float64 && target == DType::Float32) {
    kernel = "cudaverse_cast_f64_f32";
  } else if (input->dtype == DType::Integer && target == DType::Float32) {
    kernel = "cudaverse_cast_i32_f32";
  } else if (input->dtype == DType::Float64 && target == DType::Integer) {
    kernel = "cudaverse_cast_f64_i32";
  } else if (input->dtype == DType::Float32 && target == DType::Integer) {
    kernel = "cudaverse_cast_f32_i32";
  }
  if (kernel == nullptr) Rf_error("Unsupported native CUDA cast.");

  R_CheckUserInterrupt();
  CUfunction function = get_kernel(kernel);
  SharedBuffer* output = allocate_buffer(
      target, input->shape, input->elements);
  CUdeviceptr input_pointer = input->pointer;
  CUdeviceptr output_pointer = output->pointer;
  unsigned long long elements = input->elements;
  void* parameters[] = {&input_pointer, &output_pointer, &elements};
  CUresult status = launch_1d(function, input->elements, parameters);
  if (status != CUDA_SUCCESS) {
    destroy_buffer(output);
    check_cuda(status, kernel);
  }
  return make_pointer(output, false);
}

extern "C" SEXP C_cudaverse_cuda_reshape(SEXP pointer,
                                           SEXP shape_sexp) {
  require_backend();
  SharedBuffer* input = get_buffer(pointer);
  std::size_t elements = 0;
  std::vector<int> shape = parse_shape(shape_sexp, elements);
  if (elements != input->elements) {
    Rf_error("Native CUDA reshape must preserve the number of values.");
  }
  if (shape == input->shape) return make_pointer(input, true);
  return make_pointer(buffer_view(input, shape), false);
}

extern "C" SEXP C_cudaverse_cuda_gather(SEXP pointer, SEXP indices_sexp,
                                          SEXP shape_sexp) {
  require_backend();
  require_kernels();
  SharedBuffer* input = get_buffer(pointer);
  std::size_t output_elements = 0;
  std::vector<int> output_shape = parse_shape(shape_sexp, output_elements);
  std::vector<int> indices = parse_indices(
      indices_sexp, input->elements, "indices");
  if (indices.size() != output_elements) {
    Rf_error("Native CUDA gather indices do not match the output shape.");
  }

  bool contiguous = true;
  for (std::size_t index = 1; index < indices.size(); ++index) {
    if (static_cast<std::size_t>(indices[index]) !=
        static_cast<std::size_t>(indices[0]) + index) {
      contiguous = false;
      break;
    }
  }
  if (contiguous) {
    return make_pointer(
        contiguous_buffer_view(
            input, output_shape, static_cast<std::size_t>(indices[0]),
            output_elements),
        false);
  }

  const char* kernel = input->dtype == DType::Float64
      ? "cudaverse_gather_f64"
      : (input->dtype == DType::Float32
             ? "cudaverse_gather_f32"
             : "cudaverse_gather_i32");
  CUfunction function = get_kernel(kernel);
  R_CheckUserInterrupt();
  try {
    DeviceMemory device_indices = copy_host_to_device(
        indices.data(), indices.size() * sizeof(int),
        "cuMemcpyHtoD(gather indices)");
    DeviceMemory output(output_elements * dtype_size(input->dtype));
    CUdeviceptr input_values = input->pointer;
    CUdeviceptr index_values = device_indices.pointer();
    CUdeviceptr output_values = output.pointer();
    unsigned long long elements = output_elements;
    void* parameters[] = {
        &input_values, &index_values, &output_values, &elements};
    launch_or_throw(function, kernel, output_elements, parameters);
    return make_pointer(
        shared_buffer_from_device(
            std::move(output), input->dtype, output_shape, output_elements),
        false);
  } catch (const std::exception& exception) {
    Rf_error("%s", exception.what());
  }
  return R_NilValue;
}

extern "C" SEXP C_cudaverse_cuda_scatter(
    SEXP pointer, SEXP indices_sexp, SEXP replacement_pointer,
    SEXP replacement_indices_sexp) {
  require_backend();
  require_kernels();
  SharedBuffer* input = get_buffer(pointer);
  SharedBuffer* replacement = get_buffer(replacement_pointer);
  if (input->dtype != replacement->dtype) {
    Rf_error("Native CUDA replacement dtype must match the tensor dtype.");
  }
  std::vector<int> indices = parse_indices(
      indices_sexp, input->elements, "indices");
  std::vector<int> replacement_indices = parse_indices(
      replacement_indices_sexp, replacement->elements,
      "replacement_indices");
  if (indices.size() != replacement_indices.size()) {
    Rf_error("Native CUDA replacement index vectors must have equal length.");
  }

  const char* kernel = input->dtype == DType::Float64
      ? "cudaverse_scatter_f64"
      : (input->dtype == DType::Float32
             ? "cudaverse_scatter_f32"
             : "cudaverse_scatter_i32");
  CUfunction function = get_kernel(kernel);
  R_CheckUserInterrupt();
  try {
    DeviceMemory device_indices = copy_host_to_device(
        indices.data(), indices.size() * sizeof(int),
        "cuMemcpyHtoD(scatter indices)");
    DeviceMemory device_replacement_indices = copy_host_to_device(
        replacement_indices.data(),
        replacement_indices.size() * sizeof(int),
        "cuMemcpyHtoD(scatter replacement indices)");
    DeviceMemory output = copy_device_to_device(
        input->pointer, input->bytes, "cuMemcpyDtoD(scatter source)");

    CUdeviceptr output_values = output.pointer();
    CUdeviceptr index_values = device_indices.pointer();
    CUdeviceptr replacement_values = replacement->pointer;
    CUdeviceptr replacement_index_values =
        device_replacement_indices.pointer();
    unsigned long long elements = indices.size();
    void* parameters[] = {
        &output_values, &index_values, &replacement_values,
        &replacement_index_values, &elements};
    launch_or_throw(function, kernel, indices.size(), parameters);
    return make_pointer(
        shared_buffer_from_device(
            std::move(output), input->dtype, input->shape, input->elements),
        false);
  } catch (const std::exception& exception) {
    Rf_error("%s", exception.what());
  }
  return R_NilValue;
}

extern "C" SEXP C_cudaverse_cuda_broadcast(SEXP pointer,
                                             SEXP shape_sexp) {
  require_backend();
  require_kernels();
  SharedBuffer* input = get_buffer(pointer);
  std::size_t output_elements = 0;
  std::vector<int> target_shape = parse_shape(shape_sexp, output_elements);
  if (target_shape.size() > CUDAVERSE_MAX_RANK ||
      target_shape.size() < input->shape.size()) {
    Rf_error("Native CUDA broadcasting supports one to %d dimensions.",
             CUDAVERSE_MAX_RANK);
  }

  BroadcastMeta meta{};
  meta.rank = static_cast<int>(target_shape.size());
  std::size_t padding = target_shape.size() - input->shape.size();
  unsigned long long stride = 1;
  for (int dimension = 0; dimension < meta.rank; ++dimension) {
    int source_extent = dimension < static_cast<int>(padding)
        ? 1
        : input->shape[static_cast<std::size_t>(dimension) - padding];
    int target_extent = target_shape[dimension];
    if (source_extent != 1 && source_extent != target_extent) {
      Rf_error("Native CUDA tensor shapes are not broadcast-compatible.");
    }
    meta.source_shape[dimension] = source_extent;
    meta.target_shape[dimension] = target_extent;
    meta.source_stride[dimension] = stride;
    stride *= static_cast<unsigned long long>(source_extent);
  }

  const char* kernel = input->dtype == DType::Float64
      ? "cudaverse_broadcast_f64"
      : (input->dtype == DType::Float32
             ? "cudaverse_broadcast_f32"
             : "cudaverse_broadcast_i32");
  CUfunction function = get_kernel(kernel);
  SharedBuffer* output = allocate_buffer(
      input->dtype, target_shape, output_elements);
  CUdeviceptr input_pointer = input->pointer;
  CUdeviceptr output_pointer = output->pointer;
  unsigned long long kernel_elements = output_elements;
  void* parameters[] = {
      &input_pointer, &output_pointer, &meta, &kernel_elements};
  CUresult status = launch_1d(function, output_elements, parameters);
  if (status != CUDA_SUCCESS) {
    destroy_buffer(output);
    check_cuda(status, kernel);
  }
  return make_pointer(output, false);
}

extern "C" SEXP C_cudaverse_cuda_binary(SEXP left_pointer,
                                          SEXP right_pointer,
                                          SEXP operation_sexp) {
  require_backend();
  require_kernels();
  SharedBuffer* left = get_buffer(left_pointer);
  SharedBuffer* right = get_buffer(right_pointer);
  if (left->dtype != right->dtype || left->shape != right->shape) {
    Rf_error("Native CUDA binary operands must have matching dtype and shape.");
  }
  if (left->dtype == DType::Integer) {
    Rf_error("Native CUDA integer arithmetic must be promoted before execution.");
  }
  if (TYPEOF(operation_sexp) != STRSXP || XLENGTH(operation_sexp) != 1 ||
      STRING_ELT(operation_sexp, 0) == NA_STRING) {
    Rf_error("Native CUDA binary operation must be one character string.");
  }
  std::string operation_name = CHAR(STRING_ELT(operation_sexp, 0));
  int operation = operation_name == "+" ? 0
      : (operation_name == "-" ? 1
      : (operation_name == "*" ? 2
      : (operation_name == "/" ? 3
      : (operation_name == "^" ? 4 : -1))));
  if (operation < 0) Rf_error("Unsupported native CUDA binary operation.");

  const char* kernel = left->dtype == DType::Float64
      ? "cudaverse_binary_f64"
      : "cudaverse_binary_f32";
  CUfunction function = get_kernel(kernel);
  SharedBuffer* output = allocate_buffer(
      left->dtype, left->shape, left->elements);
  CUdeviceptr left_values = left->pointer;
  CUdeviceptr right_values = right->pointer;
  CUdeviceptr output_values = output->pointer;
  unsigned long long elements = left->elements;
  void* parameters[] = {
      &left_values, &right_values, &output_values, &operation, &elements};
  CUresult status = launch_1d(function, left->elements, parameters);
  if (status != CUDA_SUCCESS) {
    destroy_buffer(output);
    check_cuda(status, kernel);
  }
  return make_pointer(output, false);
}

extern "C" SEXP C_cudaverse_cuda_transpose(SEXP pointer) {
  require_backend();
  require_kernels();
  SharedBuffer* input = get_buffer(pointer);
  if (input->shape.size() != 2) {
    Rf_error("Native CUDA transpose requires one two-dimensional tensor.");
  }
  int rows = input->shape[0];
  int columns = input->shape[1];
  std::vector<int> output_shape{columns, rows};
  const char* kernel = input->dtype == DType::Float64
      ? "cudaverse_transpose_f64"
      : (input->dtype == DType::Float32
             ? "cudaverse_transpose_f32"
             : "cudaverse_transpose_i32");
  CUfunction function = get_kernel(kernel);
  SharedBuffer* output = allocate_buffer(
      input->dtype, output_shape, input->elements);
  CUdeviceptr input_values = input->pointer;
  CUdeviceptr output_values = output->pointer;
  void* parameters[] = {
      &input_values, &output_values, &rows, &columns};
  CUresult status = launch_1d(function, input->elements, parameters);
  if (status != CUDA_SUCCESS) {
    destroy_buffer(output);
    check_cuda(status, kernel);
  }
  return make_pointer(output, false);
}

extern "C" SEXP C_cudaverse_cuda_reduce(SEXP pointer, SEXP dim_sexp,
                                           SEXP keepdim_sexp,
                                           SEXP method_sexp) {
  require_backend();
  require_kernels();
  SharedBuffer* input = get_buffer(pointer);
  if (input->shape.size() > CUDAVERSE_MAX_RANK) {
    Rf_error("Native CUDA reductions support at most %d dimensions.",
             CUDAVERSE_MAX_RANK);
  }
  if (TYPEOF(dim_sexp) != INTSXP) {
    Rf_error("`dim` must be an integer vector.");
  }
  if (TYPEOF(keepdim_sexp) != LGLSXP || XLENGTH(keepdim_sexp) != 1 ||
      LOGICAL(keepdim_sexp)[0] == NA_LOGICAL) {
    Rf_error("`keepdim` must be TRUE or FALSE.");
  }
  if (TYPEOF(method_sexp) != STRSXP || XLENGTH(method_sexp) != 1) {
    Rf_error("`method` must be one character string.");
  }
  std::string method = CHAR(STRING_ELT(method_sexp, 0));
  if (method != "sum" && method != "mean") {
    Rf_error("Native CUDA reduction method must be `sum` or `mean`.");
  }

  ReductionMeta meta{};
  meta.rank = static_cast<int>(input->shape.size());
  meta.keepdim = LOGICAL(keepdim_sexp)[0] != 0;
  meta.operation = method == "mean" ? 1 : 0;
  for (int dimension = 0; dimension < meta.rank; ++dimension) {
    meta.shape[dimension] = input->shape[dimension];
  }
  if (XLENGTH(dim_sexp) == 0) {
    for (int dimension = 0; dimension < meta.rank; ++dimension) {
      meta.reduced[dimension] = 1;
    }
  } else {
    for (R_xlen_t index = 0; index < XLENGTH(dim_sexp); ++index) {
      int dimension = INTEGER(dim_sexp)[index];
      if (dimension < 1 || dimension > meta.rank) {
        Rf_error("Native CUDA reduction dimension is out of range.");
      }
      meta.reduced[dimension - 1] = 1;
    }
  }

  std::vector<int> output_shape;
  if (meta.keepdim) {
    for (int dimension = 0; dimension < meta.rank; ++dimension) {
      int output_dimension = static_cast<int>(output_shape.size());
      output_shape.push_back(meta.reduced[dimension]
                                 ? 1
                                 : input->shape[dimension]);
      meta.output_to_input[output_dimension] =
          meta.reduced[dimension] ? -1 : dimension;
    }
  } else {
    for (int dimension = 0; dimension < meta.rank; ++dimension) {
      if (!meta.reduced[dimension]) {
        int output_dimension = static_cast<int>(output_shape.size());
        output_shape.push_back(input->shape[dimension]);
        meta.output_to_input[output_dimension] = dimension;
      }
    }
    if (output_shape.empty()) {
      output_shape.push_back(1);
      meta.output_to_input[0] = -1;
    }
  }
  meta.output_rank = static_cast<int>(output_shape.size());
  std::size_t output_elements = 1;
  for (int dimension = 0; dimension < meta.output_rank; ++dimension) {
    meta.output_shape[dimension] = output_shape[dimension];
    output_elements *= static_cast<std::size_t>(output_shape[dimension]);
  }

  const char* kernel = input->dtype == DType::Float64
      ? "cudaverse_reduce_f64"
      : (input->dtype == DType::Float32
             ? "cudaverse_reduce_f32"
             : "cudaverse_reduce_i32");
  R_CheckUserInterrupt();
  CUfunction function = get_kernel(kernel);
  SharedBuffer* output = allocate_buffer(
      input->dtype, output_shape, output_elements);
  CUdeviceptr input_pointer = input->pointer;
  CUdeviceptr output_pointer = output->pointer;
  unsigned long long kernel_elements = output_elements;
  void* parameters[] = {
      &input_pointer, &output_pointer, &meta, &kernel_elements};
  CUresult status = launch_1d(function, output_elements, parameters);
  if (status != CUDA_SUCCESS) {
    destroy_buffer(output);
    check_cuda(status, kernel);
  }

  SEXP result = PROTECT(named_list({"storage", "shape"}));
  SEXP result_pointer = PROTECT(make_pointer(output, false));
  SEXP result_shape = PROTECT(Rf_allocVector(INTSXP, output_shape.size()));
  for (std::size_t index = 0; index < output_shape.size(); ++index) {
    INTEGER(result_shape)[index] = output_shape[index];
  }
  SET_VECTOR_ELT(result, 0, result_pointer);
  SET_VECTOR_ELT(result, 1, result_shape);
  UNPROTECT(3);
  return result;
}

extern "C" SEXP C_cudaverse_cuda_sparse_from_coo(
    SEXP i_sexp, SEXP j_sexp, SEXP values_sexp, SEXP shape_sexp,
    SEXP format_sexp) {
  require_backend();
  require_kernels();
  if (TYPEOF(shape_sexp) != INTSXP || XLENGTH(shape_sexp) != 2) {
    Rf_error("Sparse `shape` must contain two integers.");
  }
  int rows = INTEGER(shape_sexp)[0];
  int columns = INTEGER(shape_sexp)[1];
  if (rows < 1 || columns < 1) {
    Rf_error("Sparse dimensions must be positive.");
  }
  if (TYPEOF(format_sexp) != STRSXP || XLENGTH(format_sexp) != 1 ||
      STRING_ELT(format_sexp, 0) == NA_STRING) {
    Rf_error("Sparse `format` must be `coo` or `csr`.");
  }
  std::string format = CHAR(STRING_ELT(format_sexp, 0));
  if (format != "coo" && format != "csr") {
    Rf_error("Sparse `format` must be `coo` or `csr`.");
  }
  if (XLENGTH(i_sexp) != XLENGTH(j_sexp) ||
      XLENGTH(i_sexp) != XLENGTH(values_sexp) ||
      XLENGTH(i_sexp) > std::numeric_limits<int>::max()) {
    Rf_error("Sparse COO vectors must have matching supported lengths.");
  }

  SEXP i_values = PROTECT(Rf_coerceVector(i_sexp, INTSXP));
  SEXP j_values = PROTECT(Rf_coerceVector(j_sexp, INTSXP));
  SEXP numeric_values = PROTECT(Rf_coerceVector(values_sexp, REALSXP));
  try {
    int nnz = static_cast<int>(XLENGTH(i_values));
    std::vector<int> row_index(static_cast<std::size_t>(nnz));
    std::vector<int> col_index(static_cast<std::size_t>(nnz));
    std::vector<int> row_ptr(static_cast<std::size_t>(rows) + 1, 0);
    std::vector<double> values(static_cast<std::size_t>(nnz));
    int previous_row = -1;
    int previous_column = -1;
    for (int position = 0; position < nnz; ++position) {
      int row = INTEGER(i_values)[position] - 1;
      int column = INTEGER(j_values)[position] - 1;
      double value = REAL(numeric_values)[position];
      if (row < 0 || row >= rows || column < 0 || column >= columns) {
        throw std::runtime_error("Sparse COO indices are outside `shape`.");
      }
      if (!R_FINITE(value)) {
        throw std::runtime_error("Native CUDA sparse values must be finite.");
      }
      if (row < previous_row ||
          (row == previous_row && column <= previous_column)) {
        throw std::runtime_error(
            "Sparse COO entries must be unique and sorted by row and column.");
      }
      row_index[static_cast<std::size_t>(position)] = row;
      col_index[static_cast<std::size_t>(position)] = column;
      values[static_cast<std::size_t>(position)] = value;
      ++row_ptr[static_cast<std::size_t>(row) + 1];
      previous_row = row;
      previous_column = column;
    }
    for (int row = 0; row < rows; ++row) {
      row_ptr[static_cast<std::size_t>(row) + 1] +=
          row_ptr[static_cast<std::size_t>(row)];
    }

    DeviceMemory device_row_index = copy_host_to_device(
        row_index.data(), row_index.size() * sizeof(int),
        "cuMemcpyHtoD(sparse row index)");
    DeviceMemory device_row_ptr = copy_host_to_device(
        row_ptr.data(), row_ptr.size() * sizeof(int),
        "cuMemcpyHtoD(sparse row pointer)");
    DeviceMemory device_col_index = copy_host_to_device(
        col_index.data(), col_index.size() * sizeof(int),
        "cuMemcpyHtoD(sparse column index)");
    DeviceMemory device_values = copy_host_to_device(
        values.data(), values.size() * sizeof(double),
        "cuMemcpyHtoD(sparse values)");
    SharedSparseBuffer* sparse = shared_sparse_from_devices(
        std::move(device_row_index), std::move(device_row_ptr),
        std::move(device_col_index), std::move(device_values),
        rows, columns, nnz);
    SEXP result = make_sparse_pointer(sparse, false);
    UNPROTECT(3);
    return result;
  } catch (const std::exception& exception) {
    UNPROTECT(3);
    Rf_error("%s", exception.what());
  }
  return R_NilValue;
}

extern "C" SEXP C_cudaverse_cuda_sparse_to_host(SEXP pointer) {
  require_backend();
  SharedSparseBuffer* sparse = get_sparse_buffer(pointer);
  SEXP result = PROTECT(named_list(
      {"i", "j", "values", "row_ptr", "col_index", "shape"}));
  SEXP i = PROTECT(Rf_allocVector(INTSXP, sparse->nnz));
  SEXP j = PROTECT(Rf_allocVector(INTSXP, sparse->nnz));
  SEXP values = PROTECT(Rf_allocVector(REALSXP, sparse->nnz));
  SEXP row_ptr = PROTECT(Rf_allocVector(INTSXP, sparse->rows + 1));
  SEXP col_index = PROTECT(Rf_allocVector(INTSXP, sparse->nnz));
  SEXP shape = PROTECT(Rf_allocVector(INTSXP, 2));
  if (sparse->nnz > 0) {
    check_cuda(api.cuMemcpyDtoH(INTEGER(i), sparse->row_index,
                                sparse->row_index_bytes),
               "cuMemcpyDtoH(sparse row index)");
    check_cuda(api.cuMemcpyDtoH(INTEGER(j), sparse->col_index,
                                sparse->col_index_bytes),
               "cuMemcpyDtoH(sparse column index)");
    check_cuda(api.cuMemcpyDtoH(REAL(values), sparse->values,
                                sparse->value_bytes),
               "cuMemcpyDtoH(sparse values)");
    check_cuda(api.cuMemcpyDtoH(INTEGER(col_index), sparse->col_index,
                                sparse->col_index_bytes),
               "cuMemcpyDtoH(sparse column index)");
    for (int position = 0; position < sparse->nnz; ++position) {
      ++INTEGER(i)[position];
      ++INTEGER(j)[position];
    }
  }
  check_cuda(api.cuMemcpyDtoH(INTEGER(row_ptr), sparse->row_ptr,
                              sparse->row_ptr_bytes),
             "cuMemcpyDtoH(sparse row pointer)");
  INTEGER(shape)[0] = sparse->rows;
  INTEGER(shape)[1] = sparse->columns;
  SET_VECTOR_ELT(result, 0, i);
  SET_VECTOR_ELT(result, 1, j);
  SET_VECTOR_ELT(result, 2, values);
  SET_VECTOR_ELT(result, 3, row_ptr);
  SET_VECTOR_ELT(result, 4, col_index);
  SET_VECTOR_ELT(result, 5, shape);
  UNPROTECT(7);
  return result;
}

extern "C" SEXP C_cudaverse_cuda_sparse_transpose(SEXP pointer) {
  require_backend();
  require_kernels();
  SharedSparseBuffer* sparse = get_sparse_buffer(pointer);
  try {
    int output_rows = sparse->columns;
    DeviceMemory counts(
        static_cast<std::size_t>(output_rows) * sizeof(int));
    CUdeviceptr count_pointer = counts.pointer();
    int zero = 0;
    unsigned long long count_elements =
        static_cast<unsigned long long>(output_rows);
    void* fill_parameters[] = {&count_pointer, &zero, &count_elements};
    launch_or_throw(get_kernel("cudaverse_fill_i32"),
                    "cudaverse_fill_i32", output_rows, fill_parameters);

    void* count_parameters[] = {
        &sparse->col_index, &count_pointer, &sparse->nnz};
    launch_or_throw(
        get_kernel("cudaverse_sparse_transpose_count_i32"),
        "cudaverse_sparse_transpose_count_i32", sparse->nnz,
        count_parameters);

    DeviceMemory row_ptr(
        (static_cast<std::size_t>(output_rows) + 1) * sizeof(int));
    CUdeviceptr row_ptr_pointer = row_ptr.pointer();
    void* prefix_parameters[] = {
        &count_pointer, &row_ptr_pointer, &output_rows};
    launch_or_throw(get_kernel("cudaverse_sparse_prefix_i32"),
                    "cudaverse_sparse_prefix_i32", 1, prefix_parameters);
    DeviceMemory offsets = copy_device_to_device(
        row_ptr.pointer(), static_cast<std::size_t>(output_rows) * sizeof(int),
        "cuMemcpyDtoD(sparse transpose offsets)");

    DeviceMemory row_index(sparse->row_index_bytes);
    DeviceMemory col_index(sparse->col_index_bytes);
    DeviceMemory values(sparse->value_bytes);
    CUdeviceptr offset_pointer = offsets.pointer();
    CUdeviceptr output_row = row_index.pointer();
    CUdeviceptr output_column = col_index.pointer();
    CUdeviceptr output_values = values.pointer();
    void* scatter_parameters[] = {
        &sparse->row_index, &sparse->col_index, &sparse->values,
        &offset_pointer, &output_row, &output_column, &output_values,
        &sparse->nnz};
    launch_or_throw(
        get_kernel("cudaverse_sparse_transpose_scatter_f64"),
        "cudaverse_sparse_transpose_scatter_f64", 1,
        scatter_parameters);

    SharedSparseBuffer* output = shared_sparse_from_devices(
        std::move(row_index), std::move(row_ptr), std::move(col_index),
        std::move(values), sparse->columns, sparse->rows, sparse->nnz);
    return make_sparse_pointer(output, false);
  } catch (const std::exception& exception) {
    Rf_error("%s", exception.what());
  }
  return R_NilValue;
}

extern "C" SEXP C_cudaverse_cuda_sparse_reduce(SEXP pointer,
                                                  SEXP margin_sexp) {
  require_backend();
  require_kernels();
  SharedSparseBuffer* sparse = get_sparse_buffer(pointer);
  int margin = Rf_asInteger(margin_sexp);
  if (margin != 0 && margin != 1) {
    Rf_error("Sparse reduction margin must be zero (rows) or one (columns).");
  }
  try {
    DeviceMemory sums = sparse_sums_device(sparse, margin);
    int groups = margin == 0 ? sparse->rows : sparse->columns;
    SEXP result = PROTECT(Rf_allocVector(REALSXP, groups));
    cuda_or_throw(api.cuMemcpyDtoH(
                      REAL(result), sums.pointer(), sums.bytes()),
                  "cuMemcpyDtoH(sparse reduction)");
    UNPROTECT(1);
    return result;
  } catch (const std::exception& exception) {
    Rf_error("%s", exception.what());
  }
  return R_NilValue;
}

extern "C" SEXP C_cudaverse_cuda_sparse_normalize(
    SEXP pointer, SEXP margin_sexp, SEXP scale_sexp, SEXP log1p_sexp) {
  require_backend();
  require_kernels();
  SharedSparseBuffer* sparse = get_sparse_buffer(pointer);
  int margin = Rf_asInteger(margin_sexp);
  double scale_factor = Rf_asReal(scale_sexp);
  int use_log1p = Rf_asLogical(log1p_sexp);
  if ((margin != 0 && margin != 1) || !R_FINITE(scale_factor) ||
      scale_factor <= 0 || use_log1p == NA_LOGICAL) {
    Rf_error("Invalid native sparse normalization parameters.");
  }
  try {
    DeviceMemory sums = sparse_sums_device(sparse, margin);
    int groups = margin == 0 ? sparse->rows : sparse->columns;
    std::vector<double> host_sums(static_cast<std::size_t>(groups));
    cuda_or_throw(api.cuMemcpyDtoH(
                      host_sums.data(), sums.pointer(), sums.bytes()),
                  "cuMemcpyDtoH(sparse normalization sums)");
    for (double value : host_sums) {
      if (!std::isfinite(value) || value <= 0) {
        throw std::runtime_error(
            "Every normalized sparse margin must have a positive finite sum.");
      }
    }

    DeviceMemory normalized(sparse->value_bytes);
    CUdeviceptr sum_pointer = sums.pointer();
    CUdeviceptr output_pointer = normalized.pointer();
    CUfunction function = get_kernel("cudaverse_sparse_normalize_f64");
    void* parameters[] = {
        &sparse->row_index, &sparse->col_index, &sparse->values,
        &sum_pointer, &output_pointer, &sparse->nnz, &margin,
        &scale_factor, &use_log1p};
    launch_or_throw(function, "cudaverse_sparse_normalize_f64",
                    sparse->nnz, parameters);

    DeviceMemory row_index = copy_device_to_device(
        sparse->row_index, sparse->row_index_bytes,
        "cuMemcpyDtoD(sparse row index)");
    DeviceMemory row_ptr = copy_device_to_device(
        sparse->row_ptr, sparse->row_ptr_bytes,
        "cuMemcpyDtoD(sparse row pointer)");
    DeviceMemory col_index = copy_device_to_device(
        sparse->col_index, sparse->col_index_bytes,
        "cuMemcpyDtoD(sparse column index)");
    SharedSparseBuffer* output = shared_sparse_from_devices(
        std::move(row_index), std::move(row_ptr), std::move(col_index),
        std::move(normalized), sparse->rows, sparse->columns, sparse->nnz);
    SEXP output_pointer_sexp = PROTECT(make_sparse_pointer(output, false));
    SEXP host_values = PROTECT(Rf_allocVector(REALSXP, sparse->nnz));
    if (sparse->nnz > 0) {
      cuda_or_throw(api.cuMemcpyDtoH(
                        REAL(host_values), output->values,
                        output->value_bytes),
                    "cuMemcpyDtoH(normalized sparse values)");
    }
    SEXP result = PROTECT(named_list({"storage", "values"}));
    SET_VECTOR_ELT(result, 0, output_pointer_sexp);
    SET_VECTOR_ELT(result, 1, host_values);
    UNPROTECT(3);
    return result;
  } catch (const std::exception& exception) {
    Rf_error("%s", exception.what());
  }
  return R_NilValue;
}

extern "C" SEXP C_cudaverse_cuda_sparse_matmul_dense(
    SEXP sparse_pointer, SEXP dense_pointer) {
  require_backend();
  require_kernels();
  SharedSparseBuffer* sparse = get_sparse_buffer(sparse_pointer);
  SharedBuffer* dense = get_buffer(dense_pointer);
  if (dense->dtype != DType::Float64 || dense->shape.size() != 2 ||
      dense->shape[0] != sparse->columns) {
    Rf_error("Native sparse-dense multiplication requires a conformable float64 matrix.");
  }
  int dense_columns = dense->shape[1];
  std::size_t elements =
      static_cast<std::size_t>(sparse->rows) * dense_columns;
  try {
    DeviceMemory device_output(elements * sizeof(double));
    CUdeviceptr dense_values = dense->pointer;
    CUdeviceptr output_values = device_output.pointer();
    CUfunction function = get_kernel("cudaverse_csr_spmm_f64");
    void* parameters[] = {
        &sparse->row_ptr, &sparse->col_index, &sparse->values,
        &dense_values, &output_values, &sparse->rows, &sparse->columns,
        &dense_columns};
    launch_or_throw(function, "cudaverse_csr_spmm_f64",
                    elements, parameters);
    SharedBuffer* output = shared_buffer_from_device(
        std::move(device_output), DType::Float64,
        {sparse->rows, dense_columns}, elements);
    return make_pointer(output, false);
  } catch (const std::exception& exception) {
    Rf_error("%s", exception.what());
  }
  return R_NilValue;
}

extern "C" SEXP C_cudaverse_cuda_sparse_to_dense(SEXP pointer) {
  require_backend();
  require_kernels();
  SharedSparseBuffer* sparse = get_sparse_buffer(pointer);
  try {
    DeviceMemory device_output = sparse_to_dense_device(sparse);
    std::size_t elements =
        static_cast<std::size_t>(sparse->rows) * sparse->columns;
    SharedBuffer* output = shared_buffer_from_device(
        std::move(device_output), DType::Float64,
        {sparse->rows, sparse->columns}, elements);
    return make_pointer(output, false);
  } catch (const std::exception& exception) {
    Rf_error("%s", exception.what());
  }
  return R_NilValue;
}

extern "C" SEXP C_cudaverse_cuda_svd(SEXP pointer, SEXP nu_sexp,
                                        SEXP nv_sexp) {
  SharedBuffer* input = get_buffer(pointer);
  if (input->dtype != DType::Float64 || input->shape.size() != 2) {
    Rf_error("Native CUDA SVD requires one float64 matrix.");
  }
  int rows = input->shape[0];
  int columns = input->shape[1];
  int rank = rows < columns ? rows : columns;
  int nu = Rf_asInteger(nu_sexp);
  int nv = Rf_asInteger(nv_sexp);
  if (nu == NA_INTEGER || nv == NA_INTEGER ||
      nu < 0 || nu > rank || nv < 0 || nv > rank) {
    Rf_error("Native CUDA SVD vector counts are out of range.");
  }
  R_CheckUserInterrupt();
  try {
    DeviceSvd decomposition = run_device_svd(input->pointer, rows, columns);
    std::vector<double> values = copy_double_to_host(
        decomposition.values.pointer(), decomposition.rank,
        "cuMemcpyDtoH(SVD values)");
    std::vector<double> left = copy_double_to_host(
        decomposition.left.pointer(), static_cast<std::size_t>(rows) * nu,
        "cuMemcpyDtoH(SVD left vectors)");
    std::vector<double> right = copy_double_to_host(
        decomposition.right.pointer(), static_cast<std::size_t>(columns) * nv,
        "cuMemcpyDtoH(SVD right vectors)");

    SEXP result = PROTECT(named_list({"d", "u", "v"}));
    SEXP d = PROTECT(numeric_vector(values));
    SEXP u = PROTECT(numeric_vector(left));
    SEXP v = PROTECT(numeric_vector(right));
    SET_VECTOR_ELT(result, 0, d);
    SET_VECTOR_ELT(result, 1, u);
    SET_VECTOR_ELT(result, 2, v);
    UNPROTECT(4);
    return result;
  } catch (const std::exception& exception) {
    Rf_error("%s", exception.what());
  }
  return R_NilValue;
}

extern "C" SEXP C_cudaverse_cuda_pca(SEXP pointer,
                                        SEXP components_sexp,
                                        SEXP center_sexp,
                                        SEXP scale_sexp) {
  SharedBuffer* input = get_buffer(pointer);
  if (input->dtype != DType::Float64 || input->shape.size() != 2) {
    Rf_error("Native CUDA PCA requires one float64 matrix.");
  }
  int rows = input->shape[0];
  int columns = input->shape[1];
  int components = Rf_asInteger(components_sexp);
  int max_components = (rows - 1) < columns ? (rows - 1) : columns;
  if (components == NA_INTEGER || components < 1 ||
      components > max_components) {
    Rf_error("Native CUDA PCA component count is out of range.");
  }
  int use_center = Rf_asLogical(center_sexp);
  int use_scale = Rf_asLogical(scale_sexp);
  if (use_center == NA_LOGICAL || use_scale == NA_LOGICAL) {
    Rf_error("Native CUDA PCA flags must be TRUE or FALSE.");
  }
  R_CheckUserInterrupt();
  try {
    std::string reason;
    if (!load_cusolver(reason)) throw std::runtime_error(reason);
    require_kernels();
    CUfunction statistics = get_kernel("cudaverse_column_stats_f64");
    CUfunction transform = get_kernel("cudaverse_center_scale_f64");
    CUfunction scale_columns = get_kernel("cudaverse_scale_columns_f64");

    DeviceMemory centers(static_cast<std::size_t>(columns) * sizeof(double));
    DeviceMemory scales(static_cast<std::size_t>(columns) * sizeof(double));
    DeviceMemory transformed(
        static_cast<std::size_t>(rows) * columns * sizeof(double));
    CUdeviceptr input_pointer = input->pointer;
    CUdeviceptr center_pointer = centers.pointer();
    CUdeviceptr scale_pointer = scales.pointer();
    void* statistics_parameters[] = {
        &input_pointer, &center_pointer, &scale_pointer, &rows, &columns,
        &use_center, &use_scale};
    launch_or_throw(statistics, "cudaverse_column_stats_f64", columns,
                    statistics_parameters);

    std::vector<double> host_centers = copy_double_to_host(
        centers.pointer(), columns, "cuMemcpyDtoH(PCA center)");
    std::vector<double> host_scales = copy_double_to_host(
        scales.pointer(), columns, "cuMemcpyDtoH(PCA scale)");
    if (use_scale) {
      for (double value : host_scales) {
        if (!std::isfinite(value) || value <= 0.0) {
          throw std::runtime_error(
              "Cannot scale constant or non-finite features.");
        }
      }
    }

    CUdeviceptr transformed_pointer = transformed.pointer();
    void* transform_parameters[] = {
        &input_pointer, &transformed_pointer, &center_pointer,
        &scale_pointer, &rows, &columns};
    launch_or_throw(
        transform, "cudaverse_center_scale_f64",
        static_cast<std::size_t>(rows) * columns, transform_parameters);

    DeviceSvd decomposition = run_device_svd(
        transformed.pointer(), rows, columns);
    DeviceMemory scores(
        static_cast<std::size_t>(rows) * components * sizeof(double));
    CUdeviceptr left_pointer = decomposition.left.pointer();
    CUdeviceptr scores_pointer = scores.pointer();
    CUdeviceptr singular_pointer = decomposition.values.pointer();
    void* score_parameters[] = {
        &left_pointer, &scores_pointer, &singular_pointer,
        &rows, &components};
    launch_or_throw(
        scale_columns, "cudaverse_scale_columns_f64",
        static_cast<std::size_t>(rows) * components, score_parameters);

    std::vector<double> host_singular = copy_double_to_host(
        decomposition.values.pointer(), components,
        "cuMemcpyDtoH(PCA singular values)");
    std::vector<double> host_rotation = copy_double_to_host(
        decomposition.right.pointer(),
        static_cast<std::size_t>(columns) * components,
        "cuMemcpyDtoH(PCA rotation)");
    std::vector<double> host_scores = copy_double_to_host(
        scores.pointer(), static_cast<std::size_t>(rows) * components,
        "cuMemcpyDtoH(PCA scores)");
    double denominator = std::sqrt(static_cast<double>(rows - 1));
    for (double& value : host_singular) value /= denominator;

    std::size_t score_elements =
        static_cast<std::size_t>(rows) * components;
    auto* score_buffer = buffer_from_device(
        scores.release(), score_elements * sizeof(double), score_elements,
        DType::Float64, {rows, components});
    SEXP score_storage = PROTECT(make_pointer(score_buffer, false));
    SEXP result = PROTECT(named_list({
        "sdev", "rotation", "x", "center", "scale", "scores_storage"}));
    SEXP sdev = PROTECT(numeric_vector(host_singular));
    SEXP rotation = PROTECT(numeric_vector(host_rotation));
    SEXP host_x = PROTECT(numeric_vector(host_scores));
    SEXP host_center = PROTECT(numeric_vector(host_centers));
    SEXP host_scale = PROTECT(numeric_vector(host_scales));
    SET_VECTOR_ELT(result, 0, sdev);
    SET_VECTOR_ELT(result, 1, rotation);
    SET_VECTOR_ELT(result, 2, host_x);
    SET_VECTOR_ELT(result, 3, host_center);
    SET_VECTOR_ELT(result, 4, host_scale);
    SET_VECTOR_ELT(result, 5, score_storage);
    UNPROTECT(7);
    return result;
  } catch (const std::exception& exception) {
    Rf_error("%s", exception.what());
  }
  return R_NilValue;
}

extern "C" SEXP C_cudaverse_cuda_row_norms(SEXP pointer) {
  require_backend();
  require_kernels();
  SharedBuffer* input = get_buffer(pointer);
  if (input->dtype != DType::Float64 || input->shape.size() != 2) {
    Rf_error("Native CUDA row norms require one float64 matrix.");
  }
  try {
    DeviceMemory norms = row_norms_squared(
        input->pointer, input->shape[0], input->shape[1]);
    std::size_t elements = static_cast<std::size_t>(input->shape[0]);
    auto* result = buffer_from_device(
        norms.release(), elements * sizeof(double), elements,
        DType::Float64, {input->shape[0]});
    return make_pointer(result, false);
  } catch (const std::exception& exception) {
    Rf_error("%s", exception.what());
  }
  return R_NilValue;
}

extern "C" SEXP C_cudaverse_cuda_normalize_rows(SEXP pointer) {
  require_backend();
  require_kernels();
  SharedBuffer* input = get_buffer(pointer);
  if (input->dtype != DType::Float64 || input->shape.size() != 2) {
    Rf_error("Native CUDA row normalization requires one float64 matrix.");
  }
  try {
    CUfunction normalize = get_kernel("cudaverse_normalize_rows_f64");
    DeviceMemory output(input->bytes);
    CUdeviceptr input_pointer = input->pointer;
    CUdeviceptr output_pointer = output.pointer();
    int rows = input->shape[0];
    int columns = input->shape[1];
    void* parameters[] = {
        &input_pointer, &output_pointer, &rows, &columns};
    launch_or_throw(normalize, "cudaverse_normalize_rows_f64", rows,
                    parameters);
    auto* result = buffer_from_device(
        output.release(), input->bytes, input->elements, DType::Float64,
        input->shape);
    return make_pointer(result, false);
  } catch (const std::exception& exception) {
    Rf_error("%s", exception.what());
  }
  return R_NilValue;
}

extern "C" SEXP C_cudaverse_cuda_distance(SEXP query_pointer,
                                             SEXP reference_pointer,
                                             SEXP metric_sexp,
                                             SEXP self_sexp) {
  require_backend();
  require_kernels();
  SharedBuffer* query = get_buffer(query_pointer);
  SharedBuffer* reference = get_buffer(reference_pointer);
  if (query->dtype != DType::Float64 ||
      reference->dtype != DType::Float64 ||
      query->shape.size() != 2 || reference->shape.size() != 2 ||
      query->shape[1] != reference->shape[1]) {
    Rf_error("Native CUDA distance requires conformable float64 matrices.");
  }
  std::string metric_name = CHAR(STRING_ELT(metric_sexp, 0));
  int metric = metric_name == "cosine" ? 1 : 0;
  int self = Rf_asLogical(self_sexp) == 1 ? 1 : 0;
  try {
    DeviceMemory distance = distance_device(
        query->pointer, query->shape[0], reference->pointer,
        reference->shape[0], query->shape[1], metric, 0, self);
    std::size_t elements = static_cast<std::size_t>(query->shape[0]) *
                           reference->shape[0];
    auto* result = buffer_from_device(
        distance.release(), elements * sizeof(double), elements,
        DType::Float64, {query->shape[0], reference->shape[0]});
    return make_pointer(result, false);
  } catch (const std::exception& exception) {
    Rf_error("%s", exception.what());
  }
  return R_NilValue;
}

extern "C" SEXP C_cudaverse_cuda_kmeans(
    SEXP input_pointer, SEXP centers_pointer, SEXP iter_max_sexp,
    SEXP tolerance_sexp) {
  require_backend();
  require_kernels();
  SharedBuffer* input = get_buffer(input_pointer);
  SharedBuffer* initial_centers = get_buffer(centers_pointer);
  if (input->dtype != DType::Float64 ||
      initial_centers->dtype != DType::Float64 ||
      input->shape.size() != 2 || initial_centers->shape.size() != 2 ||
      input->shape[1] != initial_centers->shape[1]) {
    Rf_error("Native CUDA k-means requires conformable float64 matrices.");
  }
  int rows = input->shape[0];
  int columns = input->shape[1];
  int centers_count = initial_centers->shape[0];
  int iter_max = Rf_asInteger(iter_max_sexp);
  double tolerance = Rf_asReal(tolerance_sexp);
  if (centers_count < 1 || centers_count >= rows) {
    Rf_error("Native CUDA k-means centre count is out of range.");
  }
  if (iter_max == NA_INTEGER || iter_max < 1 ||
      !std::isfinite(tolerance) || tolerance <= 0.0) {
    Rf_error("Native CUDA k-means iteration parameters are invalid.");
  }

  R_CheckUserInterrupt();
  try {
    CUfunction fill_double = get_kernel("cudaverse_fill_f64");
    CUfunction fill_integer = get_kernel("cudaverse_fill_i32");
    CUfunction assign = get_kernel("cudaverse_kmeans_assign_f64");
    CUfunction count = get_kernel("cudaverse_kmeans_count_i32");
    CUfunction accumulate = get_kernel("cudaverse_kmeans_accumulate_f64");
    CUfunction update = get_kernel("cudaverse_kmeans_update_f64");

    std::size_t center_elements =
        static_cast<std::size_t>(centers_count) * columns;
    DeviceMemory centers = copy_device_to_device(
        initial_centers->pointer, center_elements * sizeof(double),
        "cuMemcpyDtoD(k-means initial centers)");
    DeviceMemory sums(center_elements * sizeof(double));
    DeviceMemory counts(static_cast<std::size_t>(centers_count) * sizeof(int));
    DeviceMemory assignments(static_cast<std::size_t>(rows) * sizeof(int));
    DeviceMemory minimum_distance(
        static_cast<std::size_t>(rows) * sizeof(double));
    DeviceMemory movement(center_elements * sizeof(double));

    auto assign_nearest = [&](DeviceMemory& distance) {
      CUdeviceptr distance_pointer = distance.pointer();
      CUdeviceptr assignment_pointer = assignments.pointer();
      CUdeviceptr minimum_pointer = minimum_distance.pointer();
      void* parameters[] = {
          &distance_pointer, &assignment_pointer, &minimum_pointer,
          &rows, &centers_count};
      launch_or_throw(assign, "cudaverse_kmeans_assign_f64", rows,
                      parameters);
    };

    DeviceMemory distances = distance_device(
        input->pointer, rows, centers.pointer(), centers_count,
        columns, 0, 0, 0);
    assign_nearest(distances);

    int iteration = 0;
    bool converged = false;
    for (iteration = 1; iteration <= iter_max; ++iteration) {
      CUdeviceptr sum_pointer = sums.pointer();
      double zero_double = 0.0;
      unsigned long long sum_elements =
          static_cast<unsigned long long>(center_elements);
      void* fill_sum_parameters[] = {
          &sum_pointer, &zero_double, &sum_elements};
      launch_or_throw(fill_double, "cudaverse_fill_f64", center_elements,
                      fill_sum_parameters);

      CUdeviceptr count_pointer = counts.pointer();
      int zero_integer = 0;
      unsigned long long count_elements =
          static_cast<unsigned long long>(centers_count);
      void* fill_count_parameters[] = {
          &count_pointer, &zero_integer, &count_elements};
      launch_or_throw(fill_integer, "cudaverse_fill_i32", centers_count,
                      fill_count_parameters);

      CUdeviceptr assignment_pointer = assignments.pointer();
      void* count_parameters[] = {
          &assignment_pointer, &count_pointer, &rows};
      launch_or_throw(count, "cudaverse_kmeans_count_i32", rows,
                      count_parameters);

      CUdeviceptr input_device = input->pointer;
      void* accumulate_parameters[] = {
          &input_device, &assignment_pointer, &sum_pointer, &rows,
          &columns, &centers_count};
      launch_or_throw(
          accumulate, "cudaverse_kmeans_accumulate_f64",
          static_cast<std::size_t>(rows) * columns,
          accumulate_parameters);

      DeviceMemory updated(center_elements * sizeof(double));
      CUdeviceptr previous_pointer = centers.pointer();
      CUdeviceptr updated_pointer = updated.pointer();
      CUdeviceptr movement_pointer = movement.pointer();
      void* update_parameters[] = {
          &previous_pointer, &sum_pointer, &count_pointer, &updated_pointer,
          &movement_pointer, &centers_count, &columns};
      launch_or_throw(update, "cudaverse_kmeans_update_f64",
                      center_elements, update_parameters);

      std::vector<double> host_movement = copy_double_to_host(
          movement.pointer(), center_elements,
          "cuMemcpyDtoH(k-means movement)");
      double maximum_movement = *std::max_element(
          host_movement.begin(), host_movement.end());
      centers = std::move(updated);

      distances = distance_device(
          input->pointer, rows, centers.pointer(), centers_count,
          columns, 0, 0, 0);
      assign_nearest(distances);
      interrupt_or_throw();
      if (maximum_movement <= tolerance) {
        converged = true;
        break;
      }
    }

    std::vector<double> host_centers = copy_double_to_host(
        centers.pointer(), center_elements,
        "cuMemcpyDtoH(k-means centers)");
    std::vector<double> host_minimum = copy_double_to_host(
        minimum_distance.pointer(), rows,
        "cuMemcpyDtoH(k-means minimum distance)");
    std::vector<int> host_assignments(static_cast<std::size_t>(rows));
    cuda_or_throw(api.cuMemcpyDtoH(
                      host_assignments.data(), assignments.pointer(),
                      static_cast<std::size_t>(rows) * sizeof(int)),
                  "cuMemcpyDtoH(k-means assignments)");
    std::vector<double> withinss(
        static_cast<std::size_t>(centers_count), 0.0);
    for (int row = 0; row < rows; ++row) {
      double distance = host_minimum[static_cast<std::size_t>(row)];
      int center = host_assignments[static_cast<std::size_t>(row)] - 1;
      withinss[static_cast<std::size_t>(center)] += distance * distance;
    }

    SEXP result = PROTECT(named_list({
        "cluster", "centers", "withinss", "iter", "converged"}));
    SEXP cluster = PROTECT(Rf_allocVector(INTSXP, rows));
    SEXP center_values = PROTECT(numeric_vector(host_centers));
    SEXP withinss_values = PROTECT(numeric_vector(withinss));
    std::memcpy(INTEGER(cluster), host_assignments.data(),
                static_cast<std::size_t>(rows) * sizeof(int));
    SET_VECTOR_ELT(result, 0, cluster);
    SET_VECTOR_ELT(result, 1, center_values);
    SET_VECTOR_ELT(result, 2, withinss_values);
    SET_VECTOR_ELT(result, 3, Rf_ScalarInteger(
        iteration > iter_max ? iter_max : iteration));
    SET_VECTOR_ELT(result, 4, Rf_ScalarLogical(converged));
    UNPROTECT(4);
    return result;
  } catch (const std::exception& exception) {
    Rf_error("%s", exception.what());
  }
  return R_NilValue;
}

extern "C" SEXP C_cudaverse_cuda_knn_block(
    SEXP reference_pointer, SEXP reference_norms_pointer,
    SEXP first_sexp, SEXP count_sexp, SEXP k_sexp, SEXP metric_sexp) {
  require_backend();
  require_kernels();
  SharedBuffer* reference = get_buffer(reference_pointer);
  if (reference->dtype != DType::Float64 || reference->shape.size() != 2) {
    Rf_error("Native CUDA kNN requires one float64 reference matrix.");
  }
  int reference_rows = reference->shape[0];
  int columns = reference->shape[1];
  int first = Rf_asInteger(first_sexp);
  int count = Rf_asInteger(count_sexp);
  int k = Rf_asInteger(k_sexp);
  if (first == NA_INTEGER || count == NA_INTEGER || k == NA_INTEGER ||
      first < 0 || count < 1 || first + count > reference_rows ||
      k < 1 || k >= reference_rows) {
    Rf_error("Native CUDA kNN block parameters are out of range.");
  }
  std::string metric_name = CHAR(STRING_ELT(metric_sexp, 0));
  int metric = metric_name == "cosine" ? 1 : 0;
  CUdeviceptr cached_norms = 0;
  if (metric == 0) {
    SharedBuffer* norms = get_buffer(reference_norms_pointer);
    if (norms->dtype != DType::Float64 || norms->elements !=
        static_cast<std::size_t>(reference_rows)) {
      Rf_error("Native CUDA kNN reference norms are invalid.");
    }
    cached_norms = norms->pointer;
  }

  try {
    CUfunction gather = get_kernel("cudaverse_gather_rows_f64");
    DeviceMemory query(
        static_cast<std::size_t>(count) * columns * sizeof(double));
    CUdeviceptr reference_device = reference->pointer;
    CUdeviceptr query_device = query.pointer();
    void* gather_parameters[] = {
        &reference_device, &query_device, &reference_rows, &columns,
        &first, &count};
    launch_or_throw(gather, "cudaverse_gather_rows_f64",
                    static_cast<std::size_t>(count) * columns,
                    gather_parameters);

    DeviceMemory distances = distance_device(
        query.pointer(), count, reference->pointer, reference_rows,
        columns, metric, first, 1, cached_norms);
    std::size_t output_elements = static_cast<std::size_t>(count) * k;
    DeviceMemory output_index(output_elements * sizeof(int));
    DeviceMemory output_distance(output_elements * sizeof(double));
    CUdeviceptr distance_pointer = distances.pointer();
    CUdeviceptr index_pointer = output_index.pointer();
    CUdeviceptr output_distance_pointer = output_distance.pointer();
    int self = 1;
    void* topk_parameters[] = {
        &distance_pointer, &index_pointer, &output_distance_pointer,
        &count, &reference_rows, &k, &first, &self};
    if (k <= 32) {
      constexpr unsigned int threads = 128;
      unsigned int shared_bytes = static_cast<unsigned int>(
          static_cast<std::size_t>(threads) * k *
          (sizeof(double) + sizeof(int)));
      cuda_or_throw(
          api.cuLaunchKernel(
              get_kernel("cudaverse_topk_stable_f64"), count, 1, 1,
              threads, 1, 1, shared_bytes, nullptr, topk_parameters, nullptr),
          "cudaverse_topk_stable_f64");
    } else {
      launch_or_throw(
          get_kernel("cudaverse_topk_stable_general_f64"),
          "cudaverse_topk_stable_general_f64", count, topk_parameters);
    }

    std::vector<int> host_index(output_elements);
    std::vector<double> host_distance(output_elements);
    cuda_or_throw(api.cuMemcpyDtoH(
                      host_index.data(), output_index.pointer(),
                      output_elements * sizeof(int)),
                  "cuMemcpyDtoH(kNN index)");
    cuda_or_throw(api.cuMemcpyDtoH(
                      host_distance.data(), output_distance.pointer(),
                      output_elements * sizeof(double)),
                  "cuMemcpyDtoH(kNN distance)");

    SEXP result = PROTECT(named_list({"index", "distance"}));
    SEXP index = PROTECT(Rf_allocVector(INTSXP, output_elements));
    SEXP distance = PROTECT(Rf_allocVector(REALSXP, output_elements));
    std::memcpy(INTEGER(index), host_index.data(),
                output_elements * sizeof(int));
    std::memcpy(REAL(distance), host_distance.data(),
                output_elements * sizeof(double));
    SET_VECTOR_ELT(result, 0, index);
    SET_VECTOR_ELT(result, 1, distance);
    UNPROTECT(3);
    return result;
  } catch (const std::exception& exception) {
    Rf_error("%s", exception.what());
  }
  return R_NilValue;
}

extern "C" SEXP C_cudaverse_cuda_matmul(SEXP left_pointer,
                                          SEXP right_pointer) {
  require_backend();
  SharedBuffer* left = get_buffer(left_pointer);
  SharedBuffer* right = get_buffer(right_pointer);
  if (left->dtype != right->dtype ||
      (left->dtype != DType::Float64 && left->dtype != DType::Float32)) {
    Rf_error(
        "Native CUDA matmul requires matching float32 or float64 tensors.");
  }
  if (left->shape.size() != 2 || right->shape.size() != 2 ||
      left->shape[1] != right->shape[0]) {
    Rf_error("Native CUDA matrix dimensions are not conformable.");
  }
  int m = left->shape[0];
  int k = left->shape[1];
  int n = right->shape[1];
  R_CheckUserInterrupt();
  SharedBuffer* output = allocate_buffer(
      left->dtype, {m, n}, static_cast<std::size_t>(m) * n);
  cublasStatus_t status = CUBLAS_STATUS_SUCCESS;
  const char* operation = nullptr;
  if (left->dtype == DType::Float64) {
    const double alpha = 1.0;
    const double beta = 0.0;
    status = api.cublasDgemm(
        api.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha,
        reinterpret_cast<const double*>(left->pointer), m,
        reinterpret_cast<const double*>(right->pointer), k, &beta,
        reinterpret_cast<double*>(output->pointer), m);
    operation = "cublasDgemm";
  } else {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    status = api.cublasSgemm(
        api.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha,
        reinterpret_cast<const float*>(left->pointer), m,
        reinterpret_cast<const float*>(right->pointer), k, &beta,
        reinterpret_cast<float*>(output->pointer), m);
    operation = "cublasSgemm";
  }
  if (status != CUBLAS_STATUS_SUCCESS) {
    destroy_buffer(output);
    check_cublas(status, operation);
  }
  return make_pointer(output, false);
}

extern "C" SEXP C_cudaverse_cuda_synchronize() {
  require_backend();
  check_cuda(api.cuCtxSynchronize(), "cuCtxSynchronize");
  return Rf_ScalarLogical(1);
}

extern "C" SEXP C_cudaverse_cuda_test_inject_error(SEXP bytes_sexp) {
  require_backend();
  int bytes = Rf_asInteger(bytes_sexp);
  if (bytes == NA_INTEGER || bytes < 1) {
    Rf_error("`bytes` must be one positive integer.");
  }
  R_CheckUserInterrupt();
  try {
    DeviceMemory temporary(static_cast<std::size_t>(bytes));
    cuda_or_throw(static_cast<CUresult>(2), "injected cuMemAlloc");
  } catch (const std::exception& exception) {
    Rf_error("%s", exception.what());
  }
  return R_NilValue;
}

extern "C" SEXP C_cudaverse_cuda_release(SEXP pointer) {
  get_buffer(pointer);
  release_reference(pointer);
  return Rf_ScalarLogical(1);
}

extern "C" SEXP C_cudaverse_cuda_share(SEXP pointer) {
  SharedBuffer* buffer = get_buffer(pointer);
  return make_pointer(buffer, true);
}

extern "C" SEXP C_cudaverse_cuda_sparse_release(SEXP pointer) {
  get_sparse_buffer(pointer);
  release_sparse_reference(pointer);
  return Rf_ScalarLogical(1);
}

extern "C" SEXP C_cudaverse_cuda_sparse_share(SEXP pointer) {
  SharedSparseBuffer* buffer = get_sparse_buffer(pointer);
  return make_sparse_pointer(buffer, true);
}

extern "C" SEXP C_cudaverse_cuda_memory_info() {
  std::string reason;
  if (!load_driver(reason)) Rf_error("%s", reason.c_str());
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  check_cuda(api.cuMemGetInfo(&free_bytes, &total_bytes), "cuMemGetInfo");
  SEXP result = PROTECT(named_list({"free", "total", "used"}));
  SET_VECTOR_ELT(result, 0, Rf_ScalarReal(static_cast<double>(free_bytes)));
  SET_VECTOR_ELT(result, 1, Rf_ScalarReal(static_cast<double>(total_bytes)));
  SET_VECTOR_ELT(result, 2,
                 Rf_ScalarReal(static_cast<double>(total_bytes - free_bytes)));
  UNPROTECT(1);
  return result;
}

extern "C" SEXP C_cudaverse_cuda_memory_tracker(SEXP reset_sexp) {
  int reset = Rf_asLogical(reset_sexp);
  if (reset == NA_LOGICAL) Rf_error("`reset` must be TRUE or FALSE.");
  std::size_t current = tracked_device_bytes.load();
  if (reset) peak_tracked_device_bytes.store(current);
  std::size_t peak = peak_tracked_device_bytes.load();
  SEXP result = PROTECT(named_list({"current", "peak"}));
  SET_VECTOR_ELT(result, 0, Rf_ScalarReal(static_cast<double>(current)));
  SET_VECTOR_ELT(result, 1, Rf_ScalarReal(static_cast<double>(peak)));
  UNPROTECT(1);
  return result;
}

static const R_CallMethodDef call_methods[] = {
    {"C_cudaverse_cuda_load_kernels",
     reinterpret_cast<DL_FUNC>(&C_cudaverse_cuda_load_kernels), 1},
    {"C_cudaverse_cuda_diagnostics",
     reinterpret_cast<DL_FUNC>(&C_cudaverse_cuda_diagnostics), 0},
    {"C_cudaverse_cuda_from_host",
     reinterpret_cast<DL_FUNC>(&C_cudaverse_cuda_from_host), 3},
    {"C_cudaverse_cuda_to_host",
     reinterpret_cast<DL_FUNC>(&C_cudaverse_cuda_to_host), 1},
    {"C_cudaverse_cuda_sparse_from_coo",
     reinterpret_cast<DL_FUNC>(&C_cudaverse_cuda_sparse_from_coo), 5},
    {"C_cudaverse_cuda_sparse_to_host",
     reinterpret_cast<DL_FUNC>(&C_cudaverse_cuda_sparse_to_host), 1},
    {"C_cudaverse_cuda_sparse_transpose",
     reinterpret_cast<DL_FUNC>(&C_cudaverse_cuda_sparse_transpose), 1},
    {"C_cudaverse_cuda_sparse_reduce",
     reinterpret_cast<DL_FUNC>(&C_cudaverse_cuda_sparse_reduce), 2},
    {"C_cudaverse_cuda_sparse_normalize",
     reinterpret_cast<DL_FUNC>(&C_cudaverse_cuda_sparse_normalize), 4},
    {"C_cudaverse_cuda_sparse_matmul_dense",
     reinterpret_cast<DL_FUNC>(&C_cudaverse_cuda_sparse_matmul_dense), 2},
    {"C_cudaverse_cuda_sparse_to_dense",
     reinterpret_cast<DL_FUNC>(&C_cudaverse_cuda_sparse_to_dense), 1},
     {"C_cudaverse_cuda_cast",
      reinterpret_cast<DL_FUNC>(&C_cudaverse_cuda_cast), 2},
     {"C_cudaverse_cuda_reshape",
      reinterpret_cast<DL_FUNC>(&C_cudaverse_cuda_reshape), 2},
     {"C_cudaverse_cuda_gather",
      reinterpret_cast<DL_FUNC>(&C_cudaverse_cuda_gather), 3},
     {"C_cudaverse_cuda_scatter",
      reinterpret_cast<DL_FUNC>(&C_cudaverse_cuda_scatter), 4},
     {"C_cudaverse_cuda_broadcast",
      reinterpret_cast<DL_FUNC>(&C_cudaverse_cuda_broadcast), 2},
     {"C_cudaverse_cuda_binary",
      reinterpret_cast<DL_FUNC>(&C_cudaverse_cuda_binary), 3},
     {"C_cudaverse_cuda_transpose",
      reinterpret_cast<DL_FUNC>(&C_cudaverse_cuda_transpose), 1},
    {"C_cudaverse_cuda_reduce",
     reinterpret_cast<DL_FUNC>(&C_cudaverse_cuda_reduce), 4},
    {"C_cudaverse_cuda_svd",
     reinterpret_cast<DL_FUNC>(&C_cudaverse_cuda_svd), 3},
    {"C_cudaverse_cuda_pca",
     reinterpret_cast<DL_FUNC>(&C_cudaverse_cuda_pca), 4},
    {"C_cudaverse_cuda_row_norms",
     reinterpret_cast<DL_FUNC>(&C_cudaverse_cuda_row_norms), 1},
    {"C_cudaverse_cuda_normalize_rows",
     reinterpret_cast<DL_FUNC>(&C_cudaverse_cuda_normalize_rows), 1},
    {"C_cudaverse_cuda_distance",
     reinterpret_cast<DL_FUNC>(&C_cudaverse_cuda_distance), 4},
    {"C_cudaverse_cuda_kmeans",
     reinterpret_cast<DL_FUNC>(&C_cudaverse_cuda_kmeans), 4},
    {"C_cudaverse_cuda_knn_block",
     reinterpret_cast<DL_FUNC>(&C_cudaverse_cuda_knn_block), 6},
    {"C_cudaverse_cuda_matmul",
     reinterpret_cast<DL_FUNC>(&C_cudaverse_cuda_matmul), 2},
    {"C_cudaverse_cuda_synchronize",
     reinterpret_cast<DL_FUNC>(&C_cudaverse_cuda_synchronize), 0},
    {"C_cudaverse_cuda_test_inject_error",
     reinterpret_cast<DL_FUNC>(&C_cudaverse_cuda_test_inject_error), 1},
    {"C_cudaverse_cuda_release",
     reinterpret_cast<DL_FUNC>(&C_cudaverse_cuda_release), 1},
    {"C_cudaverse_cuda_share",
     reinterpret_cast<DL_FUNC>(&C_cudaverse_cuda_share), 1},
    {"C_cudaverse_cuda_sparse_release",
     reinterpret_cast<DL_FUNC>(&C_cudaverse_cuda_sparse_release), 1},
    {"C_cudaverse_cuda_sparse_share",
     reinterpret_cast<DL_FUNC>(&C_cudaverse_cuda_sparse_share), 1},
    {"C_cudaverse_cuda_memory_info",
     reinterpret_cast<DL_FUNC>(&C_cudaverse_cuda_memory_info), 0},
    {"C_cudaverse_cuda_memory_tracker",
     reinterpret_cast<DL_FUNC>(&C_cudaverse_cuda_memory_tracker), 1},
    {nullptr, nullptr, 0}};

extern "C" void R_init_cudaverse(DllInfo* dll) {
  R_registerRoutines(dll, nullptr, call_methods, nullptr, nullptr);
  R_useDynamicSymbols(dll, static_cast<Rboolean>(0));
  R_forceSymbols(dll, static_cast<Rboolean>(1));
}
