// Compiled against the pinned CUDA 12.8.1 development headers in CI. The
// production bridge intentionally does not include CUDA headers or link to
// NVIDIA libraries so that ordinary R installation works without a toolkit.
#include <cstdint>

#include <cuda.h>
#include <cublas_v2.h>

static_assert(sizeof(CUdevice) == sizeof(int), "CUdevice ABI changed");
static_assert(sizeof(CUdeviceptr) == sizeof(std::uint64_t),
              "CUdeviceptr ABI changed");
static_assert(sizeof(CUresult) == sizeof(int), "CUresult ABI changed");
static_assert(sizeof(CUdevice_attribute) == sizeof(int),
              "CUdevice_attribute ABI changed");
static_assert(sizeof(cublasStatus_t) == sizeof(int),
              "cublasStatus_t ABI changed");
static_assert(sizeof(cublasHandle_t) == sizeof(void*),
              "cublasHandle_t ABI changed");
static_assert(sizeof(cublasOperation_t) == sizeof(int),
              "cublasOperation_t ABI changed");
static_assert(sizeof(cudaDataType_t) == sizeof(int),
              "cudaDataType_t ABI changed");
static_assert(sizeof(cublasComputeType_t) == sizeof(int),
              "cublasComputeType_t ABI changed");
static_assert(sizeof(cublasGemmAlgo_t) == sizeof(int),
              "cublasGemmAlgo_t ABI changed");

static_assert(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR == 75,
              "TF32 compute-capability attribute value changed");
static_assert(CUBLAS_OP_N == 0, "cuBLAS no-transpose value changed");
static_assert(CUDA_R_32F == 0, "CUDA float32 dtype value changed");
static_assert(CUBLAS_COMPUTE_32F_FAST_TF32 == 77,
              "cuBLAS fast TF32 compute type value changed");
static_assert(CUBLAS_GEMM_DEFAULT == -1,
              "cuBLAS default GEMM algorithm value changed");

using DeviceGetAttribute = CUresult (*)(int*, CUdevice_attribute, CUdevice);
using GemmEx = cublasStatus_t (*)(
    cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int,
    const void*, const void*, cudaDataType_t, int, const void*,
    cudaDataType_t, int, const void*, void*, cudaDataType_t, int,
    cublasComputeType_t, cublasGemmAlgo_t);

// The casts fail compilation if CUDA changes either entrypoint's typed
// signature. No NVIDIA library is linked or required to run this check.
DeviceGetAttribute device_get_attribute = &cuDeviceGetAttribute;
GemmEx gemm_ex = static_cast<GemmEx>(&cublasGemmEx);
