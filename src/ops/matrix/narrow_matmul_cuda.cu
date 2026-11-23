#include <Cellerator/ops/matrix/narrow_matmul_cuda.hh>
#include <Cellerator/common/real/bf16.cuh>
#include <Cellerator/common/real/fp8.cuh>
#include <cuda_runtime.h>

namespace cellerator::ops::matrix {
namespace {

constexpr unsigned block_size = 256;

#if CUDART_VERSION >= 11080
cudaDataType_t fp8_type(NarrowFormat format) {
    return format == NarrowFormat::e4m3 ? CUDA_R_8F_E4M3 : CUDA_R_8F_E5M2;
}
#endif

bool overlaps(const void *a, size_t a_bytes, const void *b, size_t b_bytes) {
    if (!a_bytes || !b_bytes) return false;
    const auto x = reinterpret_cast<uintptr_t>(a), y = reinterpret_cast<uintptr_t>(b);
    return x <= y ? y - x < a_bytes : x - y < b_bytes;
}

uint32_t alignment(const void *p) {
    const auto x = reinterpret_cast<uintptr_t>(p);
    const auto low = x & (uintptr_t(0) - x);
    return low > 256 ? 256 : uint32_t(low);
}

__global__ void decode(NarrowOperand operand, size_t count, float *out) {
    using namespace ::cellerator::real;
    for (size_t i = size_t(blockIdx.x)*blockDim.x + threadIdx.x; i < count; i += size_t(gridDim.x)*blockDim.x) {
        if (operand.format == NarrowFormat::bf16) {
            out[i] = load_bf16(static_cast<const uint16_t *>(operand.data)[i]);
        } else {
            const uint8_t bits = static_cast<const uint8_t *>(operand.data)[i];
            out[i] = operand.format == NarrowFormat::e4m3 ? load_e4m3(bits) : load_e5m2(bits);
        }
    }
}

unsigned decode_grid(size_t count) {
    const size_t blocks = (count - 1)/block_size + 1;
    return blocks > 1024 ? 1024 : unsigned(blocks);
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 610
__device__ int packed4(const int8_t *p) {
    const uint32_t bits = uint8_t(p[0]) | (uint32_t(uint8_t(p[1])) << 8)
        | (uint32_t(uint8_t(p[2])) << 16) | (uint32_t(uint8_t(p[3])) << 24);
    return int(bits);
}
#endif

__global__ void dot_partials(const int8_t *a, const int8_t *b, size_t count, int32_t *partial) {
    __shared__ int32_t sums[block_size];
    const unsigned lane = threadIdx.x;
    int32_t sum = 0;
    for (size_t i = 4*(size_t(blockIdx.x)*blockDim.x + lane); i < count; i += 4*size_t(gridDim.x)*blockDim.x) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 610
        if (count - i >= 4) {
            sum = __dp4a(packed4(a + i), packed4(b + i), sum);
            continue;
        }
#endif
        for (unsigned j = 0; j < 4 && i + j < count; ++j)
            sum += int32_t(a[i + j])*int32_t(b[i + j]);
    }
    sums[lane] = sum;
    __syncthreads();
    for (unsigned stride = block_size/2; stride; stride /= 2) {
        if (lane < stride) sums[lane] += sums[lane + stride];
        __syncthreads();
    }
    if (!lane) partial[blockIdx.x] = sums[0];
}

__global__ void finish_dot(const int32_t *partial, size_t count, int32_t *out) {
    int32_t sum = 0;
    for (size_t i = 0; i < count; ++i) sum += partial[i];
    *out = sum;
}

}

NarrowMatmulCuda::~NarrowMatmulCuda() { clear(); }

void NarrowMatmulCuda::clear() noexcept {
    ready_ = false;
    if (output_) cublasLtMatrixLayoutDestroy(output_);
    if (b_) cublasLtMatrixLayoutDestroy(b_);
    if (a_) cublasLtMatrixLayoutDestroy(a_);
    if (operation_) cublasLtMatmulDescDestroy(operation_);
    a_ = b_ = output_ = nullptr;
    operation_ = nullptr;
}

NarrowStatus NarrowMatmulCuda::select(cudaDataType_t a, cudaDataType_t b, cublasComputeType_t compute) noexcept {
    clear();
    // A and B are swapped when expressing the row-major output in column-major cuBLAS.
    const cublasOperation_t trans = CUBLAS_OP_T;
    if (cublasLtMatmulDescCreate(&operation_, compute, CUDA_R_32F) != CUBLAS_STATUS_SUCCESS
        || cublasLtMatmulDescSetAttribute(operation_, CUBLASLT_MATMUL_DESC_TRANSA,
             &trans, sizeof(trans)) != CUBLAS_STATUS_SUCCESS
        || cublasLtMatrixLayoutCreate(&a_, b, p_.k, p_.n, p_.k) != CUBLAS_STATUS_SUCCESS
        || cublasLtMatrixLayoutCreate(&b_, a, p_.k, p_.m, p_.k) != CUBLAS_STATUS_SUCCESS
        || cublasLtMatrixLayoutCreate(&output_, CUDA_R_32F, p_.n, p_.m, p_.n) != CUBLAS_STATUS_SUCCESS)
        return NarrowStatus::backend_error;

    cublasLtMatmulPreference_t preference = nullptr;
    if (cublasLtMatmulPreferenceCreate(&preference) != CUBLAS_STATUS_SUCCESS)
        return NarrowStatus::backend_error;
    const uint32_t align_a = alignment(decode_ ? static_cast<const void *>(workspace_.decoded_b) : p_.b.data);
    const uint32_t align_b = alignment(decode_ ? static_cast<const void *>(workspace_.decoded_a) : p_.a.data);
    const uint32_t align_out = alignment(p_.out);
    cublasLtMatmulHeuristicResult_t result{};
    int count = 0;
    const bool configured =
        cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &workspace_.library_bytes, sizeof(workspace_.library_bytes)) == CUBLAS_STATUS_SUCCESS
        && cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES,
            &align_a, sizeof(align_a)) == CUBLAS_STATUS_SUCCESS
        && cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
            &align_b, sizeof(align_b)) == CUBLAS_STATUS_SUCCESS
        && cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES,
            &align_out, sizeof(align_out)) == CUBLAS_STATUS_SUCCESS
        && cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES,
            &align_out, sizeof(align_out)) == CUBLAS_STATUS_SUCCESS;
    const auto status = configured
        ? cublasLtMatmulAlgoGetHeuristic(handle_, operation_, a_, b_, output_, output_,
                                         preference, 1, &result, &count)
        : CUBLAS_STATUS_INTERNAL_ERROR;
    cublasLtMatmulPreferenceDestroy(preference);
    if (status == CUBLAS_STATUS_NOT_SUPPORTED || status == CUBLAS_STATUS_ARCH_MISMATCH
        || status == CUBLAS_STATUS_INVALID_VALUE || (status == CUBLAS_STATUS_SUCCESS && count == 0))
        return NarrowStatus::unsupported;
    if (status != CUBLAS_STATUS_SUCCESS) return NarrowStatus::backend_error;
    if (result.state != CUBLAS_STATUS_SUCCESS) return NarrowStatus::unsupported;
    algorithm_ = result.algo;
    ready_ = true;
    return NarrowStatus::ok;
}

NarrowStatus NarrowMatmulCuda::prepare(cublasLtHandle_t handle, const NarrowMatmul &p,
                                        const NarrowCudaWorkspace &workspace) noexcept {
    clear();
    const auto valid = validate_narrow_matmul(p);
    if (valid != NarrowStatus::ok) return valid;
    if (!handle || (workspace.library_bytes && (!workspace.library || alignment(workspace.library) < 256)))
        return NarrowStatus::invalid;

    const bool bf16 = p.a.format == NarrowFormat::bf16;
    const size_t a_elements = size_t(p.m)*p.k, b_elements = size_t(p.n)*p.k;
    const size_t a_bytes = a_elements*(bf16 ? 2 : 1), b_bytes = b_elements*(bf16 ? 2 : 1);
    const size_t out_bytes = size_t(p.m)*p.n*sizeof(float);
    if (overlaps(workspace.library, workspace.library_bytes, p.a.data, a_bytes)
        || overlaps(workspace.library, workspace.library_bytes, p.b.data, b_bytes)
        || overlaps(workspace.library, workspace.library_bytes, p.out, out_bytes))
        return NarrowStatus::invalid;

    int device = 0, major = 0, minor = 0;
    if (cudaGetDevice(&device) != cudaSuccess
        || cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess
        || cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device) != cudaSuccess)
        return NarrowStatus::backend_error;
    handle_ = handle;
    p_ = p;
    workspace_ = workspace;
    decode_ = false;
    const int sm = major*10 + minor;
    NarrowStatus native = NarrowStatus::unsupported;
    if (bf16 && sm >= 80) native = select(CUDA_R_16BF, CUDA_R_16BF, CUBLAS_COMPUTE_32F);
#if CUDART_VERSION >= 11080
    if (!bf16 && sm >= 89) native = select(fp8_type(p.a.format), fp8_type(p.b.format), CUBLAS_COMPUTE_32F);
#endif
    if (native == NarrowStatus::ok) return native;
    clear();
    if (native == NarrowStatus::backend_error) return native;
    if (!workspace.decoded_a || workspace.decoded_a_elements < a_elements
        || !workspace.decoded_b || workspace.decoded_b_elements < b_elements)
        return NarrowStatus::insufficient_storage;
    size_t scratch_a, scratch_b;
    if (__builtin_mul_overflow(a_elements, sizeof(float), &scratch_a)
        || __builtin_mul_overflow(b_elements, sizeof(float), &scratch_b))
        return NarrowStatus::invalid;
    if (alignment(workspace.decoded_a) < alignof(float) || alignment(workspace.decoded_b) < alignof(float)
        || overlaps(workspace.decoded_a, scratch_a, workspace.decoded_b, scratch_b)
        || overlaps(workspace.decoded_a, scratch_a, p.a.data, a_bytes)
        || overlaps(workspace.decoded_a, scratch_a, p.b.data, b_bytes)
        || overlaps(workspace.decoded_b, scratch_b, p.a.data, a_bytes)
        || overlaps(workspace.decoded_b, scratch_b, p.b.data, b_bytes)
        || overlaps(workspace.decoded_a, scratch_a, p.out, out_bytes)
        || overlaps(workspace.decoded_b, scratch_b, p.out, out_bytes)
        || overlaps(workspace.decoded_a, scratch_a, workspace.library, workspace.library_bytes)
        || overlaps(workspace.decoded_b, scratch_b, workspace.library, workspace.library_bytes))
        return NarrowStatus::invalid;
    decode_ = true;
    const auto fallback = select(CUDA_R_32F, CUDA_R_32F, CUBLAS_COMPUTE_32F);
    if (fallback != NarrowStatus::ok) clear();
    return fallback;
}

NarrowStatus NarrowMatmulCuda::enqueue(cudaStream_t stream) const noexcept {
    if (!ready_) return NarrowStatus::invalid;
    const void *a = p_.a.data, *b = p_.b.data;
    if (decode_) {
        const size_t a_elements = size_t(p_.m)*p_.k, b_elements = size_t(p_.n)*p_.k;
        decode<<<decode_grid(a_elements), block_size, 0, stream>>>(p_.a, a_elements, workspace_.decoded_a);
        if (cudaGetLastError() != cudaSuccess) return NarrowStatus::backend_error;
        decode<<<decode_grid(b_elements), block_size, 0, stream>>>(p_.b, b_elements, workspace_.decoded_b);
        if (cudaGetLastError() != cudaSuccess) return NarrowStatus::backend_error;
        a = workspace_.decoded_a;
        b = workspace_.decoded_b;
    }
    const float alpha = p_.a.scale*p_.b.scale, beta = 0.0f;
    return cublasLtMatmul(handle_, operation_, &alpha, b, a_, a, b_, &beta,
                          p_.out, output_, p_.out, output_, &algorithm_,
                          workspace_.library, workspace_.library_bytes, stream) == CUBLAS_STATUS_SUCCESS
        ? NarrowStatus::ok : NarrowStatus::backend_error;
}

NarrowStatus int8_dot_cuda(const int8_t *a, const int8_t *b, size_t count, int32_t *partial,
                            size_t partial_count, int32_t *out, cudaStream_t stream) noexcept {
    if (!out || alignment(out) < alignof(int32_t) || count > int8_dot_max_count
        || (count && (!a || !b))) return NarrowStatus::invalid;
    const size_t blocks = int8_dot_cuda_partials(count);
    if (blocks && (!partial || partial_count < blocks)) return NarrowStatus::insufficient_storage;
    const size_t bytes = blocks*sizeof(int32_t);
    if (blocks && (alignment(partial) < alignof(int32_t)
        || overlaps(partial, bytes, a, count) || overlaps(partial, bytes, b, count)
        || overlaps(partial, bytes, out, sizeof(*out))
        || overlaps(out, sizeof(*out), a, count) || overlaps(out, sizeof(*out), b, count)))
        return NarrowStatus::invalid;
    if (blocks) {
        dot_partials<<<unsigned(blocks), block_size, 0, stream>>>(a, b, count, partial);
        if (cudaGetLastError() != cudaSuccess) return NarrowStatus::backend_error;
    }
    finish_dot<<<1, 1, 0, stream>>>(partial, blocks, out);
    return cudaGetLastError() == cudaSuccess ? NarrowStatus::ok : NarrowStatus::backend_error;
}

}
