#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#if defined(__has_include)
#if __has_include(<cuda_bf16.h>)
#define CELLERATOR_CORE_REAL_HAS_CUDA_BF16_HEADER 1
#include <cuda_bf16.h>
#else
#define CELLERATOR_CORE_REAL_HAS_CUDA_BF16_HEADER 0
#endif
#if __has_include(<cuda_fp8.h>)
#define CELLERATOR_CORE_REAL_HAS_CUDA_FP8_HEADER 1
#include <cuda_fp8.h>
#else
#define CELLERATOR_CORE_REAL_HAS_CUDA_FP8_HEADER 0
#endif
#else
#define CELLERATOR_CORE_REAL_HAS_CUDA_BF16_HEADER 0
#define CELLERATOR_CORE_REAL_HAS_CUDA_FP8_HEADER 0
#endif

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#if defined(__has_include)
#if __has_include(<Cellerator/core/config.cuh>)
#include <Cellerator/core/config.cuh>
#define CELLERATOR_CORE_REAL_HAS_GENERATED_CONFIG 1
#else
#define CELLERATOR_CORE_REAL_HAS_GENERATED_CONFIG 0
#endif
#else
#define CELLERATOR_CORE_REAL_HAS_GENERATED_CONFIG 0
#endif

#if !CELLERATOR_CORE_REAL_HAS_GENERATED_CONFIG
namespace cellerator::core::config {

enum class real_kind {
    f16 = 3,
    f32 = 4,
    f64 = 5,
    bf16 = 6,
    fp8_e4m3 = 7,
    fp8_e5m2 = 8
};

enum class sparse_layout_kind {
    blocked_ell,
    sliced_ell,
    csr
};

enum class training_precision {
    f32,
    mixed_f16,
    mixed_bf16,
    mixed_fp8
};

enum class gradient_clipping {
    none,
    global_norm,
    value
};

enum class host_vector_level {
    scalar,
    sse2,
    sse4_1,
    sse4_2,
    avx,
    avx2,
    avx512,
    neon,
    sve
};

enum class cuda_arch_family {
    unknown,
    volta,
    turing,
    ampere,
    ada,
    hopper,
    blackwell
};

inline constexpr int default_real_storage_code = 3;
inline constexpr int default_real_compute_code = 4;
inline constexpr int default_real_accum_code = 4;
inline constexpr real_kind default_real_storage = real_kind::f16;
inline constexpr real_kind default_real_compute = real_kind::f32;
inline constexpr real_kind default_real_accum = real_kind::f32;
inline constexpr sparse_layout_kind default_sparse_layout = sparse_layout_kind::blocked_ell;
inline constexpr training_precision default_training_precision = training_precision::f32;
inline constexpr gradient_clipping default_gradient_clipping = gradient_clipping::none;
inline constexpr double default_gradient_clip_norm = 1.0;
inline constexpr double default_gradient_clip_value = 1.0;
inline constexpr const char *default_real_storage_name = "f16";
inline constexpr const char *default_real_compute_name = "f32";
inline constexpr const char *default_real_accum_name = "f32";
inline constexpr const char *default_sparse_layout_name = "blocked_ell";
inline constexpr const char *default_training_precision_name = "f32";
inline constexpr const char *default_gradient_clipping_name = "none";
inline constexpr const char *host_cpu_arch = "unknown";
inline constexpr const char *host_cpu_model = "unknown";
inline constexpr host_vector_level detected_host_vector_level = host_vector_level::scalar;
inline constexpr const char *detected_host_vector_level_name = "scalar";
inline constexpr bool host_has_sse2 = false;
inline constexpr bool host_has_sse3 = false;
inline constexpr bool host_has_ssse3 = false;
inline constexpr bool host_has_sse41 = false;
inline constexpr bool host_has_sse42 = false;
inline constexpr bool host_has_avx = false;
inline constexpr bool host_has_avx2 = false;
inline constexpr bool host_has_avx512f = false;
inline constexpr bool host_has_avx512bw = false;
inline constexpr bool host_has_avx512vnni = false;
inline constexpr bool host_has_f16c = false;
inline constexpr bool host_has_fma = false;
inline constexpr bool host_has_neon = false;
inline constexpr bool host_has_asimd = false;
inline constexpr bool host_has_sve = false;
inline constexpr bool hardware_probe_ran = false;
inline constexpr bool hardware_probe_ok = false;
inline constexpr int detected_cuda_device_count = 0;
inline constexpr int detected_cuda_device_index = -1;
inline constexpr int detected_cuda_compute_major = 0;
inline constexpr int detected_cuda_compute_minor = 0;
inline constexpr int detected_cuda_sm = 0;
inline constexpr const char *detected_cuda_device_name = "unknown";
inline constexpr int detected_cuda_total_global_mem_mib = 0;
inline constexpr int detected_cuda_multiprocessors = 0;
inline constexpr int detected_cuda_warp_size = 0;
inline constexpr cuda_arch_family detected_cuda_arch = cuda_arch_family::unknown;
inline constexpr const char *detected_cuda_arch_family_name = "unknown";
inline constexpr bool detected_cuda_has_fp16_tensor_cores = false;
inline constexpr bool detected_cuda_has_bf16_tensor_cores = false;
inline constexpr bool detected_cuda_has_tf32_tensor_cores = false;
inline constexpr bool detected_cuda_has_fp8_tensor_cores = false;
inline constexpr bool stored_payload_precision_is_explicit = true;

} // namespace cellerator::core::config
#endif

namespace cellerator::core::real {

enum {
    value_f16 = 3,
    value_f32 = 4,
    value_f64 = 5,
    value_bf16 = 6,
    value_fp8_e4m3 = 7,
    value_fp8_e5m2 = 8
};

static constexpr int has_bf16_types = CELLERATOR_CORE_REAL_HAS_CUDA_BF16_HEADER;
static constexpr int has_fp8_types = CELLERATOR_CORE_REAL_HAS_CUDA_FP8_HEADER;

using f16_t = __half;
using f32_t = float;
using f64_t = double;

#if CELLERATOR_CORE_REAL_HAS_CUDA_BF16_HEADER
using bf16_t = __nv_bfloat16;
#endif

#if CELLERATOR_CORE_REAL_HAS_CUDA_FP8_HEADER
using fp8_e4m3_t = __nv_fp8_e4m3;
using fp8_e5m2_t = __nv_fp8_e5m2;
#endif

using storage_t = __half;
using compute_t = float;
using accum_t = float;

template<typename Storage, typename Compute = float, typename Accum = float>
struct precision {
    using storage_t = Storage;
    using compute_t = Compute;
    using accum_t = Accum;
};

using f16_precision = precision<f16_t, float, float>;
using f32_precision = precision<f32_t, float, float>;
using f64_precision = precision<f64_t, double, double>;

#if CELLERATOR_CORE_REAL_HAS_CUDA_BF16_HEADER
using bf16_precision = precision<bf16_t, float, float>;
#endif

#if CELLERATOR_CORE_REAL_HAS_CUDA_FP8_HEADER
using fp8_e4m3_precision = precision<fp8_e4m3_t, float, float>;
using fp8_e5m2_precision = precision<fp8_e5m2_t, float, float>;
#endif

template<typename T>
struct is_real_type {
    enum { value = 0 };
};

template<typename T>
struct is_real_type<const T> : is_real_type<T> {};

template<typename T>
struct is_real_type<volatile T> : is_real_type<T> {};

template<typename T>
struct is_real_type<const volatile T> : is_real_type<T> {};

template<>
struct is_real_type<__half> {
    enum { value = 1 };
};

template<>
struct is_real_type<float> {
    enum { value = 1 };
};

template<>
struct is_real_type<double> {
    enum { value = 1 };
};

#if CELLERATOR_CORE_REAL_HAS_CUDA_BF16_HEADER
template<>
struct is_real_type<bf16_t> {
    enum { value = 1 };
};
#endif

#if CELLERATOR_CORE_REAL_HAS_CUDA_FP8_HEADER
template<>
struct is_real_type<fp8_e4m3_t> {
    enum { value = 1 };
};

template<>
struct is_real_type<fp8_e5m2_t> {
    enum { value = 1 };
};
#endif

template<typename T>
struct require_real {
    using type = typename std::remove_cv<typename std::remove_reference<T>::type>::type;
    static_assert(is_real_type<type>::value, "real type required");
};

template<typename T>
struct code_of;

template<typename T>
struct code_of<const T> : code_of<T> {};

template<typename T>
struct code_of<volatile T> : code_of<T> {};

template<typename T>
struct code_of<const volatile T> : code_of<T> {};

template<typename T>
struct code_of<T &> : code_of<T> {};

template<typename T>
struct code_of<const T &> : code_of<T> {};

template<typename T>
struct code_of<T &&> : code_of<T> {};

template<>
struct code_of<__half> {
    enum { code = value_f16 };
};

template<>
struct code_of<float> {
    enum { code = value_f32 };
};

template<>
struct code_of<double> {
    enum { code = value_f64 };
};

#if CELLERATOR_CORE_REAL_HAS_CUDA_BF16_HEADER
template<>
struct code_of<bf16_t> {
    enum { code = value_bf16 };
};
#endif

#if CELLERATOR_CORE_REAL_HAS_CUDA_FP8_HEADER
template<>
struct code_of<fp8_e4m3_t> {
    enum { code = value_fp8_e4m3 };
};

template<>
struct code_of<fp8_e5m2_t> {
    enum { code = value_fp8_e5m2 };
};
#endif

template<int Code>
struct type_of;

template<>
struct type_of<value_f16> {
    using type = __half;
};

template<>
struct type_of<value_f32> {
    using type = float;
};

template<>
struct type_of<value_f64> {
    using type = double;
};

#if CELLERATOR_CORE_REAL_HAS_CUDA_BF16_HEADER
template<>
struct type_of<value_bf16> {
    using type = bf16_t;
};
#endif

#if CELLERATOR_CORE_REAL_HAS_CUDA_FP8_HEADER
template<>
struct type_of<value_fp8_e4m3> {
    using type = fp8_e4m3_t;
};

template<>
struct type_of<value_fp8_e5m2> {
    using type = fp8_e5m2_t;
};
#endif

using configured_storage_t = typename type_of<config::default_real_storage_code>::type;
using configured_compute_t = typename type_of<config::default_real_compute_code>::type;
using configured_accum_t = typename type_of<config::default_real_accum_code>::type;
using default_precision = precision<configured_storage_t, configured_compute_t, configured_accum_t>;

template<typename T>
struct type_traits {
    using storage_t = typename require_real<T>::type;
    using compute_t = float;
    using accum_t = float;
    enum {
        code = code_of<storage_t>::code,
        storage_bytes = sizeof(storage_t),
        packed_bits = sizeof(storage_t) * 8
    };
};

template<>
struct type_traits<double> {
    using storage_t = double;
    using compute_t = double;
    using accum_t = double;
    enum {
        code = value_f64,
        storage_bytes = sizeof(double),
        packed_bits = sizeof(double) * 8
    };
};

template<typename Precision>
struct precision_traits {
    using storage_t = typename require_real<typename Precision::storage_t>::type;
    using compute_t = typename Precision::compute_t;
    using accum_t = typename Precision::accum_t;
    enum {
        storage_code = code_of<storage_t>::code,
        compute_code = code_of<compute_t>::code,
        accum_code = code_of<accum_t>::code,
        storage_bytes = sizeof(storage_t),
        storage_bits = sizeof(storage_t) * 8
    };
};

template<typename T>
using require_real_t = typename require_real<T>::type;

template<typename T>
__host__ __device__ __forceinline__ constexpr int is_real_v() {
    using type = typename std::remove_cv<typename std::remove_reference<T>::type>::type;
    return is_real_type<type>::value;
}

} // namespace cellerator::core::real

namespace cellerator::core::types {

enum {
    value_u32 = 1,
    value_i32 = 2
};

static constexpr int value_f16 = real::value_f16;
static constexpr int value_f32 = real::value_f32;
static constexpr int value_f64 = real::value_f64;
static constexpr int value_bf16 = real::value_bf16;
static constexpr int value_fp8_e4m3 = real::value_fp8_e4m3;
static constexpr int value_fp8_e5m2 = real::value_fp8_e5m2;

static constexpr int has_bf16_types = real::has_bf16_types;
static constexpr int has_fp8_types = real::has_fp8_types;

using u32 = std::uint32_t;
using i32 = std::int32_t;
using u64 = std::uint64_t;
using i64 = std::int64_t;

using dim_t = u32;
using nnz_t = u32;
using idx_t = u32;
using ptr_t = u32;
using shard_idx_t = unsigned long;

using storage_value_t = real::storage_t;
using compute_value_t = real::compute_t;
using accum_value_t = real::accum_t;
using count_value_t = u32;

template<typename T>
struct value_code : real::code_of<T> {};

template<>
struct value_code<u32> {
    enum { code = value_u32 };
};

template<>
struct value_code<i32> {
    enum { code = value_i32 };
};

template<int Code>
struct value_type : real::type_of<Code> {};

template<>
struct value_type<value_u32> {
    using type = u32;
};

template<>
struct value_type<value_i32> {
    using type = i32;
};

} // namespace cellerator::core::types
