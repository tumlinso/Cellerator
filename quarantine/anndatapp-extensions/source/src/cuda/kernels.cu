#include "bridge.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <type_traits>

namespace {

using anndatapp::device::DenseView;
using anndatapp::device::ScalarType;
using anndatapp::device::SparseLayout;
using anndatapp::device::SparseView;
using anndatapp::device::Span;

constexpr unsigned int kBlockSize = 256;
constexpr unsigned long long kMaxGridX = 0x7fffffffULL;

inline dim3 grid_for(std::uint64_t count) {
    std::uint64_t blocks = count / kBlockSize + (count % kBlockSize != 0);
    if (blocks > kMaxGridX) blocks = kMaxGridX;
    return dim3(static_cast<unsigned int>(blocks));
}

inline cudaStream_t as_stream(void* stream) {
    return reinterpret_cast<cudaStream_t>(stream);
}

inline int launch_error() {
    return static_cast<int>(cudaGetLastError());
}

inline std::uint32_t dtype_width(ScalarType dtype) {
    switch (dtype) {
        case ScalarType::boolean:
        case ScalarType::int8:
        case ScalarType::uint8: return 1;
        case ScalarType::int16:
        case ScalarType::uint16:
        case ScalarType::float16: return 2;
        case ScalarType::int32:
        case ScalarType::uint32:
        case ScalarType::float32: return 4;
        case ScalarType::int64:
        case ScalarType::uint64:
        case ScalarType::float64: return 8;
    }
    return 0;
}

inline bool is_float_dtype(ScalarType dtype) {
    return dtype == ScalarType::float16 || dtype == ScalarType::float32 ||
           dtype == ScalarType::float64;
}

inline bool is_signed_integer_dtype(ScalarType dtype) {
    return dtype == ScalarType::int8 || dtype == ScalarType::int16 ||
           dtype == ScalarType::int32 || dtype == ScalarType::int64;
}

inline bool is_unsigned_integer_dtype(ScalarType dtype) {
    return dtype == ScalarType::uint8 || dtype == ScalarType::uint16 ||
           dtype == ScalarType::uint32 || dtype == ScalarType::uint64;
}

inline bool valid_numeric_dtype(ScalarType dtype) {
    return dtype_width(dtype) != 0;
}

inline bool cast_can_fail(ScalarType source, ScalarType destination) {
    if (source == destination || source == ScalarType::boolean) return false;
    if (destination == ScalarType::boolean) return true;
    const bool source_float = is_float_dtype(source);
    const bool destination_float = is_float_dtype(destination);
    if (source_float) return !destination_float;
    if (destination_float) return false;

    const bool source_signed = is_signed_integer_dtype(source);
    const bool destination_signed = is_signed_integer_dtype(destination);
    const std::uint32_t source_bits = dtype_width(source) * 8;
    const std::uint32_t destination_bits = dtype_width(destination) * 8;
    if (destination_signed) {
        if (source_signed) return source_bits > destination_bits;
        return source_bits >= destination_bits;
    }
    if (source_signed) return true;
    return source_bits > destination_bits;
}

__device__ __forceinline__ unsigned long long dense_address(DenseView view,
                                                             unsigned long long linear,
                                                             unsigned int element_bytes) {
    const unsigned long long row = linear / view.columns;
    const unsigned long long column = linear - row * view.columns;
    const long long element_offset =
        static_cast<long long>(row) * view.row_stride +
        static_cast<long long>(column) * view.column_stride;
    unsigned long long address =
        reinterpret_cast<unsigned long long>(view.values.data) + view.origin;
    if (element_offset < 0) {
        const unsigned long long magnitude =
            static_cast<unsigned long long>(-(element_offset + 1)) + 1ULL;
        address -= magnitude * element_bytes;
    } else {
        address += static_cast<unsigned long long>(element_offset) * element_bytes;
    }
    return address;
}

template <class U>
__device__ __forceinline__ U load_bits(DenseView view, unsigned long long linear) {
    U bits;
    const auto* address = reinterpret_cast<const void*>(
        dense_address(view, linear, sizeof(U)));
    __builtin_memcpy(&bits, address, sizeof(U));
    return bits;
}

template <class U>
__device__ __forceinline__ void store_bits(DenseView view,
                                           unsigned long long linear, U bits) {
    auto* address = reinterpret_cast<void*>(dense_address(view, linear, sizeof(U)));
    __builtin_memcpy(address, &bits, sizeof(U));
}

template <class T>
__device__ __forceinline__ T* dense_at(DenseView view, unsigned long long linear) {
    return reinterpret_cast<T*>(dense_address(view, linear, sizeof(T)));
}

template <class T>
__device__ __forceinline__ T load_dense(DenseView view, unsigned long long linear) {
    return *dense_at<T>(view, linear);
}

template <>
__device__ __forceinline__ __half load_dense<__half>(DenseView view,
                                                      unsigned long long linear) {
    return __ushort_as_half(load_bits<std::uint16_t>(view, linear));
}

template <class T>
__device__ __forceinline__ void store_dense(DenseView view,
                                            unsigned long long linear, T value) {
    *dense_at<T>(view, linear) = value;
}

template <>
__device__ __forceinline__ void store_dense<__half>(DenseView view,
                                                    unsigned long long linear,
                                                    __half value) {
    store_bits<std::uint16_t>(view, linear, __half_as_ushort(value));
}

__device__ __forceinline__ void set_status(void* status, unsigned int bits) {
    atomicOr(reinterpret_cast<unsigned int*>(status), bits);
}

template <class U>
__global__ void fill_bits_kernel(DenseView view, unsigned long long bits,
                                 unsigned long long count) {
    const unsigned long long stride =
        static_cast<unsigned long long>(blockDim.x) * gridDim.x;
    unsigned long long i = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    while (i < count) {
        store_bits<U>(view, i, static_cast<U>(bits));
        if (count - i <= stride) break;
        i += stride;
    }
}

template <class T>
__device__ __forceinline__ T double_to_float(double value) {
    if constexpr (std::is_same_v<T, __half>) return __double2half(value);
    else return static_cast<T>(value);
}

template <class T>
__global__ void fill_float_kernel(DenseView view, double value,
                                  unsigned long long count) {
    const T fill_value = double_to_float<T>(value);
    const unsigned long long stride =
        static_cast<unsigned long long>(blockDim.x) * gridDim.x;
    unsigned long long i = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    while (i < count) {
        store_dense<T>(view, i, fill_value);
        if (count - i <= stride) break;
        i += stride;
    }
}

template <class U>
__global__ void copy_bits_kernel(DenseView destination, DenseView source,
                                 unsigned long long count) {
    const unsigned long long stride =
        static_cast<unsigned long long>(blockDim.x) * gridDim.x;
    unsigned long long i = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    while (i < count) {
        const U bits = load_bits<U>(source, i);
        store_bits<U>(destination, i, bits);
        if (count - i <= stride) break;
        i += stride;
    }
}

template <class T>
__device__ __forceinline__ double floating_as_double(T value) {
    if constexpr (std::is_same_v<T, __half>)
        return static_cast<double>(__half2float(value));
    else
        return static_cast<double>(value);
}

__device__ __forceinline__ bool is_finite_double(double value) {
    const unsigned long long bits =
        static_cast<unsigned long long>(__double_as_longlong(value));
    return ((bits >> 52) & 0x7ffULL) != 0x7ffULL;
}

template <class T>
__device__ __forceinline__ unsigned long long unsigned_maximum() {
    if constexpr (sizeof(T) == 8) return ~0ULL;
    else return (1ULL << (sizeof(T) * 8)) - 1ULL;
}

template <class T>
__device__ __forceinline__ long long signed_maximum() {
    if constexpr (sizeof(T) == 8) return 0x7fffffffffffffffLL;
    else return static_cast<long long>((1ULL << (sizeof(T) * 8 - 1)) - 1ULL);
}

template <class Destination, class Source>
__device__ __forceinline__ bool integer_fits(Source value) {
    constexpr bool source_signed = std::is_signed_v<Source>;
    constexpr bool destination_signed = std::is_signed_v<Destination>;
    if constexpr (source_signed) {
        const long long signed_value = static_cast<long long>(value);
        if constexpr (destination_signed) {
            const long long maximum = signed_maximum<Destination>();
            const long long minimum = -maximum - 1LL;
            return signed_value >= minimum && signed_value <= maximum;
        } else {
            return signed_value >= 0 &&
                   static_cast<unsigned long long>(signed_value) <=
                       unsigned_maximum<Destination>();
        }
    } else {
        const unsigned long long unsigned_value =
            static_cast<unsigned long long>(value);
        if constexpr (destination_signed) {
            return unsigned_value <= static_cast<unsigned long long>(
                                         signed_maximum<Destination>());
        } else {
            return unsigned_value <= unsigned_maximum<Destination>();
        }
    }
}

template <class Destination>
__device__ __forceinline__ bool floating_fits_integer(double value) {
    if (!is_finite_double(value)) return false;
    const double truncated = trunc(value);
    if constexpr (std::is_signed_v<Destination>) {
        const double bound = sizeof(Destination) == 8
                                 ? 0x1p63
                                 : static_cast<double>(1ULL << (sizeof(Destination) * 8 - 1));
        return truncated >= -bound && truncated < bound;
    } else {
        const double bound = sizeof(Destination) == 8
                                 ? 0x1p64
                                 : static_cast<double>(1ULL << (sizeof(Destination) * 8));
        return truncated >= 0.0 && truncated < bound;
    }
}

template <class Destination, class Source, bool SourceBool, bool SourceFloating>
__device__ __forceinline__ Destination to_float(Source value) {
    if constexpr (std::is_same_v<Destination, __half>) {
        if constexpr (SourceBool) {
            return __int2half_rn(value != 0 ? 1 : 0);
        } else if constexpr (SourceFloating) {
            if constexpr (std::is_same_v<Source, __half>) return value;
            else if constexpr (std::is_same_v<Source, float>) return __float2half_rn(value);
            else return __double2half(value);
        } else if constexpr (std::is_signed_v<Source>) {
            return __ll2half_rn(static_cast<long long>(value));
        } else {
            return __ull2half_rn(static_cast<unsigned long long>(value));
        }
    } else {
        if constexpr (SourceBool) {
            return static_cast<Destination>(value != 0 ? 1 : 0);
        } else if constexpr (SourceFloating && std::is_same_v<Source, __half>) {
            return static_cast<Destination>(__half2float(value));
        } else {
            return static_cast<Destination>(value);
        }
    }
}

template <class Source, bool SourceBool, bool SourceFloating>
__device__ __forceinline__ bool bool_value(Source value, unsigned char& result) {
    if constexpr (SourceBool) {
        result = value != 0 ? 1 : 0;
        return true;
    } else if constexpr (SourceFloating) {
        const double numeric = floating_as_double(value);
        if (!is_finite_double(numeric) || (numeric != 0.0 && numeric != 1.0))
            return false;
        result = numeric == 1.0 ? 1 : 0;
        return true;
    } else {
        if (value != 0 && value != 1) return false;
        result = value == 1 ? 1 : 0;
        return true;
    }
}

template <class Destination, class Source, bool SourceBool, bool SourceFloating>
__device__ __forceinline__ bool integer_value(Source value, Destination& result) {
    if constexpr (SourceBool) {
        result = static_cast<Destination>(value != 0 ? 1 : 0);
        return true;
    } else if constexpr (SourceFloating) {
        const double numeric = floating_as_double(value);
        if (!floating_fits_integer<Destination>(numeric)) return false;
        result = static_cast<Destination>(numeric);
        return true;
    } else {
        if (!integer_fits<Destination>(value)) return false;
        result = static_cast<Destination>(value);
        return true;
    }
}

template <class Source, class Destination, bool SourceBool, bool SourceFloating,
          bool DestinationBool, bool DestinationFloating>
__global__ void cast_kernel(DenseView destination, DenseView source, void* status,
                            unsigned long long count) {
    const unsigned long long stride =
        static_cast<unsigned long long>(blockDim.x) * gridDim.x;
    unsigned long long i = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    while (i < count) {
        const Source input = load_dense<Source>(source, i);
        bool valid = true;
        if constexpr (DestinationBool) {
            unsigned char converted = 0;
            valid = bool_value<Source, SourceBool, SourceFloating>(input, converted);
            *dense_at<unsigned char>(destination, i) = valid ? converted : 0;
        } else if constexpr (DestinationFloating) {
            store_dense<Destination>(
                destination, i,
                to_float<Destination, Source, SourceBool, SourceFloating>(input));
        } else {
            Destination converted = 0;
            valid = integer_value<Destination, Source, SourceBool, SourceFloating>(
                input, converted);
            store_dense<Destination>(destination, i, valid ? converted : 0);
        }
        if (!valid && status != nullptr) set_status(status, 1U);
        if (count - i <= stride) break;
        i += stride;
    }
}

// Each dispatch resolves the dtype pair before the kernel processes elements.
template <class Source, class Destination, bool SourceBool, bool SourceFloating,
          bool DestinationBool, bool DestinationFloating>
inline cudaError_t launch_cast(DenseView destination, DenseView source, void* status,
                               std::uint64_t count, void* stream) {
    cast_kernel<Source, Destination, SourceBool, SourceFloating, DestinationBool,
                DestinationFloating>
        <<<grid_for(count), kBlockSize, 0, as_stream(stream)>>>(destination, source,
                                                                status, count);
    return cudaGetLastError();
}

template <class Source, bool SourceBool, bool SourceFloating>
inline cudaError_t dispatch_cast_destination_on_stream(DenseView destination,
                                                       DenseView source,
                                                       void* status,
                                                       std::uint64_t count,
                                                       void* stream) {
#define ANNDATAPP_CAST_CASE(ENUM, TYPE, FLOATING)                                      \
    case ScalarType::ENUM:                                                          \
        return launch_cast<Source, TYPE, SourceBool, SourceFloating, false, FLOATING>( \
            destination, source, status, count, stream)
    switch (destination.dtype) {
        case ScalarType::boolean:
            return launch_cast<Source, unsigned char, SourceBool, SourceFloating,
                              true, false>(destination, source, status, count, stream);
        ANNDATAPP_CAST_CASE(int8, std::int8_t, false);
        ANNDATAPP_CAST_CASE(uint8, std::uint8_t, false);
        ANNDATAPP_CAST_CASE(int16, std::int16_t, false);
        ANNDATAPP_CAST_CASE(uint16, std::uint16_t, false);
        ANNDATAPP_CAST_CASE(int32, std::int32_t, false);
        ANNDATAPP_CAST_CASE(uint32, std::uint32_t, false);
        ANNDATAPP_CAST_CASE(int64, std::int64_t, false);
        ANNDATAPP_CAST_CASE(uint64, std::uint64_t, false);
        ANNDATAPP_CAST_CASE(float16, __half, true);
        ANNDATAPP_CAST_CASE(float32, float, true);
        ANNDATAPP_CAST_CASE(float64, double, true);
    }
#undef ANNDATAPP_CAST_CASE
    return cudaErrorInvalidValue;
}

inline cudaError_t dispatch_cast_source(DenseView destination, DenseView source,
                                        void* status, std::uint64_t count,
                                        void* stream) {
    switch (source.dtype) {
        case ScalarType::boolean:
            return dispatch_cast_destination_on_stream<std::uint8_t, true, false>(
                destination, source, status, count, stream);
        case ScalarType::int8:
            return dispatch_cast_destination_on_stream<std::int8_t, false, false>(
                destination, source, status, count, stream);
        case ScalarType::uint8:
            return dispatch_cast_destination_on_stream<std::uint8_t, false, false>(
                destination, source, status, count, stream);
        case ScalarType::int16:
            return dispatch_cast_destination_on_stream<std::int16_t, false, false>(
                destination, source, status, count, stream);
        case ScalarType::uint16:
            return dispatch_cast_destination_on_stream<std::uint16_t, false, false>(
                destination, source, status, count, stream);
        case ScalarType::int32:
            return dispatch_cast_destination_on_stream<std::int32_t, false, false>(
                destination, source, status, count, stream);
        case ScalarType::uint32:
            return dispatch_cast_destination_on_stream<std::uint32_t, false, false>(
                destination, source, status, count, stream);
        case ScalarType::int64:
            return dispatch_cast_destination_on_stream<std::int64_t, false, false>(
                destination, source, status, count, stream);
        case ScalarType::uint64:
            return dispatch_cast_destination_on_stream<std::uint64_t, false, false>(
                destination, source, status, count, stream);
        case ScalarType::float16:
            return dispatch_cast_destination_on_stream<__half, false, true>(
                destination, source, status, count, stream);
        case ScalarType::float32:
            return dispatch_cast_destination_on_stream<float, false, true>(
                destination, source, status, count, stream);
        case ScalarType::float64:
            return dispatch_cast_destination_on_stream<double, false, true>(
                destination, source, status, count, stream);
    }
    return cudaErrorInvalidValue;
}

template <class Offset, class Index>
__global__ void validate_sparse_kernel(SparseView view, void* status,
                                       unsigned long long offset_count,
                                       unsigned long long index_limit,
                                       unsigned long long work_count) {
    const unsigned long long stride =
        static_cast<unsigned long long>(blockDim.x) * gridDim.x;
    unsigned long long i = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const auto* offsets = reinterpret_cast<const Offset*>(view.offsets.data);
    const auto* indices = reinterpret_cast<const Index*>(view.indices.data);
    while (i < work_count) {
        if (i < offset_count) {
            const Offset current = offsets[i];
            bool invalid = false;
            unsigned long long current_value = 0;
            if constexpr (std::is_signed_v<Offset>) {
                if (current < 0) invalid = true;
                else current_value = static_cast<unsigned long long>(current);
            } else {
                current_value = static_cast<unsigned long long>(current);
            }
            if (current_value > view.nnz) invalid = true;
            if (i == 0 && current_value != 0) invalid = true;
            if (i + 1 == offset_count && current_value != view.nnz) invalid = true;
            if (i != 0 && current < offsets[i - 1]) invalid = true;
            if (invalid) set_status(status, 1U);
        }
        if (i < view.nnz) {
            const Index index = indices[i];
            bool invalid = false;
            unsigned long long index_value = 0;
            if constexpr (std::is_signed_v<Index>) {
                if (index < 0) invalid = true;
                else index_value = static_cast<unsigned long long>(index);
            } else {
                index_value = static_cast<unsigned long long>(index);
            }
            if (index_value >= index_limit) invalid = true;
            if (invalid) set_status(status, 2U);
        }
        if (work_count - i <= stride) break;
        i += stride;
    }
}

template <class Offset, class Index>
inline cudaError_t launch_sparse(SparseView view, void* status,
                                 unsigned long long offset_count,
                                 unsigned long long index_limit,
                                 unsigned long long work_count, void* stream) {
    validate_sparse_kernel<Offset, Index>
        <<<grid_for(work_count), kBlockSize, 0, as_stream(stream)>>>(
            view, status, offset_count, index_limit, work_count);
    return cudaGetLastError();
}

template <class Offset>
inline cudaError_t dispatch_sparse_index(SparseView view, void* status,
                                         unsigned long long offset_count,
                                         unsigned long long index_limit,
                                         unsigned long long work_count,
                                         void* stream) {
#define ANNDATAPP_INDEX_CASE(ENUM, TYPE)                                               \
    case ScalarType::ENUM:                                                          \
        return launch_sparse<Offset, TYPE>(view, status, offset_count, index_limit, \
                                           work_count, stream)
    switch (view.index_type) {
        ANNDATAPP_INDEX_CASE(int8, std::int8_t);
        ANNDATAPP_INDEX_CASE(uint8, std::uint8_t);
        ANNDATAPP_INDEX_CASE(int16, std::int16_t);
        ANNDATAPP_INDEX_CASE(uint16, std::uint16_t);
        ANNDATAPP_INDEX_CASE(int32, std::int32_t);
        ANNDATAPP_INDEX_CASE(uint32, std::uint32_t);
        ANNDATAPP_INDEX_CASE(int64, std::int64_t);
        ANNDATAPP_INDEX_CASE(uint64, std::uint64_t);
        default: return cudaErrorInvalidValue;
    }
#undef ANNDATAPP_INDEX_CASE
    return cudaErrorInvalidValue;
}

inline cudaError_t dispatch_sparse_offset(SparseView view, void* status,
                                          unsigned long long offset_count,
                                          unsigned long long index_limit,
                                          unsigned long long work_count,
                                          void* stream) {
#define ANNDATAPP_OFFSET_CASE(ENUM, TYPE)                                              \
    case ScalarType::ENUM:                                                          \
        return dispatch_sparse_index<TYPE>(view, status, offset_count, index_limit, \
                                           work_count, stream)
    switch (view.offset_type) {
        ANNDATAPP_OFFSET_CASE(int8, std::int8_t);
        ANNDATAPP_OFFSET_CASE(uint8, std::uint8_t);
        ANNDATAPP_OFFSET_CASE(int16, std::int16_t);
        ANNDATAPP_OFFSET_CASE(uint16, std::uint16_t);
        ANNDATAPP_OFFSET_CASE(int32, std::int32_t);
        ANNDATAPP_OFFSET_CASE(uint32, std::uint32_t);
        ANNDATAPP_OFFSET_CASE(int64, std::int64_t);
        ANNDATAPP_OFFSET_CASE(uint64, std::uint64_t);
        default: return cudaErrorInvalidValue;
    }
#undef ANNDATAPP_OFFSET_CASE
}

template <class ValueBits, class Index, bool Checked>
__global__ void gather_columns_kernel(DenseView destination, DenseView source,
                                      Span columns, void* status,
                                      unsigned long long count) {
    const unsigned long long stride =
        static_cast<unsigned long long>(blockDim.x) * gridDim.x;
    unsigned long long i = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const auto* selected = reinterpret_cast<const Index*>(columns.data);
    while (i < count) {
        const unsigned long long selection =
            destination.rows == 0 ? i : i % destination.columns;
        const unsigned long long row =
            destination.rows == 0 ? 0 : i / destination.columns;
        const Index selected_column = selected[selection];
        unsigned long long source_column = 0;
        bool valid = true;
        if constexpr (Checked) {
            if constexpr (std::is_signed_v<Index>) {
                if (selected_column < 0) valid = false;
                else source_column = static_cast<unsigned long long>(selected_column);
            } else {
                source_column = static_cast<unsigned long long>(selected_column);
            }
            if (source_column >= source.columns) valid = false;
        } else {
            source_column = static_cast<unsigned long long>(selected_column);
        }
        if constexpr (Checked) {
            if (!valid) set_status(status, 1U);
        }
        if (destination.rows != 0) {
            if (!valid) {
                store_bits<ValueBits>(destination, i, ValueBits{0});
            } else {
                const unsigned long long source_linear = row * source.columns + source_column;
                const ValueBits bits = load_bits<ValueBits>(source, source_linear);
                store_bits<ValueBits>(destination, i, bits);
            }
        }
        if (count - i <= stride) break;
        i += stride;
    }
}

template <class ValueBits, class Index, bool Checked>
inline cudaError_t launch_gather(DenseView destination, DenseView source,
                                 Span columns, void* status,
                                 std::uint64_t count, void* stream) {
    if (count == 0) return cudaSuccess;
    gather_columns_kernel<ValueBits, Index, Checked>
        <<<grid_for(count), kBlockSize, 0, as_stream(stream)>>>(
            destination, source, columns, status, count);
    return cudaGetLastError();
}

template <class ValueBits, bool Checked>
inline cudaError_t dispatch_gather_index(DenseView destination, DenseView source,
                                         Span columns, ScalarType index_type,
                                         void* status, std::uint64_t count,
                                         void* stream) {
#define ANNDATAPP_GATHER_INDEX_CASE(ENUM, TYPE)                                        \
    case ScalarType::ENUM:                                                          \
        return launch_gather<ValueBits, TYPE, Checked>(destination, source, columns, \
                                                       status, count, stream)
    switch (index_type) {
        ANNDATAPP_GATHER_INDEX_CASE(int8, std::int8_t);
        ANNDATAPP_GATHER_INDEX_CASE(uint8, std::uint8_t);
        ANNDATAPP_GATHER_INDEX_CASE(int16, std::int16_t);
        ANNDATAPP_GATHER_INDEX_CASE(uint16, std::uint16_t);
        ANNDATAPP_GATHER_INDEX_CASE(int32, std::int32_t);
        ANNDATAPP_GATHER_INDEX_CASE(uint32, std::uint32_t);
        ANNDATAPP_GATHER_INDEX_CASE(int64, std::int64_t);
        ANNDATAPP_GATHER_INDEX_CASE(uint64, std::uint64_t);
        default: return cudaErrorInvalidValue;
    }
#undef ANNDATAPP_GATHER_INDEX_CASE
}

inline cudaError_t dispatch_gather_width(DenseView destination, DenseView source,
                                         Span columns, ScalarType index_type,
                                         void* status, std::uint64_t count,
                                         void* stream) {
    const bool checked = status != nullptr;
    switch (destination.dtype) {
        case ScalarType::boolean:
        case ScalarType::int8:
        case ScalarType::uint8:
            return checked
                       ? dispatch_gather_index<std::uint8_t, true>(destination, source, columns,
                                                                    index_type, status, count, stream)
                       : dispatch_gather_index<std::uint8_t, false>(destination, source, columns,
                                                                     index_type, status, count, stream);
        case ScalarType::int16:
        case ScalarType::uint16:
        case ScalarType::float16:
            return checked
                       ? dispatch_gather_index<std::uint16_t, true>(destination, source, columns,
                                                                     index_type, status, count, stream)
                       : dispatch_gather_index<std::uint16_t, false>(destination, source, columns,
                                                                      index_type, status, count, stream);
        case ScalarType::int32:
        case ScalarType::uint32:
        case ScalarType::float32:
            return checked
                       ? dispatch_gather_index<std::uint32_t, true>(destination, source, columns,
                                                                     index_type, status, count, stream)
                       : dispatch_gather_index<std::uint32_t, false>(destination, source, columns,
                                                                      index_type, status, count, stream);
        case ScalarType::int64:
        case ScalarType::uint64:
        case ScalarType::float64:
            return checked
                       ? dispatch_gather_index<std::uint64_t, true>(destination, source, columns,
                                                                     index_type, status, count, stream)
                       : dispatch_gather_index<std::uint64_t, false>(destination, source, columns,
                                                                      index_type, status, count, stream);
    }
    return cudaErrorInvalidValue;
}

} // namespace

extern "C" int anndatapp_cuda_fill_bits(DenseView destination, std::uint64_t bits,
                                      void* stream) {
    const std::uint64_t count = destination.rows * destination.columns;
    const std::uint32_t width = dtype_width(destination.dtype);
    if (width == 0) return static_cast<int>(cudaErrorInvalidValue);
    if (count == 0) return static_cast<int>(cudaSuccess);
#define ANNDATAPP_FILL_BITS_CASE(TYPE)                                                \
    fill_bits_kernel<TYPE><<<grid_for(count), kBlockSize, 0, as_stream(stream)>>>( \
        destination, bits, count);                                                   \
    return launch_error()
    switch (width) {
        case 1: ANNDATAPP_FILL_BITS_CASE(std::uint8_t);
        case 2: ANNDATAPP_FILL_BITS_CASE(std::uint16_t);
        case 4: ANNDATAPP_FILL_BITS_CASE(std::uint32_t);
        case 8: ANNDATAPP_FILL_BITS_CASE(std::uint64_t);
    }
#undef ANNDATAPP_FILL_BITS_CASE
    return static_cast<int>(cudaErrorInvalidValue);
}

extern "C" int anndatapp_cuda_fill_float(DenseView destination, double value,
                                       void* stream) {
    const std::uint64_t count = destination.rows * destination.columns;
    if (destination.dtype != ScalarType::float16 &&
        destination.dtype != ScalarType::float32 &&
        destination.dtype != ScalarType::float64)
        return static_cast<int>(cudaErrorInvalidValue);
    if (count == 0) return static_cast<int>(cudaSuccess);
    switch (destination.dtype) {
        case ScalarType::float16:
            fill_float_kernel<__half><<<grid_for(count), kBlockSize, 0, as_stream(stream)>>>(
                destination, value, count);
            break;
        case ScalarType::float32:
            fill_float_kernel<float><<<grid_for(count), kBlockSize, 0, as_stream(stream)>>>(
                destination, value, count);
            break;
        case ScalarType::float64:
            fill_float_kernel<double><<<grid_for(count), kBlockSize, 0, as_stream(stream)>>>(
                destination, value, count);
            break;
        default: return static_cast<int>(cudaErrorInvalidValue);
    }
    return launch_error();
}

extern "C" int anndatapp_cuda_cast(DenseView destination, DenseView source,
                                 void* status, void* stream) {
    if (!valid_numeric_dtype(destination.dtype) || !valid_numeric_dtype(source.dtype))
        return static_cast<int>(cudaErrorInvalidValue);
    const std::uint64_t count = destination.rows * destination.columns;
    if (count == 0) return static_cast<int>(cudaSuccess);
    if (destination.dtype != source.dtype && status == nullptr &&
        cast_can_fail(source.dtype, destination.dtype))
        return static_cast<int>(cudaErrorInvalidValue);
    if (destination.dtype == source.dtype) {
        switch (dtype_width(destination.dtype)) {
            case 1:
                copy_bits_kernel<std::uint8_t><<<grid_for(count), kBlockSize, 0, as_stream(stream)>>>(
                    destination, source, count);
                return launch_error();
            case 2:
                copy_bits_kernel<std::uint16_t><<<grid_for(count), kBlockSize, 0, as_stream(stream)>>>(
                    destination, source, count);
                return launch_error();
            case 4:
                copy_bits_kernel<std::uint32_t><<<grid_for(count), kBlockSize, 0, as_stream(stream)>>>(
                    destination, source, count);
                return launch_error();
            case 8:
                copy_bits_kernel<std::uint64_t><<<grid_for(count), kBlockSize, 0, as_stream(stream)>>>(
                    destination, source, count);
                return launch_error();
            default: return static_cast<int>(cudaErrorInvalidValue);
        }
    }
    return static_cast<int>(dispatch_cast_source(destination, source, status, count, stream));
}

extern "C" int anndatapp_cuda_validate_sparse(SparseView view, void* status,
                                            void* stream) {
    if (!valid_numeric_dtype(view.dtype) || !valid_numeric_dtype(view.index_type) ||
        !valid_numeric_dtype(view.offset_type) || status == nullptr)
        return static_cast<int>(cudaErrorInvalidValue);
    if (!is_signed_integer_dtype(view.index_type) &&
        !is_unsigned_integer_dtype(view.index_type))
        return static_cast<int>(cudaErrorInvalidValue);
    if (!is_signed_integer_dtype(view.offset_type) &&
        !is_unsigned_integer_dtype(view.offset_type))
        return static_cast<int>(cudaErrorInvalidValue);
    if (view.layout != SparseLayout::csr && view.layout != SparseLayout::csc)
        return static_cast<int>(cudaErrorInvalidValue);
    const unsigned long long major = view.layout == SparseLayout::csr
                                         ? view.rows
                                         : view.columns;
    const unsigned long long minor = view.layout == SparseLayout::csr
                                         ? view.columns
                                         : view.rows;
    const unsigned long long offset_count = major + 1ULL;
    const unsigned long long work_count =
        offset_count > view.nnz ? offset_count : view.nnz;
    return static_cast<int>(dispatch_sparse_offset(view, status, offset_count,
                                                    minor, work_count, stream));
}

extern "C" int anndatapp_cuda_gather_columns(DenseView destination, DenseView source,
                                           Span columns, ScalarType index_type,
                                           void* status, void* stream) {
    if (!valid_numeric_dtype(destination.dtype) || destination.dtype != source.dtype)
        return static_cast<int>(cudaErrorInvalidValue);
    if (!is_signed_integer_dtype(index_type) &&
        !is_unsigned_integer_dtype(index_type))
        return static_cast<int>(cudaErrorInvalidValue);
    const std::uint64_t count = destination.rows == 0
                                    ? (status != nullptr ? destination.columns : 0)
                                    : destination.rows * destination.columns;
    return static_cast<int>(dispatch_gather_width(destination, source, columns,
                                                   index_type, status, count, stream));
}
