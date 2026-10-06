#pragma once
#include <Cellerator/execution/program/program_v2.h>
#include <Cellerator/execution/identity.hh>
#include <cuda_runtime_api.h>
#include <cstdint>
namespace cellerator::compute::native_numeric {
enum class device_representation : std::uint32_t { f16 = 3, f32 = 4 };
struct resident_vector { void* data=nullptr; std::uint64_t elements=0; device_representation representation=device_representation::f32; int device_ordinal=0; execution::value_generation generation{}; };
enum class linear_kind : std::uint32_t { copy, axpby, weighted_sum4 };
struct linear_stage { linear_kind kind=linear_kind::copy; std::uint64_t elements=0; device_representation representation=device_representation::f32; float alpha=1.0f; float beta=0.0f; };
cudaError_t allocate(resident_vector*,std::uint64_t,device_representation,int) noexcept;
cudaError_t release(resident_vector*) noexcept;
cudaError_t upload(resident_vector&,const void*,std::uint64_t,execution::value_generation,cudaStream_t) noexcept;
cudaError_t download(const resident_vector&,void*,std::uint64_t,cudaStream_t) noexcept;
cudaError_t reset(resident_vector&,float,cudaStream_t) noexcept;
// Enqueue contiguous FP32 operations on the active device and supplied stream.
// No allocation, synchronization, active-device change, or generation update occurs.
// The caller retains buffers through stream completion and supplies valid element
// capacities. Inputs may alias each other; output may overlap neither input.
// Valid zero-length bindings are no-ops and may have null pointers. Nonempty
// pointers must name device or same-device managed storage. Length/device/type,
// range overflow, pointer, and stream-device mismatches are rejected.
// Invalid metadata returns InvalidValue/InvalidDevice/InvalidDevicePointer;
// otherwise CUDA stream/launch errors are propagated.
cudaError_t enqueue_elementwise_multiply(const resident_vector&,
    const resident_vector&,resident_vector&,cudaStream_t) noexcept;
cudaError_t enqueue_axpby(float,const resident_vector&,float,
    const resident_vector&,resident_vector&,cudaStream_t) noexcept;
execution::program::prepared_stage_v2 make_linear_stage(std::uint64_t,std::uint64_t,const linear_stage*) noexcept;
}
