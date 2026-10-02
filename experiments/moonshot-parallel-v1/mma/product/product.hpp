#pragma once
#include <cuda_runtime_api.h>
#include "oracle.hpp"
#include <cstddef>
#include <cstdint>
#include <limits>
namespace cellerator::experimental::moonshot::product {
constexpr std::uint32_t padding_id = UINT32_MAX;
template<class T> struct View { T* data{}; std::size_t size{}; };
struct Inputs {
    View<const float> x, direction, coefficient;
    View<const std::uint32_t> src0, src1;
    std::uint32_t count{};
};
struct Outputs { View<float> value, jvp; };
struct Generations {
    std::uint64_t structure{}, value{}, activity{}, parameter{};
};
inline bool operator==(Generations a, Generations b) {
    return a.structure==b.structure && a.value==b.value &&
           a.activity==b.activity && a.parameter==b.parameter;
}
// Host-only metadata admission: declared spans must cover actual allocations.
cudaError_t validate_metadata(const Inputs&, const Outputs&);
class Prepared {
    Inputs inputs_{}; Outputs outputs_{}; Generations generations_{};
    int device_{-1}; cudaStream_t stream_{}; bool ready_{};
    friend cudaError_t prepare_product2(const Inputs&,const Outputs&,Generations,cudaStream_t,Prepared&);
    friend cudaError_t launch_product2(const Prepared&,Generations,cudaStream_t);
};
// Nonowning snapshot. Preparation copies indices to host and synchronizes stream
// for validation. The caller retains allocations and keeps index arrays immutable
// until every launch completes; values/parameters bind the recorded generations.
// Sentinel is internal padding only; live UINT32_MAX indices are rejected.
cudaError_t prepare_product2(const Inputs&,const Outputs&,Generations,cudaStream_t,Prepared&);
// No allocation or synchronization. Caller keeps one device and stream, checks
// asynchronous failures, and invalidates/re-prepares after structural mutation.
cudaError_t launch_product2(const Prepared&,Generations,cudaStream_t);
} // namespace
