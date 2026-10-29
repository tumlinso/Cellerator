#include <Cellerator/modeling/cuda/memory.hh>

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>

namespace cellerator::modeling::cuda {
namespace {

constexpr int kThreads = 256;
constexpr std::int64_t kMaxGridBlocks = 65535;

bool element_count(std::int64_t rows, std::int64_t columns,
                   std::size_t& result) noexcept {
    if (rows < 0 || columns < 0) return false;
    const auto r = static_cast<std::uint64_t>(rows);
    const auto c = static_cast<std::uint64_t>(columns);
    if (c != 0 && r > std::numeric_limits<std::size_t>::max() / c) return false;
    const auto count = r * c;
    if (count > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) return false;
    result = static_cast<std::size_t>(count);
    return true;
}

template<class T>
bool valid_memory(EventMemoryView<T> memory) noexcept {
    if (memory.entity_count < 0 || memory.channel_count <= 0 ||
        memory.time_constant_count != static_cast<std::size_t>(memory.channel_count)) return false;
    std::size_t values = 0;
    if (!element_count(memory.entity_count, memory.channel_count, values)) return false;
    return (values == 0 || (memory.values != nullptr && memory.value_capacity >= values)) &&
           (memory.entity_count == 0 ||
            (memory.anchor_time != nullptr && memory.anchor_capacity >= static_cast<std::size_t>(memory.entity_count))) &&
           memory.time_constant != nullptr;
}

bool capacity(const void* ptr, std::size_t actual, std::size_t required) noexcept {
    return required == 0 || (ptr != nullptr && actual >= required);
}

int blocks_for(std::int64_t count) noexcept {
    const auto blocks = count / kThreads + (count % kThreads != 0);
    return static_cast<int>(std::min<std::int64_t>(
        std::max<std::int64_t>(1, blocks), kMaxGridBlocks));
}

template<class T> __device__ float to_float(T value);
template<> __device__ float to_float(float value) { return value; }
template<> __device__ float to_float(__half value) { return __half2float(value); }
template<class T> __device__ T from_float(float value);
template<> __device__ float from_float(float value) { return value; }
template<> __device__ __half from_float(float value) { return __float2half_rn(value); }

template<class T>
__global__ void apply_kernel(T* values, float* anchor_time,
                             const float* tau, std::int64_t channels,
                             std::int64_t updates, const std::int64_t* entities,
                             const std::int64_t* offsets,
                             const std::int64_t* event_channels,
                             const float* amplitudes, const float* event_times,
                             float current_time) {
    for (std::int64_t update = blockIdx.x; update < updates; update += gridDim.x) {
        const auto entity = entities[update];
        const auto begin = offsets[update];
        const auto end = offsets[update + 1];
        const float anchor = anchor_time[entity];
        for (std::int64_t channel = threadIdx.x; channel < channels;
             channel += blockDim.x) {
            const auto index = static_cast<std::size_t>(entity) *
                                   static_cast<std::size_t>(channels) +
                               static_cast<std::size_t>(channel);
            const float decay = tau[channel];
            float value = to_float(values[index]);
            float time = anchor;
            for (std::int64_t event = begin; event < end; ++event) {
                const auto event_time = event_times[event];
                value *= __expf(-(event_time - time) / decay);
                if (event_channels[event] == channel) value += amplitudes[event];
                time = event_time;
            }
            value *= __expf(-(current_time - time) / decay);
            values[index] = from_float<T>(value);
        }
        __syncthreads();
        if (threadIdx.x == 0) anchor_time[entity] = current_time;
        __syncthreads();
    }
}

template<class T>
__global__ void read_kernel(const T* values, const float* anchor_time,
                            const float* tau, std::int64_t channels,
                            const std::int64_t* entities, std::int64_t reads,
                            float read_time, T* output, std::size_t elements) {
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < elements;
         index += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        const auto channel = static_cast<std::int64_t>(index % static_cast<std::size_t>(channels));
        const auto row = static_cast<std::int64_t>(index / static_cast<std::size_t>(channels));
        const auto entity = entities[row];
        const auto source = static_cast<std::size_t>(entity) *
                                static_cast<std::size_t>(channels) +
                            static_cast<std::size_t>(channel);
        const float elapsed = read_time - anchor_time[entity];
        const float value = to_float(values[source]) * __expf(-elapsed / tau[channel]);
        output[index] = from_float<T>(value);
    }
}

template<class T>
cudaError_t apply(EventMemoryView<T> memory, std::int64_t updates,
                  const std::int64_t* entities, const std::int64_t* offsets,
                  const std::int64_t* event_channels, const float* amplitudes,
                  const float* event_times, std::int64_t event_count,
                  float current_time, cudaStream_t stream) noexcept {
    if (!valid_memory(memory) || updates < 0 || event_count < 0 || !std::isfinite(current_time))
        return cudaErrorInvalidValue;
    if (updates == 0) return cudaSuccess;
    if (entities == nullptr || offsets == nullptr ||
        (event_count > 0 && (event_channels == nullptr || amplitudes == nullptr || event_times == nullptr)))
        return cudaErrorInvalidValue;
    apply_kernel<<<blocks_for(updates), kThreads, 0, stream>>>(
        memory.values, memory.anchor_time, memory.time_constant, memory.channel_count,
        updates, entities, offsets, event_channels, amplitudes, event_times, current_time);
    return cudaGetLastError();
}

template<class T>
cudaError_t read(EventMemoryView<T> memory, const std::int64_t* entities,
                 std::int64_t reads, float time, T* output,
                 std::size_t output_capacity, cudaStream_t stream) noexcept {
    if (!valid_memory(memory) || reads < 0 || !std::isfinite(time)) return cudaErrorInvalidValue;
    std::size_t elements = 0;
    if (!element_count(reads, memory.channel_count, elements) ||
        !capacity(entities, entities ? static_cast<std::size_t>(reads) : 0,
                  static_cast<std::size_t>(reads)) ||
        !capacity(output, output_capacity, elements)) return cudaErrorInvalidValue;
    if (elements == 0) return cudaSuccess;
    read_kernel<<<blocks_for(static_cast<std::int64_t>(elements)), kThreads, 0, stream>>>(
        memory.values, memory.anchor_time, memory.time_constant, memory.channel_count,
        entities, reads, time, output, elements);
    return cudaGetLastError();
}

}  // namespace

cudaError_t apply_events_fp16(EventMemoryView<__half> m, std::int64_t u,
                              const std::int64_t* e, const std::int64_t* o,
                              const std::int64_t* c, const float* a, const float* t,
                              std::int64_t n, float now, cudaStream_t s) noexcept {
    return apply(m, u, e, o, c, a, t, n, now, s);
}
cudaError_t apply_events_fp32(EventMemoryView<float> m, std::int64_t u,
                              const std::int64_t* e, const std::int64_t* o,
                              const std::int64_t* c, const float* a, const float* t,
                              std::int64_t n, float now, cudaStream_t s) noexcept {
    return apply(m, u, e, o, c, a, t, n, now, s);
}
cudaError_t read_traces_fp16(EventMemoryView<__half> m, const std::int64_t* e,
                             std::int64_t n, float t, __half* o,
                             std::size_t c, cudaStream_t s) noexcept {
    return read(m, e, n, t, o, c, s);
}
cudaError_t read_traces_fp32(EventMemoryView<float> m, const std::int64_t* e,
                             std::int64_t n, float t, float* o,
                             std::size_t c, cudaStream_t s) noexcept {
    return read(m, e, n, t, o, c, s);
}

}  // namespace cellerator::modeling::cuda
