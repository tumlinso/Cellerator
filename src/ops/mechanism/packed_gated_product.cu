#include <Cellerator/ops/mechanism/packed_gated_product.cuh>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_map>

namespace cellerator::ops::mechanism {
namespace {
constexpr unsigned threads = 256, warp_width = 32, warps_per_block = threads / warp_width;
constexpr unsigned full_warp = 0xffffffffu;

std::size_t checked_count(std::uint64_t a, std::uint64_t b) {
    if (b && a > std::numeric_limits<std::size_t>::max() / b)
        throw std::overflow_error("packed mechanism count overflow");
    return static_cast<std::size_t>(a * b);
}
std::vector<std::uint64_t> row_positions(const tensor::block_masked::Structure& structure,
                                        const std::vector<std::uint64_t>& batch_rows) {
    const auto& rows = structure.canonical_rows();
    std::unordered_map<std::uint64_t, std::uint64_t> physical_rows;
    physical_rows.reserve(rows.size());
    for (std::uint64_t physical = 0; physical < rows.size(); ++physical)
        if (!physical_rows.emplace(rows[physical], physical).second)
            throw std::invalid_argument("duplicate canonical source row");
    std::vector<std::uint64_t> positions;
    positions.reserve(batch_rows.size());
    for (const auto row : batch_rows) {
        const auto found = physical_rows.find(row);
        if (found == physical_rows.end()) throw std::invalid_argument("required batch row is absent");
        positions.push_back(found->second);
    }
    return positions;
}
std::vector<std::uint64_t> default_rows(const tensor::block_masked::Structure& structure) {
    auto rows = structure.canonical_rows(); std::sort(rows.begin(), rows.end()); return rows;
}

template<class Storage> __device__ float load(Storage value) { return static_cast<float>(value); }
template<> __device__ float load<std::uint16_t>(std::uint16_t value) {
    return __half2float(__ushort_as_half(value));
}
__device__ void flag(std::uint32_t* status, std::uint32_t bits) { atomicOr(status, bits); }
__device__ void add(float* destination, float value, std::uint32_t* status) {
    if (!isfinite(value)) { flag(status, packed_nonfinite); return; }
    const float previous = atomicAdd(destination, value);
    if (!isfinite(previous) || !isfinite(previous + value)) flag(status, packed_nonfinite);
}
__device__ float inclusive_prefix(float value, unsigned lane) {
    for (unsigned distance = 1; distance < warp_width; distance <<= 1) {
        const float left = __shfl_up_sync(full_warp, value, distance);
        if (lane >= distance) value = left * value;
    }
    return value;
}
__device__ float inclusive_suffix(float value, unsigned lane) {
    for (unsigned distance = 1; distance < warp_width; distance <<= 1) {
        const float right = __shfl_down_sync(full_warp, value, distance);
        if (lane + distance < warp_width) value *= right;
    }
    return value;
}
template<class Storage>
__device__ float argument(PackedGatedProductInput<Storage> input,
                           std::uint64_t row, std::uint64_t ordinal) {
    return load(input.x.values[input.argument_value_offsets[row * input.topology.argument_count + ordinal]]);
}
template<class Storage>
__device__ GateEvaluation<float> gate_for(PackedGatedProductInput<Storage> input,
                                         GatedProductGroup group, std::uint64_t row) {
    const auto condition = 2 * (row * input.topology.conditions + group.condition_index);
    return evaluate_gate(input.z[condition], input.z[condition + 1],
                         input.phi[group.parameter_index], input.tau[group.parameter_index]);
}

template<class Storage, bool SavedBackward = false>
__global__ void validate_groups(PackedGatedProductInput<Storage> input,
                                 const float* dy, PackedGatedProductSaved saved,
                                 std::uint32_t* status) {
    const std::uint64_t index = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= input.batch * input.topology.group_count) return;
    const auto row = index / input.topology.group_count;
    const auto group = input.topology.groups[index % input.topology.group_count];
    const auto topology = input.topology;
    if (group.argument_begin > topology.argument_count ||
        group.argument_count > topology.argument_count - group.argument_begin ||
        group.output_begin > topology.output_count ||
        group.output_count > topology.output_count - group.output_begin ||
        group.parameter_index >= topology.parameters || group.condition_index >= topology.conditions) {
        flag(status, packed_invalid_topology); return;
    }
    for (std::uint64_t slot = group.argument_begin; slot < group.argument_begin + group.argument_count; ++slot) {
        if (topology.arguments[slot] >= topology.coordinates) { flag(status, packed_invalid_topology); return; }
        if (input.argument_value_offsets[row * topology.argument_count + slot] >= input.x.layout.stored_values) {
            flag(status, packed_missing_value); return;
        }
    }
    for (std::uint64_t slot = group.output_begin; slot < group.output_begin + group.output_count; ++slot) {
        const auto output = topology.outputs[slot];
        if (output.coordinate >= topology.coordinates) { flag(status, packed_invalid_topology); return; }
        if (!isfinite(output.scale) || (dy && !isfinite(dy[row * topology.coordinates + output.coordinate]))) {
            flag(status, packed_nonfinite); return;
        }
    }
    const auto c = 2 * (row * topology.conditions + group.condition_index);
    const auto k = group.parameter_index;
    GateEvaluation<float> gate;
    if constexpr (SavedBackward) {
        // The prepared backward phase retains the forward gate decision. z0
        // and tau are not numerical inputs to this phase, including validation.
        gate = evaluate_gate(saved.gate_arguments[index], 0.0f, 0.0f, 0.0f);
    } else {
        if (!isfinite(input.z[c]) || !isfinite(input.z[c + 1]) ||
            !isfinite(input.phi[k]) || !isfinite(input.tau[k])) { flag(status, packed_nonfinite); return; }
        gate = gate_for(input, group, row);
    }
    if (!isfinite(gate.argument) || !isfinite(gate.value)) { flag(status, packed_nonfinite); return; }
    if (gate.excluded) return; // No participant or theta read on this declared branch.
    if constexpr (SavedBackward) {
        if (!isfinite(saved.products[index]) || !isfinite(input.z[c + 1]) || !isfinite(input.phi[k])) {
            flag(status, packed_nonfinite); return;
        }
    }
    if (!isfinite(input.theta[k])) { flag(status, packed_nonfinite); return; }
    for (std::uint64_t slot = group.argument_begin; slot < group.argument_begin + group.argument_count; ++slot)
        if (!isfinite(argument(input, row, slot))) { flag(status, packed_nonfinite); return; }
}

template<class Storage, bool Backward = false>
__global__ void validate_baseline(PackedGatedProductInput<Storage> input,
                                  const float* dy, std::uint32_t* status) {
    const std::uint64_t index = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= input.batch * input.topology.coordinates) return;
    if constexpr (Backward) {
        // The linear baseline VJP depends on beta and dy, never primal x.
        if (!isfinite(dy[index])) flag(status, packed_nonfinite);
    } else {
        const auto position = input.coordinate_value_offsets[index];
        if (position >= input.x.layout.stored_values) { flag(status, packed_missing_value); return; }
        if (!isfinite(load(input.x.values[position]))) flag(status, packed_nonfinite);
    }
}

template<class Storage, bool Backward>
__global__ void baseline(PackedGatedProductInput<Storage> input, float* output,
                          const float* dy, std::uint32_t* status) {
    const std::uint64_t index = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (atomicOr(status, 0u) || index >= input.batch * input.topology.coordinates) return;
    const float value = Backward ? dy[index] : load(input.x.values[input.coordinate_value_offsets[index]]);
    add(output + index, -input.beta * value, status);
}

template<class Storage>
__global__ void forward_groups(PackedGatedProductInput<Storage> input, float* output,
                                PackedGatedProductSaved saved, std::uint32_t* status) {
    const auto lane = threadIdx.x & (warp_width - 1);
    const std::uint64_t index = std::uint64_t(blockIdx.x) * warps_per_block + threadIdx.x / warp_width;
    const auto failed = __shfl_sync(full_warp, lane == 0 ? atomicOr(status, 0u) : 0u, 0);
    if (failed || index >= input.batch * input.topology.group_count) return;
    const auto row = index / input.topology.group_count;
    const auto group = input.topology.groups[index % input.topology.group_count];
    const auto gate = gate_for(input, group, row);
    if (gate.excluded) {
        if (lane == 0 && saved.gate_arguments) {
            saved.gate_arguments[index] = gate.argument; saved.products[index] = 0;
        }
        return;
    }
    float product = 1;
    if (group.argument_count <= warp_width) {
        const float value = lane < group.argument_count ? argument(input, row, group.argument_begin + lane) : 1.0f;
        const float prefix = inclusive_prefix(value, lane);
        product = __shfl_sync(full_warp, prefix, warp_width - 1);
    } else if (lane == 0) {
        for (std::uint64_t slot = group.argument_begin; slot < group.argument_begin + group.argument_count; ++slot)
            product *= argument(input, row, slot);
    }
    if (lane != 0) return;
    if (saved.gate_arguments) { saved.gate_arguments[index] = gate.argument; saved.products[index] = product; }
    const float contribution = gated_contribution(input.theta[group.parameter_index], gate, product);
    for (std::uint64_t slot = group.output_begin; slot < group.output_begin + group.output_count; ++slot) {
        const auto incidence = input.topology.outputs[slot];
        add(output + row * input.topology.coordinates + incidence.coordinate,
            contribution * incidence.scale, status);
    }
}

template<class Storage>
__global__ void backward_groups(PackedGatedProductInput<Storage> input, const float* dy,
                                 PackedGatedProductGradients gradients,
                                 PackedGatedProductSaved saved, std::uint32_t* status) {
    const auto lane = threadIdx.x & (warp_width - 1);
    const std::uint64_t index = std::uint64_t(blockIdx.x) * warps_per_block + threadIdx.x / warp_width;
    const auto failed = __shfl_sync(full_warp, lane == 0 ? atomicOr(status, 0u) : 0u, 0);
    if (failed || index >= input.batch * input.topology.group_count) return;
    const auto row = index / input.topology.group_count;
    const auto group = input.topology.groups[index % input.topology.group_count];
    const auto gate = saved.gate_arguments
        ? evaluate_gate(saved.gate_arguments[index], 0.0f, 0.0f, 0.0f)
        : gate_for(input, group, row);
    if (gate.excluded) return;
    float upstream = 0;
    if (lane == 0)
        for (std::uint64_t slot = group.output_begin; slot < group.output_begin + group.output_count; ++slot) {
            const auto output = input.topology.outputs[slot];
            upstream += output.scale * dy[row * input.topology.coordinates + output.coordinate];
        }
    upstream = __shfl_sync(full_warp, upstream, 0);
    float product = 1, exclusive = 1;
    if (group.argument_count <= warp_width) {
        const float value = lane < group.argument_count ? argument(input, row, group.argument_begin + lane) : 1.0f;
        const float prefix = inclusive_prefix(value, lane);
        const float suffix = inclusive_suffix(value, lane);
        const float left = __shfl_up_sync(full_warp, prefix, 1);
        const float right = __shfl_down_sync(full_warp, suffix, 1);
        exclusive = (lane == 0 ? 1.0f : left) * (lane == warp_width - 1 ? 1.0f : right);
        product = __shfl_sync(full_warp, prefix, warp_width - 1);
    } else if (lane == 0) {
        for (std::uint64_t slot = group.argument_begin; slot < group.argument_begin + group.argument_count; ++slot)
            product *= argument(input, row, slot);
    }
    if (saved.products) product = saved.products[index];
    const auto k = group.parameter_index;
    const auto c = 2 * (row * input.topology.conditions + group.condition_index);
    const auto partials = local_partials(upstream, input.theta[k], input.phi[k], input.z[c + 1], gate, product);
    if (group.argument_count <= warp_width) {
        if (lane < group.argument_count) {
            const auto coordinate = input.topology.arguments[group.argument_begin + lane];
            add(gradients.x + row * input.topology.coordinates + coordinate,
                partials.participant_factor * exclusive, status);
        }
    } else if (lane == 0) {
        // Arbitrary arity fallback, still division-free when values contain zeros.
        float prefix = 1;
        for (std::uint64_t arg = 0; arg < group.argument_count; ++arg) {
            float suffix = 1;
            for (std::uint64_t other = arg + 1; other < group.argument_count; ++other)
                suffix *= argument(input, row, group.argument_begin + other);
            add(gradients.x + row * input.topology.coordinates + input.topology.arguments[group.argument_begin + arg],
                partials.participant_factor * prefix * suffix, status);
            prefix *= argument(input, row, group.argument_begin + arg);
        }
    }
    if (lane != 0) return;
    add(gradients.z + c, partials.z0, status);
    add(gradients.z + c + 1, partials.z1, status);
    add(gradients.theta + k, partials.theta, status);
    add(gradients.phi + k, partials.phi, status);
    add(gradients.tau + k, partials.tau, status);
}

template<class Storage>
bool valid(PackedGatedProductInput<Storage> input, PackedGatedProductSaved saved) {
    const auto t = input.topology;
    if ((input.include_baseline && !std::isfinite(input.beta)) ||
        (saved.gate_arguments == nullptr) != (saved.products == nullptr)) return false;
    if (input.x.layout.stored_values && !input.x.values) return false;
    if (t.group_count && (!t.groups || !input.z || !input.theta || !input.phi || !input.tau)) return false;
    if (t.argument_count && (!t.arguments || !input.argument_value_offsets)) return false;
    if (t.output_count && !t.outputs) return false;
    if (input.include_baseline && !input.coordinate_value_offsets) return false;
    try {
        checked_count(input.batch, t.coordinates); checked_count(input.batch, t.group_count);
        checked_count(input.batch, t.argument_count); checked_count(input.batch, checked_count(t.conditions, 2));
    } catch (...) { return false; }
    return input.batch * t.group_count <= std::uint64_t(INT32_MAX) * warps_per_block &&
        input.batch * t.coordinates <= std::uint64_t(INT32_MAX) * threads;
}
unsigned grid(std::uint64_t count, unsigned width) {
    return static_cast<unsigned>(count / width + (count % width != 0));
}

struct Range { const void* pointer; std::size_t bytes, capacity, alignment; };
bool bounded(Range range) {
    if (range.capacity < range.bytes) return false;
    if (!range.bytes) return true;
    const auto address = reinterpret_cast<std::uintptr_t>(range.pointer);
    return address && address % range.alignment == 0 &&
        range.bytes <= std::numeric_limits<std::uintptr_t>::max() - address;
}
bool overlaps(Range a, Range b) {
    if (!a.bytes || !b.bytes) return false;
    const auto aa = reinterpret_cast<std::uintptr_t>(a.pointer), bb = reinterpret_cast<std::uintptr_t>(b.pointer);
    return aa < bb + b.bytes && bb < aa + a.bytes;
}
template<class Storage>
bool checked_buffers(PackedGatedProductInput<Storage> input, PackedGatedProductCapacities capacity,
                     float* output, const float* dy, PackedGatedProductGradients gradients,
                     PackedGatedProductSaved saved, std::uint32_t* status, bool backward) {
    if (!valid(input, saved)) return false;
    const auto topology = input.topology;
    const auto values_bytes = checked_count(input.x.layout.stored_values, sizeof(Storage));
    const auto coordinate_bytes = checked_count(checked_count(input.batch, topology.coordinates), sizeof(float));
    const auto condition_bytes = topology.group_count
        ? checked_count(checked_count(input.batch, checked_count(topology.conditions, 2)), sizeof(float)) : 0;
    const auto parameter_bytes = topology.group_count ? checked_count(topology.parameters, sizeof(float)) : 0;
    const auto saved_bytes = checked_count(checked_count(input.batch, topology.group_count), sizeof(float));
    const auto argument_map_bytes = checked_count(checked_count(input.batch, topology.argument_count), sizeof(std::uint64_t));
    const auto coordinate_map_bytes = input.include_baseline
        ? checked_count(checked_count(input.batch, topology.coordinates), sizeof(std::uint64_t)) : 0;
    std::array<Range, 13> reads{{
        {input.x.values, values_bytes, capacity.values, alignof(Storage)},
        {topology.groups, checked_count(topology.group_count, sizeof(GatedProductGroup)), capacity.groups, alignof(GatedProductGroup)},
        {topology.arguments, checked_count(topology.argument_count, sizeof(std::uint64_t)), capacity.arguments, alignof(std::uint64_t)},
        {topology.outputs, checked_count(topology.output_count, sizeof(OutputIncidence)), capacity.outputs, alignof(OutputIncidence)},
        {input.argument_value_offsets, argument_map_bytes, capacity.argument_value_offsets, alignof(std::uint64_t)},
        {input.coordinate_value_offsets, coordinate_map_bytes, capacity.coordinate_value_offsets, alignof(std::uint64_t)},
        {input.z, condition_bytes, capacity.z, alignof(float)},
        {input.theta, parameter_bytes, capacity.theta, alignof(float)},
        {input.phi, parameter_bytes, capacity.phi, alignof(float)},
        {input.tau, parameter_bytes, capacity.tau, alignof(float)},
        {dy, backward ? coordinate_bytes : 0, capacity.output_gradient, alignof(float)},
        {saved.gate_arguments, backward && saved.gate_arguments ? saved_bytes : 0, capacity.gate_arguments, alignof(float)},
        {saved.products, backward && saved.products ? saved_bytes : 0, capacity.products, alignof(float)}
    }};
    for (const auto read : reads) if (!bounded(read)) return false;
    std::array<Range, 8> writes{{
        {output, backward ? 0 : coordinate_bytes, capacity.output, alignof(float)},
        {gradients.x, backward ? coordinate_bytes : 0, capacity.gradient_x, alignof(float)},
        {gradients.z, backward ? condition_bytes : 0, capacity.gradient_z, alignof(float)},
        {gradients.theta, backward ? parameter_bytes : 0, capacity.gradient_theta, alignof(float)},
        {gradients.phi, backward ? parameter_bytes : 0, capacity.gradient_phi, alignof(float)},
        {gradients.tau, backward ? parameter_bytes : 0, capacity.gradient_tau, alignof(float)},
        {saved.gate_arguments, !backward && saved.gate_arguments ? saved_bytes : 0, capacity.gate_arguments, alignof(float)},
        {saved.products, !backward && saved.products ? saved_bytes : 0, capacity.products, alignof(float)}
    }};
    const Range status_range{status, sizeof(std::uint32_t), capacity.status, alignof(std::uint32_t)};
    if (!bounded(status_range)) return false;
    for (const auto read : reads) if (overlaps(status_range, read)) return false;
    for (std::size_t i = 0; i < writes.size(); ++i) {
        if (!bounded(writes[i]) || overlaps(status_range, writes[i])) return false;
        for (const auto read : reads) if (overlaps(writes[i], read)) return false;
        for (std::size_t j = 0; j < i; ++j) if (overlaps(writes[i], writes[j])) return false;
    }
    return true;
}
}

std::vector<std::uint64_t> prepare_argument_value_offsets(
    const tensor::block_masked::Structure& structure, GatedProductTopologyView topology) {
    return prepare_argument_value_offsets(structure, topology, default_rows(structure));
}
std::vector<std::uint64_t> prepare_argument_value_offsets(
    const tensor::block_masked::Structure& structure, GatedProductTopologyView topology,
    const std::vector<std::uint64_t>& batch_rows) {
    const auto rows = row_positions(structure, batch_rows);
    std::vector<std::uint64_t> offsets(checked_count(rows.size(), topology.argument_count), UINT64_MAX);
    if ((topology.group_count && !topology.groups) || (topology.argument_count && !topology.arguments))
        throw std::invalid_argument("missing host mechanism topology");
    for (std::uint64_t group_index = 0; group_index < topology.group_count; ++group_index) {
        const auto group = topology.groups[group_index];
        if (group.argument_begin > topology.argument_count || group.argument_count > topology.argument_count - group.argument_begin)
            throw std::invalid_argument("host group argument range invalid");
        for (std::uint64_t slot = group.argument_begin; slot < group.argument_begin + group.argument_count; ++slot) {
            if (topology.arguments[slot] >= topology.coordinates)
                throw std::invalid_argument("host argument coordinate invalid");
            const auto feature = structure.physical_feature(topology.arguments[slot]);
            for (std::uint64_t row = 0; row < rows.size(); ++row)
                offsets[row * topology.argument_count + slot] = structure.value_offset(rows[row], feature);
        }
    }
    return offsets;
}

std::vector<std::uint64_t> prepare_coordinate_value_offsets(
    const tensor::block_masked::Structure& structure, std::uint64_t coordinates) {
    return prepare_coordinate_value_offsets(structure, coordinates, default_rows(structure));
}
std::vector<std::uint64_t> prepare_coordinate_value_offsets(
    const tensor::block_masked::Structure& structure, std::uint64_t coordinates,
    const std::vector<std::uint64_t>& batch_rows) {
    const auto rows = row_positions(structure, batch_rows);
    std::vector<std::uint64_t> offsets(checked_count(rows.size(), coordinates));
    for (std::uint64_t coordinate = 0; coordinate < coordinates; ++coordinate) {
        const auto feature = structure.physical_feature(coordinate);
        for (std::uint64_t row = 0; row < rows.size(); ++row)
            offsets[row * coordinates + coordinate] = structure.value_offset(rows[row], feature);
    }
    return offsets;
}

template<class Storage>
cudaError_t launch_packed_forward(PackedGatedProductInput<Storage> input, float* output,
                                  PackedGatedProductSaved saved, std::uint32_t* status,
                                  cudaStream_t stream) noexcept {
    if (!input.batch || (!input.topology.group_count && !input.include_baseline)) return cudaSuccess;
    if (!valid(input, saved) || !output || !status) return cudaErrorInvalidValue;
    const auto work = input.batch * input.topology.group_count;
    const auto coordinates = input.batch * input.topology.coordinates;
    if (work) validate_groups<<<grid(work, threads), threads, 0, stream>>>(input, nullptr, {}, status);
    if (input.include_baseline && coordinates)
        validate_baseline<<<grid(coordinates, threads), threads, 0, stream>>>(input, nullptr, status);
    auto error = cudaGetLastError(); if (error != cudaSuccess) return error;
    if (input.include_baseline && coordinates)
        baseline<Storage, false><<<grid(coordinates, threads), threads, 0, stream>>>(input, output, nullptr, status);
    if (work) forward_groups<<<grid(work, warps_per_block), threads, 0, stream>>>(input, output, saved, status);
    return cudaGetLastError();
}

template<class Storage>
cudaError_t launch_packed_backward(PackedGatedProductInput<Storage> input, const float* dy,
                                   PackedGatedProductGradients gradients, PackedGatedProductSaved saved,
                                   std::uint32_t* status, cudaStream_t stream) noexcept {
    if (!input.batch || (!input.topology.group_count && !input.include_baseline)) return cudaSuccess;
    if (!valid(input, saved) || !dy || !gradients.x || !status ||
        (input.topology.group_count && (!gradients.z || !gradients.theta || !gradients.phi || !gradients.tau)))
        return cudaErrorInvalidValue;
    const auto work = input.batch * input.topology.group_count;
    const auto coordinates = input.batch * input.topology.coordinates;
    if (work) {
        if (saved.gate_arguments)
            validate_groups<Storage, true><<<grid(work, threads), threads, 0, stream>>>(input, dy, saved, status);
        else
            validate_groups<<<grid(work, threads), threads, 0, stream>>>(input, dy, {}, status);
    }
    if (input.include_baseline && coordinates)
        validate_baseline<Storage, true><<<grid(coordinates, threads), threads, 0, stream>>>(input, dy, status);
    auto error = cudaGetLastError(); if (error != cudaSuccess) return error;
    if (input.include_baseline && coordinates)
        baseline<Storage, true><<<grid(coordinates, threads), threads, 0, stream>>>(input, gradients.x, dy, status);
    if (work) backward_groups<<<grid(work, warps_per_block), threads, 0, stream>>>(input, dy, gradients, saved, status);
    return cudaGetLastError();
}

template<class Storage>
bool validate_packed_forward_buffers(PackedGatedProductInput<Storage> input,
                                     PackedGatedProductCapacities capacity, float* output,
                                     PackedGatedProductSaved saved, std::uint32_t* status) noexcept {
    if (!input.batch || (!input.topology.group_count && !input.include_baseline)) return true;
    if (!output || !status) return false;
    try {
        return checked_buffers(input, capacity, output, nullptr, {}, saved, status, false);
    } catch (...) { return false; }
}
template<class Storage>
bool validate_packed_backward_buffers(PackedGatedProductInput<Storage> input,
                                      PackedGatedProductCapacities capacity, const float* dy,
                                      PackedGatedProductGradients gradients, PackedGatedProductSaved saved,
                                      std::uint32_t* status) noexcept {
    if (!input.batch || (!input.topology.group_count && !input.include_baseline)) return true;
    if (!dy || !gradients.x || !status ||
        (input.topology.group_count && (!gradients.z || !gradients.theta || !gradients.phi || !gradients.tau))) return false;
    try {
        return checked_buffers(input, capacity, nullptr, dy, gradients, saved, status, true);
    } catch (...) { return false; }
}
template<class Storage>
cudaError_t launch_packed_forward_checked(PackedGatedProductInput<Storage> input,
                                          PackedGatedProductCapacities capacity, float* output,
                                          PackedGatedProductSaved saved, std::uint32_t* status,
                                          cudaStream_t stream) noexcept {
    if (!validate_packed_forward_buffers(input, capacity, output, saved, status)) return cudaErrorInvalidValue;
    return launch_packed_forward(input, output, saved, status, stream);
}
template<class Storage>
cudaError_t launch_packed_backward_checked(PackedGatedProductInput<Storage> input,
                                           PackedGatedProductCapacities capacity, const float* dy,
                                           PackedGatedProductGradients gradients, PackedGatedProductSaved saved,
                                           std::uint32_t* status, cudaStream_t stream) noexcept {
    if (!validate_packed_backward_buffers(input, capacity, dy, gradients, saved, status)) return cudaErrorInvalidValue;
    return launch_packed_backward(input, dy, gradients, saved, status, stream);
}

template cudaError_t launch_packed_forward<std::uint16_t>(PackedGatedProductInput<std::uint16_t>, float*, PackedGatedProductSaved, std::uint32_t*, cudaStream_t) noexcept;
template cudaError_t launch_packed_forward<float>(PackedGatedProductInput<float>, float*, PackedGatedProductSaved, std::uint32_t*, cudaStream_t) noexcept;
template cudaError_t launch_packed_backward<std::uint16_t>(PackedGatedProductInput<std::uint16_t>, const float*, PackedGatedProductGradients, PackedGatedProductSaved, std::uint32_t*, cudaStream_t) noexcept;
template cudaError_t launch_packed_backward<float>(PackedGatedProductInput<float>, const float*, PackedGatedProductGradients, PackedGatedProductSaved, std::uint32_t*, cudaStream_t) noexcept;
template cudaError_t launch_packed_forward_checked<std::uint16_t>(PackedGatedProductInput<std::uint16_t>, PackedGatedProductCapacities, float*, PackedGatedProductSaved, std::uint32_t*, cudaStream_t) noexcept;
template cudaError_t launch_packed_forward_checked<float>(PackedGatedProductInput<float>, PackedGatedProductCapacities, float*, PackedGatedProductSaved, std::uint32_t*, cudaStream_t) noexcept;
template cudaError_t launch_packed_backward_checked<std::uint16_t>(PackedGatedProductInput<std::uint16_t>, PackedGatedProductCapacities, const float*, PackedGatedProductGradients, PackedGatedProductSaved, std::uint32_t*, cudaStream_t) noexcept;
template cudaError_t launch_packed_backward_checked<float>(PackedGatedProductInput<float>, PackedGatedProductCapacities, const float*, PackedGatedProductGradients, PackedGatedProductSaved, std::uint32_t*, cudaStream_t) noexcept;
template bool validate_packed_forward_buffers<std::uint16_t>(PackedGatedProductInput<std::uint16_t>, PackedGatedProductCapacities, float*, PackedGatedProductSaved, std::uint32_t*) noexcept;
template bool validate_packed_forward_buffers<float>(PackedGatedProductInput<float>, PackedGatedProductCapacities, float*, PackedGatedProductSaved, std::uint32_t*) noexcept;
template bool validate_packed_backward_buffers<std::uint16_t>(PackedGatedProductInput<std::uint16_t>, PackedGatedProductCapacities, const float*, PackedGatedProductGradients, PackedGatedProductSaved, std::uint32_t*) noexcept;
template bool validate_packed_backward_buffers<float>(PackedGatedProductInput<float>, PackedGatedProductCapacities, const float*, PackedGatedProductGradients, PackedGatedProductSaved, std::uint32_t*) noexcept;

} // namespace cellerator::ops::mechanism
