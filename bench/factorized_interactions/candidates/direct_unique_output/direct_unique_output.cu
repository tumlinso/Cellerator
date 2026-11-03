#include "direct_unique_output.hh"

#include "factorized_interaction_numerics.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace cellerator::bench::factorized_interactions::candidates {
namespace {

namespace mc = modeling::cuda;
namespace stable_detail = ::cellerator::modeling::cuda::stable_detail;
using layout::FactorizedInteractionRecord;
using layout::FactorizedInteractionTermRecord;
using layout::FactorizedInteractionView;

constexpr std::uint64_t kCellLanes = 32;
constexpr std::uint32_t kMaxWarps = 4;
constexpr unsigned kMaxGridX = 65535;

bool checked_bytes(std::uint64_t count, std::size_t item_size,
                   std::size_t& out) noexcept {
    if (item_size && count > std::numeric_limits<std::size_t>::max() / item_size)
        return false;
    out = static_cast<std::size_t>(count) * item_size;
    return true;
}

bool checked_product(std::uint64_t a, std::uint64_t b,
                     std::uint64_t& out) noexcept {
    if (b && a > std::numeric_limits<std::uint64_t>::max() / b) return false;
    out = a * b;
    return true;
}

std::uint64_t ceil_div(std::uint64_t value, std::uint64_t divisor) noexcept {
    return value / divisor + (value % divisor != 0);
}

bool valid_dtype(memory::cuda::DType type) noexcept {
    return type == memory::cuda::DType::float16 ||
           type == memory::cuda::DType::float32;
}

std::size_t dtype_bytes(memory::cuda::DType type) noexcept {
    return type == memory::cuda::DType::float16 ? sizeof(__half) : sizeof(float);
}

struct Counts {
    std::uint64_t input_values = 0;
    std::uint64_t context_values = 0;
    std::uint64_t alpha_values = 0;
    std::uint64_t output_values = 0;
    std::uint64_t gate_values = 0;
    std::uint64_t tape_values = 0;
    std::uint64_t forward_tasks = 0;
};

bool counts_for(const FactorizedInteractionView& plan,
                mc::FactorizedInteractionProblem problem,
                Counts& out) noexcept {
    if (plan.schema != layout::factorized_interaction_schema || plan.reserved ||
        plan.mechanism_count > std::numeric_limits<std::size_t>::max() /
                                   sizeof(FactorizedInteractionRecord) ||
        plan.term_count > std::numeric_limits<std::size_t>::max() /
                              sizeof(FactorizedInteractionTermRecord)) return false;

    if (!checked_product(plan.extents.inputs, problem.channels, out.input_values) ||
        !checked_product(out.input_values, problem.cells, out.input_values) ||
        !checked_product(plan.extents.contexts, problem.cells, out.context_values) ||
        !checked_product(plan.extents.alpha_rows, problem.channels, out.alpha_values) ||
        !checked_product(plan.extents.outputs, problem.channels, out.output_values) ||
        !checked_product(out.output_values, problem.cells, out.output_values)) return false;

    if (plan.mechanism_count && problem.cells && problem.channels) {
        if (!checked_product(plan.mechanism_count, problem.cells, out.gate_values))
            return false;
        const auto cell_tiles = ceil_div(problem.cells, kCellLanes);
        const auto warps = std::min<std::uint64_t>(problem.channels, kMaxWarps);
        const auto channel_tiles = ceil_div(problem.channels, warps);
        std::uint64_t tasks = 0;
        if (!checked_product(plan.mechanism_count, cell_tiles, tasks) ||
            !checked_product(tasks, channel_tiles, out.forward_tasks)) return false;
    }
    if (!checked_product(out.gate_values, 2, out.tape_values)) return false;

    std::size_t ignored = 0;
    return checked_bytes(out.input_values, sizeof(float), ignored) &&
           checked_bytes(out.context_values, sizeof(float), ignored) &&
           checked_bytes(out.alpha_values, sizeof(float), ignored) &&
           checked_bytes(plan.extents.biases, sizeof(float), ignored) &&
           checked_bytes(plan.extents.gammas, sizeof(float), ignored) &&
           checked_bytes(out.output_values, sizeof(float), ignored) &&
           checked_bytes(out.tape_values, sizeof(float), ignored) &&
           out.forward_tasks <= static_cast<std::uint64_t>(
               std::numeric_limits<std::int64_t>::max());
}

bool view_has(memory::cuda::ConstDeviceView view, std::size_t bytes) noexcept {
    return bytes == 0 || (view.data && view.bytes >= bytes);
}
bool view_has(memory::cuda::DeviceView view, std::size_t bytes) noexcept {
    return bytes == 0 || (view.data && view.bytes >= bytes);
}
bool aligned_for(const void* pointer, std::size_t alignment) noexcept {
    return !pointer || reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0;
}
bool aligned_view(memory::cuda::ConstDeviceView view,
                  std::size_t alignment) noexcept {
    return view.bytes == 0 || aligned_for(view.data, alignment);
}
bool aligned_view(memory::cuda::DeviceView view,
                  std::size_t alignment) noexcept {
    return view.bytes == 0 || aligned_for(view.data, alignment);
}

bool bind_device(int& device, memory::cuda::ConstDeviceView view,
                 std::size_t required_bytes) noexcept {
    if (!view_has(view, required_bytes)) return false;
    if (!view.bytes) return required_bytes == 0;
    if (!view.data || view.device < 0) return false;
    if (device < 0) device = view.device;
    return device == view.device;
}
bool bind_device(int& device, memory::cuda::DeviceView view,
                 std::size_t required_bytes) noexcept {
    if (!view_has(view, required_bytes)) return false;
    if (!view.bytes) return required_bytes == 0;
    if (!view.data || view.device < 0) return false;
    if (device < 0) device = view.device;
    return device == view.device;
}

cudaError_t current_device_matches(int device) noexcept {
    if (device < 0) return cudaSuccess;
    int current = -1;
    const auto status = cudaGetDevice(&current);
    if (status != cudaSuccess) return status;
    return current == device ? cudaSuccess : cudaErrorInvalidDevice;
}

struct Region {
    std::uintptr_t begin = 0;
    std::uintptr_t end = 0;
    bool present = false;
};

bool make_region(const void* pointer, std::size_t bytes, Region& region) noexcept {
    if (!bytes) return true;
    if (!pointer) return false;
    const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
    if (bytes > std::numeric_limits<std::uintptr_t>::max() - begin) return false;
    region = {begin, begin + bytes, true};
    return true;
}

bool regions_disjoint(const Region* writable, std::size_t writable_count,
                      const Region* readonly, std::size_t readonly_count) noexcept {
    for (std::size_t i = 0; i < writable_count; ++i) {
        if (!writable[i].present) continue;
        for (std::size_t j = i + 1; j < writable_count; ++j) {
            if (writable[j].present && writable[i].begin < writable[j].end &&
                writable[j].begin < writable[i].end) return false;
        }
        for (std::size_t j = 0; j < readonly_count; ++j) {
            if (readonly[j].present && writable[i].begin < readonly[j].end &&
                readonly[j].begin < writable[i].end) return false;
        }
    }
    return true;
}

bool forward_views_valid(const FactorizedInteractionView& plan,
                         mc::FactorizedInteractionProblem problem,
                         mc::FactorizedInteractionPrecision precision,
                         const mc::FactorizedInteractionValues& values,
                         memory::cuda::DeviceView output,
                         memory::cuda::DeviceView tape,
                         memory::cuda::DeviceView scratch,
                         const Counts& counts, int& device) noexcept {
    std::size_t x_bytes = 0, z_bytes = 0, alpha_bytes = 0, bias_bytes = 0,
                gamma_bytes = 0, output_bytes = 0, tape_bytes = 0;
    if (!checked_bytes(counts.input_values, dtype_bytes(precision.activations), x_bytes) ||
        !checked_bytes(counts.context_values, dtype_bytes(precision.activations), z_bytes) ||
        !checked_bytes(counts.alpha_values, dtype_bytes(precision.parameters), alpha_bytes) ||
        !checked_bytes(plan.extents.biases, dtype_bytes(precision.parameters), bias_bytes) ||
        !checked_bytes(plan.extents.gammas, dtype_bytes(precision.parameters), gamma_bytes) ||
        !checked_bytes(counts.output_values, dtype_bytes(precision.output), output_bytes) ||
        !checked_bytes(counts.tape_values, sizeof(float), tape_bytes)) return false;

    if (!aligned_view(values.x, dtype_bytes(precision.activations)) ||
        !aligned_view(values.z, dtype_bytes(precision.activations)) ||
        !aligned_view(values.alpha, dtype_bytes(precision.parameters)) ||
        !aligned_view(values.bias, dtype_bytes(precision.parameters)) ||
        !aligned_view(values.gamma, dtype_bytes(precision.parameters)) ||
        !aligned_view(output, dtype_bytes(precision.output)) ||
        !aligned_view(tape, alignof(float)) ||
        !aligned_view(scratch, alignof(float))) return false;

    const bool active = plan.mechanism_count && problem.cells && problem.channels;
    if (active && (!plan.mechanisms || (plan.term_count && !plan.terms) ||
        !aligned_for(plan.mechanisms, alignof(FactorizedInteractionRecord)) ||
        (plan.term_count && !aligned_for(plan.terms, alignof(FactorizedInteractionTermRecord))) ||
        !plan.extents.inputs || !plan.extents.outputs ||
        !plan.extents.alpha_rows || !plan.extents.biases)) return false;

    int d = -1;
    if (!bind_device(d, values.x, active ? x_bytes : 0) ||
        !bind_device(d, values.z, active ? z_bytes : 0) ||
        !bind_device(d, values.alpha, active ? alpha_bytes : 0) ||
        !bind_device(d, values.bias, active ? bias_bytes : 0) ||
        !bind_device(d, values.gamma, active ? gamma_bytes : 0) ||
        !bind_device(d, output, output_bytes) ||
        !bind_device(d, tape, tape_bytes) ||
        !bind_device(d, scratch, 0)) return false;

    std::size_t mechanism_bytes = 0, term_bytes = 0;
    if (active && (!checked_bytes(plan.mechanism_count,
                                  sizeof(FactorizedInteractionRecord), mechanism_bytes) ||
                   !checked_bytes(plan.term_count,
                                  sizeof(FactorizedInteractionTermRecord), term_bytes)))
        return false;
    Region writable[3], readonly[7];
    if (!make_region(output.data, output.bytes, writable[0]) ||
        !make_region(tape.data, tape.bytes, writable[1]) ||
        !make_region(scratch.data, scratch.bytes, writable[2]) ||
        !make_region(values.x.data, values.x.bytes, readonly[0]) ||
        !make_region(values.z.data, values.z.bytes, readonly[1]) ||
        !make_region(values.alpha.data, values.alpha.bytes, readonly[2]) ||
        !make_region(values.bias.data, values.bias.bytes, readonly[3]) ||
        !make_region(values.gamma.data, values.gamma.bytes, readonly[4]) ||
        !make_region(active ? plan.mechanisms : nullptr, mechanism_bytes, readonly[5]) ||
        !make_region(active ? plan.terms : nullptr, term_bytes, readonly[6]) ||
        !regions_disjoint(writable, 3, readonly, 7)) return false;
    device = d;
    return true;
}

template<class Activation, class Parameter>
__device__ __forceinline__ stable_detail::SigmoidAndSlope gate_for(
    const FactorizedInteractionRecord& mechanism,
    const FactorizedInteractionTermRecord* terms,
    const Activation* z, const Parameter* bias, const Parameter* gamma,
    std::uint64_t cells, std::uint64_t cell) {
    stable_detail::Accurate logit(stable_detail::load(bias, mechanism.bias));
    for (std::uint32_t i = 0; i < mechanism.term_count; ++i) {
        const auto term = terms[mechanism.term_begin + i];
        const auto z_value = stable_detail::load(z, term.context * cells + cell);
        logit.add_product(stable_detail::load(gamma, term.gamma), z_value);
    }
    return stable_detail::sigmoid_and_slope(logit.value());
}

template<class Activation, class Parameter, class Output>
__global__ void unique_output_forward(
    const FactorizedInteractionRecord* mechanisms,
    const FactorizedInteractionTermRecord* terms,
    std::uint64_t mechanism_count, std::uint64_t cells,
    std::uint64_t channels, std::uint64_t cell_tiles,
    std::uint64_t channel_tiles, std::uint32_t warps,
    const Activation* x, const Activation* z,
    const Parameter* alpha, const Parameter* bias, const Parameter* gamma,
    Output* output, float* saved_gates, float* saved_slopes,
    std::uint64_t task_count) {
    __shared__ float gate_tile[kCellLanes];

    for (std::uint64_t task = blockIdx.x; task < task_count;
         task += gridDim.x) {
        const auto cell_tile = task % cell_tiles;
        const auto group = task / cell_tiles;
        const auto channel_tile = group % channel_tiles;
        const auto mechanism_id = group / channel_tiles;
        if (mechanism_id >= mechanism_count) continue;

        const auto mechanism = mechanisms[mechanism_id];
        const auto warp = static_cast<std::uint32_t>(threadIdx.x >> 5);
        const auto lane = static_cast<std::uint32_t>(threadIdx.x & 31);
        const auto cell = cell_tile * kCellLanes + lane;
        const auto channel = channel_tile * warps + warp;

        if (warp == 0) {
            const auto value = cell < cells
                ? gate_for(mechanism, terms, z, bias, gamma, cells, cell)
                : stable_detail::SigmoidAndSlope{0.0f, 0.0f};
            gate_tile[lane] = value.gate;
            if (channel_tile == 0 && cell < cells) {
                const auto tape_index = mechanism_id * cells + cell;
                saved_gates[tape_index] = value.gate;
                saved_slopes[tape_index] = value.slope;
            }
        }
        __syncthreads();

        if (cell < cells && channel < channels) {
            const auto x_base = channel * cells + cell;
            const auto input_base = channels * cells;
            float product = stable_detail::load(
                x, mechanism.inputs[0] * input_base + x_base) *
                stable_detail::load(x, mechanism.inputs[1] * input_base + x_base);
            if (mechanism.arity >= 3)
                product *= stable_detail::load(
                    x, mechanism.inputs[2] * input_base + x_base);
            if (mechanism.arity >= 4)
                product *= stable_detail::load(
                    x, mechanism.inputs[3] * input_base + x_base);
            const auto a = stable_detail::load(
                alpha, mechanism.alpha * channels + channel);
            const auto value = gate_tile[lane] * a * product;
            const auto output_index = mechanism.target * channels * cells + x_base;
            // Host preparation plus max_output_mechanisms==1 establishes
            // exclusive output ownership; no output atomics or reduction.
            stable_detail::store(output, output_index, value);
        }
        __syncthreads();
    }
}

template<class Activation, class Parameter, class Output>
cudaError_t launch_unique_output(
    const FactorizedInteractionView& plan,
    mc::FactorizedInteractionProblem problem,
    const mc::FactorizedInteractionValues& values,
    memory::cuda::DeviceView output, memory::cuda::DeviceView tape,
    const Counts& counts, cudaStream_t stream) noexcept {
    const auto warps = static_cast<std::uint32_t>(
        std::min<std::uint64_t>(problem.channels, kMaxWarps));
    const auto cell_tiles = ceil_div(problem.cells, kCellLanes);
    const auto channel_tiles = ceil_div(problem.channels, warps);
    const auto blocks = static_cast<unsigned>(
        std::min<std::uint64_t>(counts.forward_tasks, kMaxGridX));
    unique_output_forward<Activation, Parameter, Output>
        <<<blocks, warps * 32, 0, stream>>>(
            plan.mechanisms, plan.terms, plan.mechanism_count,
            problem.cells, problem.channels, cell_tiles, channel_tiles, warps,
            static_cast<const Activation*>(values.x.data),
            static_cast<const Activation*>(values.z.data),
            static_cast<const Parameter*>(values.alpha.data),
            static_cast<const Parameter*>(values.bias.data),
            static_cast<const Parameter*>(values.gamma.data),
            static_cast<Output*>(output.data),
            static_cast<float*>(tape.data),
            static_cast<float*>(tape.data) + counts.gate_values,
            counts.forward_tasks);
    return cudaGetLastError();
}

template<class Activation, class Parameter>
cudaError_t dispatch_unique_output(
    const FactorizedInteractionView& plan,
    mc::FactorizedInteractionProblem problem,
    const mc::FactorizedInteractionValues& values,
    mc::FactorizedInteractionPrecision precision,
    memory::cuda::DeviceView output, memory::cuda::DeviceView tape,
    const Counts& counts, cudaStream_t stream) noexcept {
    if (precision.output == memory::cuda::DType::float16)
        return launch_unique_output<Activation, Parameter, __half>(
            plan, problem, values, output, tape, counts, stream);
    return launch_unique_output<Activation, Parameter, float>(
        plan, problem, values, output, tape, counts, stream);
}

} // namespace

cudaError_t factorized_interaction_unique_output_backend(
    const FactorizedInteractionView& plan,
    mc::FactorizedInteractionProblem problem,
    DirectUniqueOutputBackend* result) noexcept {
    if (!result) return cudaErrorInvalidValue;
    mc::FactorizedInteractionBackend native{};
    const auto status = mc::factorized_interaction_backend(
        plan, problem, mc::FactorizedInteractionRoute::direct, &native);
    if (status != cudaSuccess) return status;

    const bool unique = plan.mechanism_count && problem.cells && problem.channels &&
        plan.extents.outputs && plan.mechanism_count == plan.extents.outputs &&
        plan.max_output_mechanisms == 1;
    if (unique && native == mc::FactorizedInteractionBackend::direct)
        *result = DirectUniqueOutputBackend::direct_unique_output;
    else if (native == mc::FactorizedInteractionBackend::stable)
        *result = DirectUniqueOutputBackend::stable;
    else
        *result = DirectUniqueOutputBackend::direct;
    return cudaSuccess;
}

cudaError_t factorized_interaction_unique_output_workspace(
    const FactorizedInteractionView& plan,
    mc::FactorizedInteractionProblem problem,
    mc::FactorizedInteractionWorkspace* result) noexcept {
    if (!result) return cudaErrorInvalidValue;
    DirectUniqueOutputBackend backend{};
    auto status = factorized_interaction_unique_output_backend(plan, problem, &backend);
    if (status != cudaSuccess) return status;
    status = mc::factorized_interaction_direct_workspace(plan, problem, result);
    if (status == cudaSuccess && backend == DirectUniqueOutputBackend::direct_unique_output)
        result->forward_bytes = 0;
    return status;
}

cudaError_t factorized_interaction_unique_output_forward(
    const FactorizedInteractionView& plan,
    mc::FactorizedInteractionProblem problem,
    mc::FactorizedInteractionPrecision precision,
    mc::FactorizedInteractionValues values,
    memory::cuda::DeviceView output, memory::cuda::DeviceView tape,
    memory::cuda::DeviceView scratch, cudaStream_t stream) noexcept {
    DirectUniqueOutputBackend backend{};
    auto status = factorized_interaction_unique_output_backend(plan, problem, &backend);
    if (status != cudaSuccess) return status;
    if (backend != DirectUniqueOutputBackend::direct_unique_output)
        return mc::factorized_interaction_direct_forward(
            plan, problem, precision, values, output, tape, scratch, stream);

    if (!valid_dtype(precision.activations) || !valid_dtype(precision.parameters) ||
        !valid_dtype(precision.output)) return cudaErrorInvalidValue;
    Counts counts;
    if (!counts_for(plan, problem, counts) || !counts.forward_tasks)
        return cudaErrorInvalidValue;
    int device = -1;
    if (!forward_views_valid(plan, problem, precision, values, output, tape,
                             scratch, counts, device)) return cudaErrorInvalidValue;
    status = current_device_matches(device);
    if (status != cudaSuccess) return status;

    if (precision.activations == memory::cuda::DType::float16) {
        if (precision.parameters == memory::cuda::DType::float16)
            return dispatch_unique_output<__half, __half>(
                plan, problem, values, precision, output, tape, counts, stream);
        return dispatch_unique_output<__half, float>(
            plan, problem, values, precision, output, tape, counts, stream);
    }
    if (precision.parameters == memory::cuda::DType::float16)
        return dispatch_unique_output<float, __half>(
            plan, problem, values, precision, output, tape, counts, stream);
    return dispatch_unique_output<float, float>(
        plan, problem, values, precision, output, tape, counts, stream);
}

cudaError_t factorized_interaction_unique_output_backward(
    const FactorizedInteractionView& plan,
    mc::FactorizedInteractionProblem problem,
    mc::FactorizedInteractionPrecision precision,
    mc::FactorizedInteractionValues values,
    memory::cuda::ConstDeviceView output_gradient,
    memory::cuda::ConstDeviceView tape,
    mc::FactorizedInteractionGradients gradients,
    memory::cuda::DeviceView scratch,
    mc::FactorizedInteractionGradientMode mode,
    cudaStream_t stream) noexcept {
    return mc::factorized_interaction_direct_backward(
        plan, problem, precision, values, output_gradient, tape, gradients,
        scratch, mode, stream);
}

} // namespace cellerator::bench::factorized_interactions::candidates
