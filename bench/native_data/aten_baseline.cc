#include "aten_baseline.hh"

#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime_api.h>
#include <torch/csrc/autograd/autograd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iterator>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

namespace cellerator::bench::native_data {
namespace {

using clock_type = std::chrono::steady_clock;
using at::Tensor;

void cuda_check(cudaError_t status, const char* action) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(action) + ": " + cudaGetErrorString(status));
}

double elapsed_ms(clock_type::time_point start) {
    return std::chrono::duration<double, std::milli>(clock_type::now() - start).count();
}

void synchronize(int device) {
    cuda_check(cudaSetDevice(device), "cudaSetDevice");
    cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
}

template<class Function>
double timed_stage(const std::array<int, 2>& devices, Function&& function) {
    const auto start = clock_type::now();
    function();
    synchronize(devices[0]);
    synchronize(devices[1]);
    return elapsed_ms(start);
}

Tensor cpu_tensor(const std::vector<float>& values, std::vector<std::int64_t> shape) {
    if (values.empty()) return at::empty(shape, at::TensorOptions().dtype(at::kFloat).device(at::kCPU));
    return at::from_blob(const_cast<float*>(values.data()), shape,
                         at::TensorOptions().dtype(at::kFloat).device(at::kCPU)).clone();
}

Tensor cpu_index_tensor(const std::vector<std::int64_t>& values,
                        std::vector<std::int64_t> shape) {
    if (values.empty()) return at::empty(shape, at::TensorOptions().dtype(at::kLong).device(at::kCPU));
    return at::from_blob(const_cast<std::int64_t*>(values.data()), shape,
                         at::TensorOptions().dtype(at::kLong).device(at::kCPU)).clone();
}

struct HostShard {
    std::size_t groups = 0;
    std::size_t outputs = 0;
    std::size_t argument_width = 0;
    bool has_padding = false;
    std::vector<std::int64_t> argument_indices;
    std::vector<std::int64_t> active_group_indices;
    std::vector<std::int64_t> parameter_indices;
    std::vector<std::int64_t> condition_indices;
    std::vector<std::int64_t> output_groups;
    std::vector<std::int64_t> output_coordinates;
    std::vector<float> output_scales;
};

HostShard pack_shard(const Workload& workload, int owner) {
    HostShard packed;
    for (const auto count : workload.argument_count)
        packed.argument_width = std::max(packed.argument_width, static_cast<std::size_t>(count));
    std::vector<std::int64_t> local_group(workload.groups(), -1);
    for (std::size_t group = 0; group < workload.groups(); ++group) {
        if (workload.group_owner[group] != owner) continue;
        local_group[group] = static_cast<std::int64_t>(packed.groups++);
        packed.parameter_indices.push_back(static_cast<std::int64_t>(workload.parameter_index[group]));
        packed.condition_indices.push_back(static_cast<std::int64_t>(workload.condition_index[group]));
        packed.has_padding = packed.has_padding ||
            workload.argument_count[group] != packed.argument_width;
        const auto parameter = workload.parameter_index[group];
        const auto condition = workload.condition_index[group];
        const float gate_argument = workload.z[condition * 2] +
            workload.phi[parameter] * workload.z[condition * 2 + 1] -
            workload.tau[parameter];
        if (gate_argument > 0.0f)
            packed.active_group_indices.push_back(local_group[group]);
        for (std::size_t slot = 0; slot < packed.argument_width; ++slot) {
            const bool valid = slot < workload.argument_count[group];
            // The sentinel addresses a dedicated constant-one feature appended
            // during evaluation. It avoids reading unrelated biological data
            // for padding in variable-arity groups.
            const auto coordinate = valid
                ? workload.argument_coordinate[workload.argument_begin[group] + slot]
                : workload.coordinates;
            packed.argument_indices.push_back(static_cast<std::int64_t>(coordinate));
        }
    }
    for (std::size_t group = 0; group < workload.groups(); ++group) {
        if (local_group[group] < 0) continue;
        for (std::size_t slot = 0; slot < workload.output_count[group]; ++slot) {
            const auto incidence = workload.output_begin[group] + slot;
            packed.output_groups.push_back(local_group[group]);
            packed.output_coordinates.push_back(static_cast<std::int64_t>(workload.output_coordinate[incidence]));
            packed.output_scales.push_back(workload.output_scale[incidence]);
        }
    }
    packed.outputs = packed.output_coordinates.size();
    return packed;
}

struct Shard {
    int device = -1;
    std::size_t groups = 0;
    bool has_padding = false;
    Tensor x_storage;
    Tensor z_values;
    Tensor theta;
    Tensor phi;
    Tensor tau;
    Tensor upstream;
    Tensor argument_indices;
    Tensor active_group_indices;
    Tensor parameter_indices;
    Tensor condition_indices;
    Tensor output_groups;
    Tensor output_coordinates;
    Tensor output_scales;
};

struct HostShardTensors {
    std::size_t groups = 0;
    bool has_padding = false;
    Tensor x_storage;
    Tensor z_values;
    Tensor theta;
    Tensor phi;
    Tensor tau;
    Tensor upstream;
    Tensor argument_indices;
    Tensor active_group_indices;
    Tensor parameter_indices;
    Tensor condition_indices;
    Tensor output_groups;
    Tensor output_coordinates;
    Tensor output_scales;
};

HostShardTensors materialize_host_shard(const HostShard& packed, const Workload& workload,
                                        unsigned slot) {
    const auto f = static_cast<std::int64_t>(workload.coordinates);
    const auto c = static_cast<std::int64_t>(workload.conditions);
    const auto p = static_cast<std::int64_t>(workload.parameters);
    const auto b = static_cast<std::int64_t>(workload.batch);
    HostShardTensors host;
    host.groups = packed.groups;
    host.has_padding = packed.has_padding;
    const auto half_cpu = at::TensorOptions().dtype(at::kHalf).device(at::kCPU);
    const auto float_cpu = at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
    host.x_storage = slot == 0 ? cpu_tensor(workload.x, {b, f}).to(at::kHalf)
                               : at::empty({b, f}, half_cpu);
    host.z_values = slot == 1 ? cpu_tensor(workload.z, {b, c, 2})
                              : at::empty({b, c, 2}, float_cpu);
    host.theta = slot == 0 ? cpu_tensor(workload.theta, {p}) : at::empty({p}, float_cpu);
    host.phi = slot == 0 ? cpu_tensor(workload.phi, {p}) : at::empty({p}, float_cpu);
    host.tau = slot == 0 ? cpu_tensor(workload.tau, {p}) : at::empty({p}, float_cpu);
    host.upstream = slot == 0 ? cpu_tensor(workload.upstream, {b, f})
                              : at::empty({b, f}, float_cpu);
    host.argument_indices = cpu_index_tensor(
        packed.argument_indices,
        {static_cast<std::int64_t>(packed.groups), static_cast<std::int64_t>(packed.argument_width)});
    host.active_group_indices = cpu_index_tensor(
        packed.active_group_indices,
        {static_cast<std::int64_t>(packed.active_group_indices.size())});
    host.parameter_indices = cpu_index_tensor(packed.parameter_indices,
                                               {static_cast<std::int64_t>(packed.groups)});
    host.condition_indices = cpu_index_tensor(packed.condition_indices,
                                               {static_cast<std::int64_t>(packed.groups)});
    host.output_groups = cpu_index_tensor(packed.output_groups,
                                           {static_cast<std::int64_t>(packed.outputs)});
    host.output_coordinates = cpu_index_tensor(packed.output_coordinates,
                                                {static_cast<std::int64_t>(packed.outputs)});
    host.output_scales = cpu_tensor(packed.output_scales,
                                     {static_cast<std::int64_t>(packed.outputs)});
    return host;
}

Shard upload_shard(const HostShardTensors& host, int device, unsigned slot) {
    c10::cuda::CUDAGuard guard(device);
    const auto device_spec = at::Device(at::kCUDA, device);
    const auto zero_on_device = [&](const Tensor& source) {
        return at::zeros(source.sizes(), source.options().device(device_spec));
    };
    Shard shard;
    shard.device = device;
    shard.groups = host.groups;
    shard.has_padding = host.has_padding;
    // Match the semantic source homes: expression and gate parameters start
    // on device 0; regulators start on device 1; cotangents start on device 0.
    // The other shard allocates its destination but receives the source over
    // an explicit peer/staged ATen copy on its first use.
    shard.x_storage = slot == 0 ? host.x_storage.to(device_spec) : zero_on_device(host.x_storage);
    shard.z_values = slot == 1 ? host.z_values.to(device_spec) : zero_on_device(host.z_values);
    shard.theta = slot == 0 ? host.theta.to(device_spec) : zero_on_device(host.theta);
    shard.phi = slot == 0 ? host.phi.to(device_spec) : zero_on_device(host.phi);
    shard.tau = slot == 0 ? host.tau.to(device_spec) : zero_on_device(host.tau);
    shard.upstream = zero_on_device(host.upstream);
    shard.argument_indices = host.argument_indices.to(device_spec);
    shard.active_group_indices = host.active_group_indices.to(device_spec);
    shard.parameter_indices = host.parameter_indices.to(device_spec);
    shard.condition_indices = host.condition_indices.to(device_spec);
    shard.output_groups = host.output_groups.to(device_spec);
    shard.output_coordinates = host.output_coordinates.to(device_spec);
    shard.output_scales = host.output_scales.to(device_spec);
    shard.theta.requires_grad_(true);
    shard.z_values.requires_grad_(true);
    shard.phi.requires_grad_(true);
    shard.tau.requires_grad_(true);
    return shard;
}

struct ShardEvaluation {
    Tensor output;
    Tensor x;
    Tensor z;
    std::array<Tensor, 5> gradients;
};

ShardEvaluation forward_shard(Shard& shard, std::size_t coordinates,
                              float baseline_scale, bool include_baseline) {
    c10::cuda::CUDAGuard guard(shard.device);
    const auto f = static_cast<std::int64_t>(coordinates);
    auto x = shard.x_storage.to(at::kFloat);
    x.requires_grad_(true);
    auto z = shard.z_values;

    const auto conditions = z.select(2, 0).index_select(1, shard.condition_indices);
    const auto second_conditions = z.select(2, 1).index_select(1, shard.condition_indices);
    const auto phi = shard.phi.index_select(0, shard.parameter_indices).unsqueeze(0);
    const auto tau = shard.tau.index_select(0, shard.parameter_indices).unsqueeze(0);
    const auto argument = conditions + phi * second_conditions - tau;
    const auto& active_group = shard.active_group_indices;

    Tensor mechanism_output;
    if (active_group.numel() == 0 || shard.groups == 0 || shard.output_groups.numel() == 0) {
        mechanism_output = at::zeros({1, f}, x.options());
    } else {
        const auto local_group = active_group;
        auto gather_x = shard.has_padding ? at::cat({x, at::ones({1, 1}, x.options())}, 1) : x;
        const auto selected_arguments = shard.argument_indices.index_select(0, local_group);
        const auto selected_x = gather_x.reshape({-1}).index_select(0, selected_arguments.reshape({-1}));
        const auto products = selected_x.view({active_group.size(0), shard.argument_indices.size(1)}).prod(1);
        const auto selected_argument = argument.index_select(1, local_group).reshape({-1});
        const auto selected_theta = shard.theta.index_select(
            0, shard.parameter_indices.index_select(0, local_group));
        const auto contribution = selected_theta * selected_argument.square() * products;
        const auto all_terms = at::zeros(
            {static_cast<std::int64_t>(shard.groups)}, x.options()).index_copy(0, local_group, contribution);
        const auto output_terms = all_terms.view({1, static_cast<std::int64_t>(shard.groups)})
                                     .index_select(1, shard.output_groups);
        const auto scaled_output = output_terms * shard.output_scales.view({1, -1});
        mechanism_output = at::zeros({1, f}, x.options()).index_add(
            1, shard.output_coordinates, scaled_output);
    }

    auto output = include_baseline ? mechanism_output + baseline_scale * x : mechanism_output;
    ShardEvaluation result;
    result.output = output;
    result.x = x;
    result.z = z;
    return result;
}

void backward_shard(Shard& shard, ShardEvaluation& evaluation) {
    c10::cuda::CUDAGuard guard(shard.device);
    const auto& output = evaluation.output;
    if (!output.requires_grad()) {
        evaluation.gradients = {at::zeros_like(evaluation.x), at::zeros_like(evaluation.z),
                                at::zeros_like(shard.theta), at::zeros_like(shard.phi),
                                at::zeros_like(shard.tau)};
        return;
    }
    const auto gradients = torch::autograd::grad(
        {output}, {evaluation.x, evaluation.z, shard.theta, shard.phi, shard.tau}, {shard.upstream},
        false, false, true);
    for (std::size_t index = 0; index < gradients.size(); ++index)
        evaluation.gradients[index] = gradients[index].defined()
            ? gradients[index]
            : at::zeros_like(index == 0 ? evaluation.x : index == 1 ? evaluation.z :
                             index == 2 ? shard.theta : index == 3 ? shard.phi : shard.tau);
}

Tensor transfer_to(const Tensor& source, int device) {
    return source.to(at::Device(at::kCUDA, device));
}

std::vector<float> export_float(const Tensor& tensor) {
    auto host = tensor.detach().to(at::Device(at::kCPU), at::kFloat).contiguous();
    std::vector<float> result(static_cast<std::size_t>(host.numel()));
    if (!result.empty()) std::copy_n(host.data_ptr<float>(), result.size(), result.data());
    return result;
}

std::uint64_t allocated_peak(int device) {
    const auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(device);
    return static_cast<std::uint64_t>(stats.allocated_bytes[0].peak);
}

std::uint64_t reserved_peak(int device) {
    const auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(device);
    return static_cast<std::uint64_t>(stats.reserved_bytes[0].peak);
}

void reset_peak_stats(int device) {
    c10::cuda::CUDACachingAllocator::resetPeakStats(device);
}

} // namespace

struct AtenTwoDeviceBaseline::Impl {
    const Workload& workload;
    std::array<int, 2> devices;
    Shard shards[2];
    Tensor cotangent_stage;
    TimingBreakdown preparation_metrics;
    bool forward_sources_replicated = false;

    Impl(const Workload& w, int device0, int device1) : workload(w), devices{{device0, device1}} {
        if (device0 < 0 || device1 < 0 || device0 == device1)
            throw std::invalid_argument("ATen baseline requires two distinct visible CUDA devices");
        if (workload.batch != 1 || !workload.groups())
            throw std::invalid_argument("ATen baseline currently supports one non-empty sample");
        if (workload.group_owner.size() != workload.groups())
            throw std::invalid_argument("ATen baseline group ownership extent mismatch");
        int device_count = 0;
        cuda_check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
        if (device_count <= std::max(device0, device1))
            throw std::invalid_argument("selected ATen baseline CUDA device is not visible");

        const auto prep_start = clock_type::now();
        const HostShard host0 = pack_shard(workload, 0);
        const HostShard host1 = pack_shard(workload, 1);
        const HostShardTensors packed0 = materialize_host_shard(host0, workload, 0);
        const HostShardTensors packed1 = materialize_host_shard(host1, workload, 1);
        cotangent_stage = at::empty(
            {static_cast<std::int64_t>(workload.batch),
             static_cast<std::int64_t>(workload.coordinates)},
            at::TensorOptions().dtype(at::kFloat).device(at::kCPU).pinned_memory(true));
        std::copy(workload.upstream.begin(), workload.upstream.end(),
                  cotangent_stage.data_ptr<float>());
        preparation_metrics.host_pack_ms = elapsed_ms(prep_start);

        const auto upload_start = clock_type::now();
        shards[0] = upload_shard(packed0, device0, 0);
        shards[1] = upload_shard(packed1, device1, 1);
        preparation_metrics.upload_ms = elapsed_ms(upload_start);

        preparation_metrics.transfer_bytes = 0;
        for (unsigned slot = 0; slot != 2; ++slot) {
            const auto& shard = shards[slot];
            const Tensor* tensors[] = {&shard.x_storage, &shard.z_values, &shard.theta,
                                       &shard.phi, &shard.tau, &shard.upstream,
                                       &shard.argument_indices, &shard.active_group_indices,
                                       &shard.parameter_indices,
                                       &shard.condition_indices, &shard.output_groups,
                                       &shard.output_coordinates, &shard.output_scales};
            for (std::size_t index = 6; index < std::size(tensors); ++index)
                preparation_metrics.transfer_bytes += static_cast<std::uint64_t>(tensors[index]->nbytes());
            for (const auto* tensor : tensors)
                preparation_metrics.prepared_device_bytes += static_cast<std::uint64_t>(tensor->nbytes());
        }
        preparation_metrics.prepared_staging_bytes = cotangent_stage.nbytes();
        preparation_metrics.prepared_host_payload_bytes =
            shards[0].x_storage.nbytes() + shards[1].z_values.nbytes() +
            shards[0].theta.nbytes() + shards[0].phi.nbytes() + shards[0].tau.nbytes();
        // Keep this payload metric aligned with the Cellerator preparation
        // adapter; host upload of topology/index metadata is charged in
        // upload_ms but is not exposed as a byte counter by that API.
        preparation_metrics.transfer_bytes = preparation_metrics.prepared_host_payload_bytes;
        preparation_metrics.readiness_ms = timed_stage(devices, [] {});
        auto [active_count, min_gate, max_gate] = activity_summary(workload);
        preparation_metrics.active_groups = active_count;
        preparation_metrics.gate_argument_min = min_gate;
        preparation_metrics.gate_argument_max = max_gate;
    }

    static std::tuple<std::uint64_t, double, double> activity_summary(const Workload& w) {
        std::uint64_t active = 0;
        double minimum = std::numeric_limits<double>::infinity();
        double maximum = -std::numeric_limits<double>::infinity();
        for (std::size_t group = 0; group < w.groups(); ++group) {
            const auto p = w.parameter_index[group];
            const auto c = w.condition_index[group];
            const double z0 = static_cast<double>(w.z[c * 2]);
            const double z1 = static_cast<double>(w.z[c * 2 + 1]);
            const double argument = z0 + static_cast<double>(w.phi[p]) * z1 - w.tau[p];
            minimum = std::min(minimum, argument);
            maximum = std::max(maximum, argument);
            if (argument > 0.0) ++active;
        }
        return {active, minimum, maximum};
    }

    void replicate_forward_sources() {
        if (forward_sources_replicated) return;
        {
            c10::cuda::CUDAGuard guard(devices[0]);
            at::NoGradGuard no_grad;
            shards[0].z_values.copy_(transfer_to(shards[1].z_values, devices[0]));
        }
        {
            c10::cuda::CUDAGuard guard(devices[1]);
            at::NoGradGuard no_grad;
            shards[1].x_storage.copy_(transfer_to(shards[0].x_storage, devices[1]));
            shards[1].theta.copy_(transfer_to(shards[0].theta, devices[1]));
            shards[1].phi.copy_(transfer_to(shards[0].phi, devices[1]));
            shards[1].tau.copy_(transfer_to(shards[0].tau, devices[1]));
        }
        synchronize(devices[0]);
        synchronize(devices[1]);
        forward_sources_replicated = true;
    }
};

AtenTwoDeviceBaseline::AtenTwoDeviceBaseline(const Workload& workload, int device0, int device1)
    : impl_(std::make_unique<Impl>(workload, device0, device1)) {}
AtenTwoDeviceBaseline::~AtenTwoDeviceBaseline() = default;
AtenTwoDeviceBaseline::AtenTwoDeviceBaseline(AtenTwoDeviceBaseline&&) noexcept = default;
AtenTwoDeviceBaseline& AtenTwoDeviceBaseline::operator=(AtenTwoDeviceBaseline&&) noexcept = default;

const TimingBreakdown& AtenTwoDeviceBaseline::preparation() const noexcept {
    return impl_->preparation_metrics;
}

CycleResult AtenTwoDeviceBaseline::cycle(float learning_rate, bool update_parameters,
                                          bool collect_values) {
    CycleResult result;
    result.timing.active_groups = impl_->preparation_metrics.active_groups;
    result.timing.gate_argument_min = impl_->preparation_metrics.gate_argument_min;
    result.timing.gate_argument_max = impl_->preparation_metrics.gate_argument_max;
    const auto cycle_start = clock_type::now();
    reset_peak_stats(impl_->devices[0]);
    reset_peak_stats(impl_->devices[1]);

    ShardEvaluation evaluation[2];
    const auto forward_start = clock_type::now();
    const bool first_source_replication = !impl_->forward_sources_replicated;
    impl_->replicate_forward_sources();
    evaluation[0] = forward_shard(impl_->shards[0], impl_->workload.coordinates,
                                  impl_->workload.baseline_scale, true);
    evaluation[1] = forward_shard(impl_->shards[1], impl_->workload.coordinates,
                                  impl_->workload.baseline_scale, false);
    synchronize(impl_->devices[0]);
    synchronize(impl_->devices[1]);
    result.timing.forward_ms = elapsed_ms(forward_start);
    if (first_source_replication) {
        result.timing.condition_transfer_bytes = impl_->shards[1].z_values.nbytes();
        result.timing.participant_transfer_bytes = impl_->shards[0].x_storage.nbytes();
        result.timing.parameter_transfer_bytes = impl_->shards[0].theta.nbytes()
            + impl_->shards[0].phi.nbytes() + impl_->shards[0].tau.nbytes();
    }

    const auto backward_start = clock_type::now();
    {
        c10::cuda::CUDAGuard guard(impl_->devices[0]);
        at::NoGradGuard no_grad;
        impl_->shards[0].upstream.copy_(impl_->cotangent_stage);
    }
    {
        c10::cuda::CUDAGuard guard(impl_->devices[1]);
        at::NoGradGuard no_grad;
        impl_->shards[1].upstream.copy_(transfer_to(impl_->shards[0].upstream, impl_->devices[1]));
    }
    backward_shard(impl_->shards[0], evaluation[0]);
    backward_shard(impl_->shards[1], evaluation[1]);
    synchronize(impl_->devices[0]);
    synchronize(impl_->devices[1]);
    result.timing.backward_ms = elapsed_ms(backward_start);
    result.timing.backward_transfer_bytes = 2 * impl_->shards[0].upstream.nbytes()
        + static_cast<std::uint64_t>((impl_->workload.coordinates +
            impl_->workload.conditions * 2 + impl_->workload.parameters * 3) * sizeof(float));

    Tensor global_output;
    std::array<Tensor, 5> global_gradients;
    result.timing.reduction_ms = timed_stage(impl_->devices, [&] {
        global_output = evaluation[0].output + transfer_to(evaluation[1].output, impl_->devices[0]);
        global_gradients[0] = evaluation[0].gradients[0] +
            transfer_to(evaluation[1].gradients[0], impl_->devices[0]);
        global_gradients[1] = evaluation[1].gradients[1] +
            transfer_to(evaluation[0].gradients[1], impl_->devices[1]);
        for (std::size_t index = 2; index < global_gradients.size(); ++index)
            global_gradients[index] = evaluation[0].gradients[index] +
                transfer_to(evaluation[1].gradients[index], impl_->devices[0]);
    });
    result.timing.output_transfer_bytes = impl_->workload.coordinates * sizeof(float);

    if (update_parameters) {
        result.timing.update_ms = timed_stage(impl_->devices, [&] {
            {
                c10::cuda::CUDAGuard guard(impl_->devices[0]);
                at::NoGradGuard no_grad;
                impl_->shards[0].theta.add_(global_gradients[2], -learning_rate);
            }
            {
                c10::cuda::CUDAGuard guard(impl_->devices[1]);
                at::NoGradGuard no_grad;
                impl_->shards[1].theta.copy_(transfer_to(impl_->shards[0].theta, impl_->devices[1]));
                impl_->shards[1].phi.copy_(transfer_to(impl_->shards[0].phi, impl_->devices[1]));
                impl_->shards[1].tau.copy_(transfer_to(impl_->shards[0].tau, impl_->devices[1]));
            }
        });
        result.timing.update_includes_replica_refresh = true;
        result.timing.replica_refresh_bytes = static_cast<std::uint64_t>(
            impl_->workload.parameters * 3 * sizeof(float));
    }

    result.timing.transfer_bytes = result.timing.condition_transfer_bytes +
        result.timing.participant_transfer_bytes + result.timing.parameter_transfer_bytes +
        result.timing.output_transfer_bytes + result.timing.backward_transfer_bytes +
        result.timing.replica_refresh_bytes;
    result.timing.prepared_device_bytes = impl_->preparation_metrics.prepared_device_bytes;

    if (collect_values) {
        const auto export_start = clock_type::now();
        result.output = export_float(global_output);
        result.dx = export_float(global_gradients[0]);
        result.dz = export_float(global_gradients[1]);
        result.dtheta = export_float(global_gradients[2]);
        result.dphi = export_float(global_gradients[3]);
        result.dtau = export_float(global_gradients[4]);
        result.theta_after_update = export_float(impl_->shards[0].theta);
        result.timing.validation_export_ms = elapsed_ms(export_start);
    }
    result.timing.peak_allocated_bytes = std::max(
        allocated_peak(impl_->devices[0]), allocated_peak(impl_->devices[1]));
    result.timing.peak_reserved_bytes = std::max(
        reserved_peak(impl_->devices[0]), reserved_peak(impl_->devices[1]));
    result.timing.full_cycle_ms = elapsed_ms(cycle_start);
    return result;
}

CycleResult AtenTwoDeviceBaseline::forward_only(bool collect_values) {
    CycleResult result;
    result.timing.active_groups = impl_->preparation_metrics.active_groups;
    result.timing.gate_argument_min = impl_->preparation_metrics.gate_argument_min;
    result.timing.gate_argument_max = impl_->preparation_metrics.gate_argument_max;
    ShardEvaluation evaluation[2];
    const auto start = clock_type::now();
    const auto forward_start = clock_type::now();
    impl_->replicate_forward_sources();
    evaluation[0] = forward_shard(impl_->shards[0], impl_->workload.coordinates,
                                  impl_->workload.baseline_scale, true);
    evaluation[1] = forward_shard(impl_->shards[1], impl_->workload.coordinates,
                                  impl_->workload.baseline_scale, false);
    synchronize(impl_->devices[0]);
    synchronize(impl_->devices[1]);
    result.timing.forward_ms = elapsed_ms(forward_start);
    Tensor output;
    result.timing.reduction_ms = timed_stage(impl_->devices, [&] {
        output = evaluation[0].output + transfer_to(evaluation[1].output, impl_->devices[0]);
    });
    result.timing.transfer_bytes += static_cast<std::uint64_t>(impl_->workload.coordinates * sizeof(float));
    result.timing.output_transfer_bytes = impl_->workload.coordinates * sizeof(float);
    result.timing.prepared_device_bytes = impl_->preparation_metrics.prepared_device_bytes;
    if (collect_values) {
        const auto export_start = clock_type::now();
        result.output = export_float(output);
        result.theta_after_update = export_float(impl_->shards[0].theta);
        result.timing.validation_export_ms = elapsed_ms(export_start);
    }
    result.timing.full_cycle_ms = elapsed_ms(start);
    return result;
}

} // namespace cellerator::bench::native_data
