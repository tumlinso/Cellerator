#include "distributed_adapter.hh"

#include <Cellerator/execution/dependency/distributed.cuh>
#include <Cellerator/geometry/dependency/applicability.hh>
#include <Cellerator/ops/mechanism/gated_product_binding.hh>
#include <Cellerator/tensor/block_masked/structure.hh>

#include <c10/util/Half.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

namespace cellerator::bench::native_data {
namespace {

using clock_type = std::chrono::steady_clock;
namespace dependency = cellerator::execution::dependency;
namespace geometry = cellerator::geometry::dependency;
namespace mechanism = cellerator::ops::mechanism;
namespace block_masked = cellerator::tensor::block_masked;
namespace structure = cellerator::structure;

double elapsed_ms(clock_type::time_point start) {
    return std::chrono::duration<double, std::milli>(clock_type::now() - start).count();
}

void cuda_check(cudaError_t status, const char* action) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(action) + ": " + cudaGetErrorString(status));
}

void synchronize(int device) {
    cuda_check(cudaSetDevice(device), "cudaSetDevice");
    cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
}

std::uint16_t fp16_bits(float value) noexcept {
    const c10::Half converted(value);
    std::uint16_t bits = 0;
    static_assert(sizeof(converted) == sizeof(bits));
    std::memcpy(&bits, &converted, sizeof(bits));
    return bits;
}

std::uint64_t delta(std::uint64_t before, std::uint64_t after) noexcept {
    return after >= before ? after - before : after;
}

dependency::PackedAtomIdentities packed_atom_identities(
    std::uint64_t producer_namespace, std::uint64_t first_local_identity) noexcept {
    return {{producer_namespace, first_local_identity},
            {producer_namespace, first_local_identity + 1},
            {producer_namespace, first_local_identity + 2},
            {producer_namespace, first_local_identity + 3},
            {producer_namespace, first_local_identity + 4}};
}

std::uint64_t packed_structure_signature(const Workload& workload) noexcept {
    // This identity describes the immutable packed state geometry. It omits
    // values, gates, and group semantics so value changes do not rename the
    // physical structure.
    std::uint64_t hash = 14695981039346656037ULL;
    const auto add = [&](std::uint64_t value) {
        for (unsigned shift = 0; shift < 64; shift += 8) {
            hash ^= static_cast<std::uint8_t>(value >> shift);
            hash *= 1099511628211ULL;
        }
    };
    add(workload.batch);
    add(workload.coordinates);
    add(32); // physical block and tile widths used by this adapter
    add(32);
    for (std::size_t physical = 0; physical < workload.coordinates; ++physical)
        add(workload.coordinates - 1 - physical);
    return hash;
}

geometry::SourceDescriptor source(std::uint64_t identity,
                                  structure::domain_id domain,
                                  structure::order_id order,
                                  std::uint64_t extent) {
    return {{geometry::source_id{identity, 0}, domain,
             geometry::representation_id{identity + 1000, 0}},
            {domain, order, extent}};
}

std::tuple<std::uint64_t, double, double> activity_summary(const Workload& workload) {
    std::uint64_t active = 0;
    double minimum = std::numeric_limits<double>::infinity();
    double maximum = -std::numeric_limits<double>::infinity();
    for (std::size_t group = 0; group < workload.groups(); ++group) {
        const auto parameter = workload.parameter_index[group];
        const auto condition = workload.condition_index[group];
        const double argument = workload.z[condition * 2] +
            workload.phi[parameter] * workload.z[condition * 2 + 1] - workload.tau[parameter];
        minimum = std::min(minimum, argument);
        maximum = std::max(maximum, argument);
        if (argument > 0.0) ++active;
    }
    return {active, minimum, maximum};
}

} // namespace

struct DistributedGatedProductAdapter::Impl {
    Workload workload;
    std::array<int, 2> devices{};
    std::vector<mechanism::GatedProductGroup> groups;
    std::vector<std::uint64_t> arguments;
    std::vector<mechanism::OutputIncidence> outputs;
    std::vector<std::uint8_t> group_slots;
    std::shared_ptr<const block_masked::Structure> layout_owner;
    std::shared_ptr<const mechanism::PreparedGatedProductBinding> semantics;
    std::vector<std::uint16_t> packed_values;
    std::unique_ptr<dependency::PreparedDistributedGatedProduct> program;
    TimingBreakdown preparation_metrics;

    Impl(const Workload& input, int device0, int device1) : devices{{device0, device1}} {
        if (device0 < 0 || device1 < 0 || device0 == device1)
            throw std::invalid_argument("specialist runner requires two distinct visible CUDA devices");
        workload = input;
        if (workload.batch != 1 || !workload.groups())
            throw std::invalid_argument("specialist benchmark currently supports one non-empty sample");
        if (workload.group_owner.size() != workload.groups())
            throw std::invalid_argument("specialist group ownership extent mismatch");
        int device_count = 0;
        cuda_check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
        if (device_count <= std::max(device0, device1))
            throw std::invalid_argument("selected specialist CUDA device is not visible");

        const auto host_pack_start = clock_type::now();
        make_topology();
        make_structure_and_values();
        make_semantics();
        preparation_metrics.host_pack_ms = elapsed_ms(host_pack_start);
        const auto input_payload_bytes = packed_values.size() * sizeof(std::uint16_t) +
            workload.z.size() * sizeof(float) +
            (workload.theta.size() + workload.phi.size() + workload.tau.size()) * sizeof(float);
        preparation_metrics.prepared_host_payload_bytes = input_payload_bytes;

        dependency::DistributedGatedProductOptions options;
        options.structure = layout_owner;
        options.semantics = semantics;
        options.source_bindings = semantics->sources();
        options.access_revision = semantics->access().revision;
        options.topology = semantics->topology();
        options.group_slots = group_slots;
        options.devices = devices;
        options.transfer_mode = dependency::TransferMode::peer_if_supported;
        options.maximum_live_forwards = 2;
        options.beta = -workload.baseline_scale;
        // These are physical realization identities, separate from the
        // semantic state order bound above. The benchmark uses a common
        // packed feature order, but the two prepared device realizations still
        // receive explicit, stable identities as required by the executor.
        options.physical_orders = {
            structure::order_id{1401, 1}, structure::order_id{1402, 1}};
        options.atoms = {
            packed_atom_identities(0x43504750484f4d45ull, 1),
            packed_atom_identities(0x43504750524f4a31ull, 1)};
        const auto physical_signature = packed_structure_signature(workload);
        options.physical_structures = {
            structure::structure_id{physical_signature, 0x4e41544956454401ull},
            structure::structure_id{physical_signature, 0x4e41544956454402ull}};

        const auto upload_start = clock_type::now();
        program = std::make_unique<dependency::PreparedDistributedGatedProduct>(
            std::move(options), packed_values, workload.z, workload.theta,
            workload.phi, workload.tau);
        preparation_metrics.upload_ms = elapsed_ms(upload_start);
        const auto readiness_start = clock_type::now();
        synchronize(devices[0]);
        synchronize(devices[1]);
        preparation_metrics.readiness_ms = elapsed_ms(readiness_start);

        const auto stats = program->stats();
        preparation_metrics.prepared_device_bytes = stats.prepared_device_bytes;
        preparation_metrics.prepared_staging_bytes = stats.prepared_staging_bytes;
        preparation_metrics.peer_forward = stats.peer_forward;
        preparation_metrics.peer_reverse = stats.peer_reverse;
        const auto [active, low, high] = activity_summary(workload);
        preparation_metrics.active_groups = active;
        preparation_metrics.gate_argument_min = low;
        preparation_metrics.gate_argument_max = high;
    }

    void make_topology() {
        groups.resize(workload.groups());
        group_slots.reserve(workload.groups());
        arguments.reserve(workload.argument_coordinate.size());
        outputs.reserve(workload.output_coordinate.size());
        for (const auto coordinate : workload.argument_coordinate) arguments.push_back(coordinate);
        for (std::size_t group = 0; group < workload.groups(); ++group) {
            groups[group] = {
                workload.argument_begin[group], workload.argument_count[group],
                workload.output_begin[group], workload.output_count[group],
                workload.parameter_index[group], workload.condition_index[group],
                static_cast<std::uint64_t>(group + 101)};
            const int owner = workload.group_owner[group];
            if (owner < 0 || owner > 1)
                throw std::invalid_argument("group owner must be device slot 0 or 1");
            group_slots.push_back(static_cast<std::uint8_t>(owner));
        }
        for (std::size_t incidence = 0; incidence < workload.output_coordinate.size(); ++incidence)
            outputs.push_back({workload.output_coordinate[incidence], workload.output_scale[incidence]});
    }

    mechanism::GatedProductTopologyView topology() const noexcept {
        return {groups.data(), groups.size(), arguments.data(), arguments.size(),
                outputs.data(), outputs.size(), workload.coordinates,
                workload.conditions, workload.parameters};
    }

    void make_structure_and_values() {
        if (workload.coordinates > std::numeric_limits<std::uint32_t>::max())
            throw std::overflow_error("benchmark coordinates exceed block-masked physical index width");
        std::vector<std::uint64_t> canonical_features(workload.coordinates);
        for (std::size_t physical = 0; physical < workload.coordinates; ++physical)
            canonical_features[physical] = workload.coordinates - 1 - physical;
        std::vector<std::uint32_t> full_physical_row;
        full_physical_row.reserve(workload.coordinates);
        for (std::size_t physical = 0; physical < workload.coordinates; ++physical)
            full_physical_row.push_back(static_cast<std::uint32_t>(physical));
        std::vector<std::vector<std::uint32_t>> rows(workload.batch, full_physical_row);
        auto owned = block_masked::Structure::from_rows(
            workload.coordinates, rows, 32, 32, std::move(canonical_features));
        layout_owner = std::make_shared<const block_masked::Structure>(std::move(owned));

        packed_values.resize(layout_owner->stored_values());
        for (std::size_t batch = 0; batch < workload.batch; ++batch) {
            for (std::size_t coordinate = 0; coordinate < workload.coordinates; ++coordinate) {
                const auto physical = layout_owner->physical_feature(coordinate);
                const auto offset = layout_owner->value_offset(batch, physical);
                packed_values[offset] = fp16_bits(workload.x[batch * workload.coordinates + coordinate]);
            }
        }
    }

    void make_semantics() {
        const structure::domain_id state_domain{1001, 0};
        const structure::order_id state_order{1002, 0};
        const structure::domain_id regulator_domain{1011, 0};
        const structure::order_id regulator_order{1012, 0};
        const structure::domain_id parameter_domain{1021, 0};
        const structure::order_id parameter_order{1022, 0};
        const mechanism::GatedProductSourceBindings bindings{
            source(1101, state_domain, state_order, workload.coordinates),
            source(1102, regulator_domain, regulator_order, workload.conditions * 2),
            source(1103, parameter_domain, parameter_order, workload.parameters),
            source(1104, parameter_domain, parameter_order, workload.parameters),
            source(1105, parameter_domain, parameter_order, workload.parameters),
            source(1106, state_domain, state_order, workload.coordinates)};
        const geometry::Applicability applicability{
            {1201, 0}, {1202, 0}, {1203, 0}, {1204, 0}, {1}};
        semantics = std::make_shared<const mechanism::PreparedGatedProductBinding>(
            topology(), bindings, applicability, geometry::foundation_id{1301, 0},
            mechanism::operation_id{1302, 0}, mechanism::access_contract_id{1303, 0}, 1);
    }
};

DistributedGatedProductAdapter::DistributedGatedProductAdapter(
    const Workload& workload, int device0, int device1)
    : impl_(std::make_unique<Impl>(workload, device0, device1)) {}
DistributedGatedProductAdapter::~DistributedGatedProductAdapter() = default;
DistributedGatedProductAdapter::DistributedGatedProductAdapter(
    DistributedGatedProductAdapter&&) noexcept = default;
DistributedGatedProductAdapter& DistributedGatedProductAdapter::operator=(
    DistributedGatedProductAdapter&&) noexcept = default;

const TimingBreakdown& DistributedGatedProductAdapter::preparation() const noexcept {
    return impl_->preparation_metrics;
}

CycleResult DistributedGatedProductAdapter::cycle(float learning_rate,
                                                    bool update_parameters,
                                                    bool collect_values) {
    CycleResult result;
    result.timing.active_groups = impl_->preparation_metrics.active_groups;
    result.timing.gate_argument_min = impl_->preparation_metrics.gate_argument_min;
    result.timing.gate_argument_max = impl_->preparation_metrics.gate_argument_max;
    const auto before = impl_->program->stats();
    const auto cycle_start = clock_type::now();

    const auto forward_start = clock_type::now();
    auto tape = impl_->program->forward();
    synchronize(impl_->devices[0]);
    synchronize(impl_->devices[1]);
    result.timing.forward_ms = elapsed_ms(forward_start);
    if (collect_values) {
        const auto values_start = clock_type::now();
        result.output = impl_->program->values(tape);
        result.timing.validation_export_ms += elapsed_ms(values_start);
    }

    const auto backward_start = clock_type::now();
    impl_->program->backward(tape, std::span<const float>(
        impl_->workload.upstream.data(), impl_->workload.upstream.size()));
    synchronize(impl_->devices[0]);
    synchronize(impl_->devices[1]);
    result.timing.backward_ms = elapsed_ms(backward_start);

    const auto ready_start = clock_type::now();
    tape.wait();
    result.timing.readiness_ms = elapsed_ms(ready_start);

    if (collect_values) {
        const auto export_start = clock_type::now();
        auto gradient = impl_->program->gradients(tape);
        result.dx = std::move(gradient.dx);
        result.dz = std::move(gradient.dz);
        result.dtheta = std::move(gradient.dtheta);
        result.dphi = std::move(gradient.dphi);
        result.dtau = std::move(gradient.dtau);
        result.timing.validation_export_ms += elapsed_ms(export_start);
    }

    if (update_parameters) {
        const auto update_start = clock_type::now();
        impl_->program->update(tape, learning_rate);
        synchronize(impl_->devices[0]);
        synchronize(impl_->devices[1]);
        result.timing.update_ms = elapsed_ms(update_start);
        result.timing.update_includes_replica_refresh = true;
    }
    if (collect_values) {
        const auto parameter_export_start = clock_type::now();
        result.theta_after_update = impl_->program->parameters()[0];
        result.timing.validation_export_ms += elapsed_ms(parameter_export_start);
    }

    const auto after = impl_->program->stats();
    result.timing.condition_transfer_bytes = delta(before.condition_bytes, after.condition_bytes);
    result.timing.participant_transfer_bytes = delta(before.participant_bytes, after.participant_bytes);
    result.timing.parameter_transfer_bytes = delta(before.parameter_bytes, after.parameter_bytes);
    result.timing.output_transfer_bytes = delta(before.output_bytes, after.output_bytes);
    result.timing.backward_transfer_bytes = delta(before.backward_bytes, after.backward_bytes);
    result.timing.replica_refresh_bytes = delta(before.replica_refresh_bytes, after.replica_refresh_bytes);
    result.timing.omitted_participant_transfers = delta(
        before.omitted_participant_transfers, after.omitted_participant_transfers);
    result.timing.projection_refreshes = delta(before.projection_refreshes, after.projection_refreshes);
    result.timing.projected_participant_values = delta(
        before.projected_participant_values, after.projected_participant_values);
    result.timing.transfer_bytes = result.timing.condition_transfer_bytes +
        result.timing.participant_transfer_bytes + result.timing.parameter_transfer_bytes +
        result.timing.output_transfer_bytes + result.timing.backward_transfer_bytes +
        result.timing.replica_refresh_bytes;
    result.timing.prepared_device_bytes = after.prepared_device_bytes;
    result.timing.prepared_staging_bytes = after.prepared_staging_bytes;
    result.timing.peer_forward = after.peer_forward;
    result.timing.peer_reverse = after.peer_reverse;
    result.timing.full_cycle_ms = elapsed_ms(cycle_start);
    return result;
}

CycleResult DistributedGatedProductAdapter::forward_only(bool collect_values) {
    CycleResult result;
    result.timing.active_groups = impl_->preparation_metrics.active_groups;
    result.timing.gate_argument_min = impl_->preparation_metrics.gate_argument_min;
    result.timing.gate_argument_max = impl_->preparation_metrics.gate_argument_max;
    const auto before = impl_->program->stats();
    const auto full_start = clock_type::now();
    const auto forward_start = clock_type::now();
    auto tape = impl_->program->forward();
    synchronize(impl_->devices[0]);
    synchronize(impl_->devices[1]);
    result.timing.forward_ms = elapsed_ms(forward_start);
    if (collect_values) {
        const auto export_start = clock_type::now();
        result.output = impl_->program->values(tape);
        result.timing.validation_export_ms = elapsed_ms(export_start);
        result.theta_after_update = impl_->program->parameters()[0];
    }
    const auto ready_start = clock_type::now();
    tape.wait();
    result.timing.readiness_ms = elapsed_ms(ready_start);
    const auto after = impl_->program->stats();
    result.timing.output_transfer_bytes = delta(before.output_bytes, after.output_bytes);
    result.timing.transfer_bytes = result.timing.output_transfer_bytes;
    result.timing.prepared_device_bytes = after.prepared_device_bytes;
    result.timing.prepared_staging_bytes = after.prepared_staging_bytes;
    result.timing.peer_forward = after.peer_forward;
    result.timing.peer_reverse = after.peer_reverse;
    result.timing.full_cycle_ms = elapsed_ms(full_start);
    return result;
}

} // namespace cellerator::bench::native_data
