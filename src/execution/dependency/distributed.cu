#include <Cellerator/execution/dependency/distributed.cuh>
#include <Cellerator/ops/mechanism/gated_product_binding.hh>
#include <Cellerator/ops/mechanism/packed_gated_product.cuh>
#include <Cellerator/tensor/block_masked/owner.hh>

#include <cuda_runtime.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <exception>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>

namespace cellerator::execution::dependency {
namespace mc = memory::cuda;
namespace bm = tensor::block_masked;
namespace op = ops::mechanism;
namespace gd = geometry::dependency;
namespace {
void require(bool value, const char* message) {
    if (!value) throw std::invalid_argument(message);
}
std::size_t compact_gradient_values(std::size_t gradients, std::size_t condition_values,
                                    std::size_t state_values) {
    require(gradients >= condition_values && gradients - condition_values >= state_values,
            "distributed gradient packet is smaller than its state and condition sections");
    return gradients - condition_values;
}
std::size_t product(std::size_t a, std::size_t b) {
    return mc::checked_bytes(a, b);
}
unsigned blocks(std::size_t count) {
    const auto n = count / 256 + (count % 256 != 0);
    if (n > std::numeric_limits<unsigned>::max()) throw std::overflow_error("distributed launch grid overflow");
    return static_cast<unsigned>(n);
}
class DrainOnFailure {
public:
    explicit DrainOnFailure(const mc::Stream& first, const mc::Stream* second = nullptr)
        : first_(first), second_(second), exceptions_(std::uncaught_exceptions()) {}
    ~DrainOnFailure() noexcept {
        if (std::uncaught_exceptions() <= exceptions_) return;
        try { first_.synchronize(); if (second_) second_->synchronize(); }
        catch (...) { std::terminate(); }
    }
private:
    const mc::Stream& first_;
    const mc::Stream* second_;
    int exceptions_;
};
template<class T>
mc::DeviceBuffer upload(std::span<const T> values, const mc::Stream& stream) {
    mc::DeviceBuffer result(product(values.size(), sizeof(T)), stream.device());
    result.upload_async(values.data(), result.bytes(), stream);
    return result;
}
__global__ void add_values(float* target, const float* source, std::size_t count, std::uint32_t* status) {
    const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) {
        const float sum = target[i] + source[i];
        if (!isfinite(sum)) atomicOr(status, static_cast<std::uint32_t>(op::packed_nonfinite));
        target[i] = sum;
    }
}
__global__ void sgd_proposal(const float* parameters, float* proposal, const float* gradient,
                            std::size_t parameter_count, ParameterLearningRates rates,
                            std::uint32_t* status) {
    const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < parameter_count * 3) {
        const auto field = i / parameter_count;
        const auto rate = field == 0 ? rates.theta : (field == 1 ? rates.phi : rates.tau);
        const float value = rate != 0.0f ? parameters[i] - rate * gradient[i] : parameters[i];
        if (rate != 0.0f && !isfinite(value)) atomicOr(status, static_cast<std::uint32_t>(op::packed_nonfinite));
        proposal[i] = value;
    }
}

// A fixed, compact remote-source projection. The posting list identifies the
// local mechanisms that may read each stored coordinate. Only their certified
// active conditions permit a source value read; stored zeros remain present.
__global__ void pack_remote(
    const std::uint16_t* source, std::uint16_t* destination,
    const std::uint64_t* source_offsets, const std::uint64_t* rows,
    const std::uint64_t* coordinates, const std::uint64_t* user_offsets,
    const std::uint32_t* users, const op::GatedProductGroup* groups,
    const float* z, const float* phi, const float* tau,
    std::uint64_t conditions, std::size_t count, std::uint32_t* reads) {
    const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const auto coordinate = coordinates[i];
    bool needed = false;
    for (auto u = user_offsets[coordinate]; u < user_offsets[coordinate + 1]; ++u) {
        const auto group = groups[users[u]];
        const auto at = (rows[i] * conditions + group.condition_index) * 2;
        if (!op::evaluate_gate(z[at], z[at + 1], phi[group.parameter_index], tau[group.parameter_index]).excluded) {
            needed = true; break;
        }
    }
    destination[i] = needed ? source[source_offsets[i]] : std::uint16_t{0};
    if (needed) atomicAdd(reads, 1u);
}

void add(mc::DeviceView target, mc::ConstDeviceView source, std::size_t count, mc::DeviceView status,
         const mc::Stream& stream) {
    if (!count) return;
    mc::DeviceGuard guard(stream.device());
    add_values<<<blocks(count), 256, 0, stream.get()>>>(
        static_cast<float*>(target.data), static_cast<const float*>(source.data), count,
        static_cast<std::uint32_t*>(status.data));
    mc::check(cudaGetLastError(), "distributed gradient/output accumulation");
}

bool same_topology(op::GatedProductTopologyView a, op::GatedProductTopologyView b) {
    if (a.group_count != b.group_count || a.argument_count != b.argument_count
        || a.output_count != b.output_count || a.coordinates != b.coordinates
        || a.conditions != b.conditions || a.parameters != b.parameters) return false;
    if ((a.group_count && !a.groups) || (a.argument_count && !a.arguments)
        || (a.output_count && !a.outputs)) return false;
    for (std::size_t i = 0; i < a.group_count; ++i) {
        const auto x = a.groups[i], y = b.groups[i];
        if (x.argument_begin != y.argument_begin || x.argument_count != y.argument_count
            || x.output_begin != y.output_begin || x.output_count != y.output_count
            || x.parameter_index != y.parameter_index || x.condition_index != y.condition_index
            || x.logical_id != y.logical_id) return false;
    }
    for (std::size_t i = 0; i < a.argument_count; ++i) if (a.arguments[i] != b.arguments[i]) return false;
    for (std::size_t i = 0; i < a.output_count; ++i)
        if (a.outputs[i].coordinate != b.outputs[i].coordinate || a.outputs[i].scale != b.outputs[i].scale) return false;
    return true;
}

struct LocalDevice {
    mc::Stream stream;
    std::shared_ptr<const bm::DeviceStructure> structure;
    bm::NativeTensor<std::uint16_t> values;
    mc::DeviceBuffer regulators, parameters, groups, arguments, outputs, argument_offsets, coordinate_offsets;
    op::GatedProductTopologyView topology;
    mc::Event x_ready, z_ready, parameter_ready;
    std::unique_ptr<PreparedPackedProgram> program;
    std::uint64_t x_generation = 0, z_generation = 0, parameter_generation = 0;
    LocalDevice(int device, std::shared_ptr<const bm::Structure> host,
                op::GatedProductTopologyView full, const std::vector<op::GatedProductGroup>& subset,
                std::size_t batch, bool baseline)
        : stream(device), structure(std::make_shared<bm::DeviceStructure>(std::move(host), stream)),
          values(structure), regulators(product(product(batch, full.conditions), 2 * sizeof(float)), device),
          parameters(product(full.parameters, 3 * sizeof(float)), device),
          x_ready(device), z_ready(device), parameter_ready(device) {
        op::GatedProductTopologyView local = full;
        local.groups = subset.data(); local.group_count = subset.size();
        const auto slots = op::prepare_argument_value_offsets(structure->host(), local);
        const auto coordinates = baseline
            ? op::prepare_coordinate_value_offsets(structure->host(), full.coordinates)
            : std::vector<std::uint64_t>{};
        DrainOnFailure drain(stream);
        groups = upload<op::GatedProductGroup>(subset, stream);
        arguments = upload<std::uint64_t>({full.arguments, static_cast<std::size_t>(full.argument_count)}, stream);
        outputs = upload<op::OutputIncidence>({full.outputs, static_cast<std::size_t>(full.output_count)}, stream);
        argument_offsets = upload<std::uint64_t>(slots, stream);
        coordinate_offsets = upload<std::uint64_t>(coordinates, stream);
        topology = {static_cast<const op::GatedProductGroup*>(groups.data()), subset.size(),
            static_cast<const std::uint64_t*>(arguments.data()), full.argument_count,
            static_cast<const op::OutputIncidence*>(outputs.data()), full.output_count,
            full.coordinates, full.conditions, full.parameters};
        stream.synchronize();
    }
    const float* theta() const { return static_cast<const float*>(parameters.data()); }
    const float* phi() const { return theta() + topology.parameters; }
    const float* tau() const { return phi() + topology.parameters; }
};

struct Slot {
    bool occupied = false, consumed = false, inputs_retired = false, updated = false, failed = false;
    std::uint64_t incarnation = 0, parameter_generation = 0, context_generation = 0, state_generation = 0;
    std::array<std::optional<bm::RetainedRead<std::uint16_t>>, 2> reads;
    std::array<std::unique_ptr<mc::DeviceBuffer>, 2> output, cotangent, gradient, saved, status;
    std::array<std::unique_ptr<mc::Event>, 2> completed;
    std::array<std::unique_ptr<mc::Event>, 2> cotangent_ready;
    mc::DeviceBuffer remote_output, remote_gradient, remote_condition_gradient;
    mc::PinnedBuffer cotangent_stage;
    std::unique_ptr<PreparedTransfer> output_transfer, cotangent_transfer, gradient_transfer,
        parameter_gradient_transfer, condition_gradient_transfer;
    Slot(const std::array<int, 2>& devices, TransferMode mode, std::size_t values,
         std::size_t gradients, std::size_t condition_values,
         std::size_t batch, std::array<std::size_t, 2> group_counts)
        : remote_output(product(values, sizeof(float)), devices[0]),
          remote_gradient(product(compact_gradient_values(gradients, condition_values, values), sizeof(float)), devices[0]),
          remote_condition_gradient(product(condition_values, sizeof(float)), devices[1]),
          cotangent_stage(product(values, sizeof(float))) {
        for (unsigned d = 0; d != 2; ++d) {
            output[d] = std::make_unique<mc::DeviceBuffer>(product(values, sizeof(float)), devices[d]);
            cotangent[d] = std::make_unique<mc::DeviceBuffer>(product(values, sizeof(float)), devices[d]);
            gradient[d] = std::make_unique<mc::DeviceBuffer>(product(gradients, sizeof(float)), devices[d]);
            saved[d] = std::make_unique<mc::DeviceBuffer>(product(product(batch, group_counts[d]), 2 * sizeof(float)), devices[d]);
            status[d] = std::make_unique<mc::DeviceBuffer>(sizeof(std::uint32_t), devices[d]);
            completed[d] = std::make_unique<mc::Event>(devices[d]);
            cotangent_ready[d] = std::make_unique<mc::Event>(devices[d]);
        }
        output_transfer = std::make_unique<PreparedTransfer>(devices[1], devices[0], remote_output.bytes(), mode);
        cotangent_transfer = std::make_unique<PreparedTransfer>(devices[0], devices[1], cotangent_stage.bytes(), mode);
        const auto value_bytes = product(values, sizeof(float));
        const auto parameter_bytes = remote_gradient.bytes() - value_bytes;
        gradient_transfer = std::make_unique<PreparedTransfer>(devices[1], devices[0], value_bytes, mode);
        parameter_gradient_transfer = std::make_unique<PreparedTransfer>(devices[1], devices[0], parameter_bytes, mode);
        condition_gradient_transfer = std::make_unique<PreparedTransfer>(devices[0], devices[1], product(condition_values, sizeof(float)), mode);
    }
};
}

namespace detail {
struct DistributedProductState {
    DistributedGatedProductOptions options;
    std::shared_ptr<const op::PreparedGatedProductBinding> semantics;
    std::vector<gd::SourceDescriptor> actual_sources;
    std::array<std::unique_ptr<LocalDevice>, 2> devices;
    std::vector<std::unique_ptr<Slot>> slots;
    std::vector<TransferRequirement> requirements;
    std::unique_ptr<PreparedTransfer> x_transfer, z_transfer, parameter_transfer;
    mc::DeviceBuffer projection, projection_source_offsets, projection_rows, projection_coordinates,
        user_offsets, users, projection_groups, projection_reads;
    mc::PinnedBuffer x_stage, z_stage, parameter_stage, projection_read_stage;
    std::size_t batch = 0, value_count = 0, condition_count = 0, parameter_count = 0, gradient_count = 0;
    std::size_t active_readers = 0;
    bool poisoned = false;
    std::uint64_t x_generation = 1, z_generation = 1, parameter_generation = 1,
                  gate_parameter_generation = 1;
    std::array<std::uint64_t, 3> projection_stamp{};
    DistributedTransferStats accounting;

    DistributedProductState(DistributedGatedProductOptions config,
        std::span<const std::uint16_t> x, std::span<const float> z,
        std::span<const float> theta, std::span<const float> phi, std::span<const float> tau)
        : options(std::move(config)), semantics(options.semantics),
          actual_sources(options.source_bindings.begin(), options.source_bindings.end()),
          projection_read_stage(sizeof(std::uint32_t)) {
        require(bool(semantics) && bool(options.structure), "distributed operation requires structure and semantic binding");
        require(semantics->validate_sources(options.source_bindings).code == gd::StatusCode::success,
                "distributed source meaning/domain/order mismatch");
        require(options.access_revision == semantics->access().revision, "stale distributed access revision");
        require(same_topology(options.topology, semantics->topology()), "distributed topology differs from bound foundation");
        options.topology = semantics->topology();
        const auto topology = options.topology;
        require(topology.coordinates && topology.parameters && topology.conditions && topology.group_count,
                "distributed example requires nonempty dimensions");
        require(options.devices[0] >= 0 && options.devices[1] >= 0 && options.devices[0] != options.devices[1],
                "distributed operation requires two distinct devices");
        require(structure::valid_identity(options.physical_orders[0])
            && structure::valid_identity(options.physical_orders[1]),
            "distributed physical realizations require explicit nonzero order identities");
        for (unsigned d = 0; d != 2; ++d) {
            require(structure::valid_identity(options.physical_structures[d]) && options.structure_epochs[d],
                "distributed realization requires an explicit structure identity and epoch");
            const auto& ids = options.atoms[d];
            for (const auto id : {ids.atom, ids.requirement, ids.coverage, ids.affordance, ids.binding})
                require(bool(joint_compiler::validate_persistent_identity_v1(id)), "invalid distributed atom identity");
        }
        require(options.maximum_live_forwards && options.group_slots.size() == topology.group_count && std::isfinite(options.beta),
                "invalid distributed preparation/capacity");
        batch = options.structure->rows();
        require(batch && options.structure->features() == topology.coordinates, "packed structure/source extent mismatch");
        value_count = product(batch, topology.coordinates);
        condition_count = product(product(batch, topology.conditions), 2);
        parameter_count = topology.parameters;
        gradient_count = value_count + condition_count + product(parameter_count, 3);
        require(gradient_count >= value_count && gradient_count >= condition_count, "gradient extent overflow");
        require(x.size() == options.structure->stored_values() && z.size() == condition_count
            && theta.size() == parameter_count && phi.size() == parameter_count && tau.size() == parameter_count,
            "distributed initial source extents mismatch");
        std::array<std::vector<op::GatedProductGroup>, 2> groups;
        std::array<std::vector<std::uint64_t>, 2> group_indices;
        std::vector<std::vector<std::uint32_t>> coordinate_users(topology.coordinates);
        for (std::size_t i = 0; i < topology.group_count; ++i) {
            require(options.group_slots[i] < 2, "invalid group device slot");
            require(topology.groups[i].output_count != 0,
                "distributed packed consumer does not support outputless groups");
            for (auto omission : {op::Omission::input_read, op::Omission::forward_contribution,
                                   op::Omission::input_vjp, op::Omission::condition_vjp, op::Omission::parameter_vjp})
                require(semantics->certifies_inactive_omission(i, omission), "bound operation does not certify the requested first-order omission");
            const auto group = topology.groups[i];
            if (options.group_slots[i] == 1) {
                const auto local_index = static_cast<std::uint32_t>(groups[1].size());
                for (auto slot = group.argument_begin; slot < group.argument_begin + group.argument_count; ++slot)
                    coordinate_users[topology.arguments[slot]].push_back(local_index);
            }
            groups[options.group_slots[i]].push_back(group);
            group_indices[options.group_slots[i]].push_back(i);
        }
        require(!groups[0].empty() && !groups[1].empty(), "both devices must own actual mechanisms");
        std::vector<std::uint32_t> physical_features;
        std::vector<std::uint64_t> posting_offsets{0};
        std::vector<std::uint32_t> posting_users;
        for (std::uint64_t coordinate = 0; coordinate < topology.coordinates; ++coordinate) {
            auto& list = coordinate_users[coordinate];
            std::sort(list.begin(), list.end()); list.erase(std::unique(list.begin(), list.end()), list.end());
            if (!list.empty()) physical_features.push_back(options.structure->physical_feature(coordinate));
            posting_users.insert(posting_users.end(), list.begin(), list.end());
            posting_offsets.push_back(posting_users.size());
        }
        std::sort(physical_features.begin(), physical_features.end());
        std::vector<std::vector<std::uint32_t>> remote_rows(batch, physical_features);
        auto remote_structure = std::make_shared<bm::Structure>(bm::Structure::from_rows(
            topology.coordinates, remote_rows, 32, options.structure->view().tile_row_width,
            options.structure->canonical_features(), options.structure->canonical_rows()));
        devices[0] = std::make_unique<LocalDevice>(options.devices[0], options.structure, topology, groups[0], batch, true);
        devices[1] = std::make_unique<LocalDevice>(options.devices[1], remote_structure, topology, groups[1], batch, false);
        for (unsigned d = 0; d != 2; ++d) {
            auto& device = *devices[d];
            op::PackedGatedProductInput<std::uint16_t> prototype{};
            prototype.x.layout = device.structure->view(); prototype.topology = device.topology;
            prototype.argument_value_offsets = static_cast<const std::uint64_t*>(device.argument_offsets.data());
            prototype.coordinate_value_offsets = static_cast<const std::uint64_t*>(device.coordinate_offsets.data());
            prototype.batch = batch; prototype.beta = options.beta; prototype.include_baseline = d == 0;
            device.program = std::make_unique<PreparedPackedProgram>(PackedProgramPreparation{
                semantics, device.structure, group_indices[d], options.atoms[d], options.physical_structures[d],
                options.physical_orders[d], options.structure_epochs[d], 1, prototype});
        }
        // Do not retain caller spans after preparation.
        options.group_slots = {}; options.source_bindings = {};
        auto& home = *devices[0]; auto& remote = *devices[1];
        const auto remote_values = remote_structure->stored_values();
        std::vector<std::uint64_t> offsets(remote_values), rows(remote_values), coordinates(remote_values);
        auto canonical_batch_rows = options.structure->canonical_rows();
        std::sort(canonical_batch_rows.begin(), canonical_batch_rows.end());
        for (std::size_t row = 0; row < batch; ++row) {
            const auto row_id = options.structure->canonical_rows()[row];
            const auto batch_ordinal = static_cast<std::uint64_t>(
                std::lower_bound(canonical_batch_rows.begin(), canonical_batch_rows.end(), row_id)
                - canonical_batch_rows.begin());
            for (auto feature : physical_features) {
                const auto destination = remote_structure->value_offset(row, feature);
                offsets[destination] = options.structure->value_offset(row, feature);
                rows[destination] = batch_ordinal;
                coordinates[destination] = options.structure->canonical_features()[feature];
            }
        }
        DrainOnFailure drain(home.stream, &remote.stream);
        projection = mc::DeviceBuffer(product(remote_values, sizeof(std::uint16_t)), options.devices[0]);
        projection_source_offsets = upload<std::uint64_t>(offsets, home.stream);
        projection_rows = upload<std::uint64_t>(rows, home.stream);
        projection_coordinates = upload<std::uint64_t>(coordinates, home.stream);
        user_offsets = upload<std::uint64_t>(posting_offsets, home.stream);
        users = upload<std::uint32_t>(posting_users, home.stream);
        projection_groups = upload<op::GatedProductGroup>(groups[1], home.stream);
        projection_reads = mc::DeviceBuffer(sizeof(std::uint32_t), options.devices[0]);
        x_transfer = std::make_unique<PreparedTransfer>(options.devices[0], options.devices[1], projection.bytes(), options.transfer_mode);
        z_transfer = std::make_unique<PreparedTransfer>(options.devices[1], options.devices[0], home.regulators.bytes(), options.transfer_mode);
        parameter_transfer = std::make_unique<PreparedTransfer>(options.devices[0], options.devices[1], home.parameters.bytes(), options.transfer_mode);
        x_stage = mc::PinnedBuffer(product(x.size(), sizeof(std::uint16_t)));
        z_stage = mc::PinnedBuffer(product(z.size(), sizeof(float)));
        parameter_stage = mc::PinnedBuffer(home.parameters.bytes());
        std::memcpy(x_stage.data(), x.data(), x_stage.bytes());
        std::memcpy(z_stage.data(), z.data(), z_stage.bytes());
        auto* p = static_cast<float*>(parameter_stage.data());
        std::copy(theta.begin(), theta.end(), p);
        std::copy(phi.begin(), phi.end(), p + parameter_count);
        std::copy(tau.begin(), tau.end(), p + 2 * parameter_count);
        auto edit0 = home.values.edit();
        mc::DeviceGuard initial_device(options.devices[0]);
        mc::check(cudaMemcpyAsync(edit0.buffer().data, x_stage.data(), x_stage.bytes(), cudaMemcpyHostToDevice, home.stream.get()), "initial packed state upload");
        edit0.publish(x_generation); home.x_generation = x_generation; home.x_ready.record(home.stream);
        auto edit1 = remote.values.edit();
        mc::DeviceGuard guard1(options.devices[1]);
        if (edit1.buffer().bytes) mc::check(cudaMemsetAsync(edit1.buffer().data, 0, edit1.buffer().bytes, remote.stream.get()), "initial remote slots");
        edit1.publish(1); remote.x_generation = 1; remote.x_ready.record(remote.stream);
        remote.regulators.upload_async(z_stage.data(), z_stage.bytes(), remote.stream);
        remote.z_generation = z_generation; remote.z_ready.record(remote.stream);
        home.parameters.upload_async(parameter_stage.data(), parameter_stage.bytes(), home.stream);
        home.parameter_generation = parameter_generation; home.parameter_ready.record(home.stream);
        for (std::size_t i = 0; i < options.maximum_live_forwards; ++i)
            slots.push_back(std::make_unique<Slot>(options.devices, options.transfer_mode,
                value_count, gradient_count, condition_count, batch,
                std::array<std::size_t, 2>{groups[0].size(), groups[1].size()}));
        home.stream.synchronize(); remote.stream.synchronize();
        for (auto phase : {op::Phase::condition, op::Phase::primal, op::Phase::input_vjp,
                           op::Phase::condition_vjp, op::Phase::parameter_vjp}) {
            const auto& demand = semantics->phase(phase);
            for (const auto& read : demand.reads) {
                const bool context = gd::same_source(read.source, semantics->source(op::GatedProductSource::regulators).reference);
                requirements.push_back({phase, read, options.devices[context ? 1 : 0],
                    options.devices[context ? 0 : 1], context ? z_stage.bytes() : std::max(x_stage.bytes(), parameter_stage.bytes())});
            }
        }
        require(bool(validate_transfer_requirements(semantics->access(), requirements)), "phase transfer access validation failed");
        accounting.peer_forward = x_transfer->uses_peer(); accounting.peer_reverse = z_transfer->uses_peer();
        accounting.prepared_staging_bytes = x_stage.bytes() + z_stage.bytes() + parameter_stage.bytes() + projection_read_stage.bytes();
        auto account_transfer = [&](const PreparedTransfer& transfer) {
            if (!transfer.uses_peer()) accounting.prepared_staging_bytes += transfer.capacity();
        };
        account_transfer(*x_transfer); account_transfer(*z_transfer); account_transfer(*parameter_transfer);
        for (unsigned d = 0; d != 2; ++d) {
            const auto& device = *devices[d];
            accounting.prepared_device_bytes += device.structure->metadata_bytes()
                + product(device.structure->view().stored_values, sizeof(std::uint16_t))
                + device.regulators.bytes() + device.parameters.bytes() + device.groups.bytes()
                + device.arguments.bytes() + device.outputs.bytes() + device.argument_offsets.bytes() + device.coordinate_offsets.bytes();
        }
        for (const auto& slot : slots) {
            accounting.prepared_device_bytes += slot->remote_output.bytes() + slot->remote_gradient.bytes()
                + slot->remote_condition_gradient.bytes();
            accounting.prepared_staging_bytes += slot->cotangent_stage.bytes();
            for (unsigned d = 0; d != 2; ++d)
                accounting.prepared_device_bytes += slot->output[d]->bytes() + slot->cotangent[d]->bytes()
                    + slot->gradient[d]->bytes() + slot->saved[d]->bytes() + slot->status[d]->bytes();
            account_transfer(*slot->output_transfer); account_transfer(*slot->cotangent_transfer);
            account_transfer(*slot->gradient_transfer); account_transfer(*slot->parameter_gradient_transfer);
            account_transfer(*slot->condition_gradient_transfer);
        }
        accounting.prepared_device_bytes += projection.bytes() + projection_source_offsets.bytes()
            + projection_rows.bytes() + projection_coordinates.bytes() + user_offsets.bytes()
            + users.bytes() + projection_groups.bytes() + projection_reads.bytes();
    }
    ~DistributedProductState() noexcept {
        // Events and transfers are destroyed before streams/values. The streams
        // drain all final consumers before retained reads may be released.
        try { for (auto& device : devices) if (device) device->stream.synchronize(); }
        catch (...) { std::terminate(); }
    }
    Slot& checked(std::size_t index, std::uint64_t incarnation) {
        if (index >= slots.size() || !slots[index]->occupied || slots[index]->incarnation != incarnation)
            throw std::logic_error("stale distributed evaluation handle");
        return *slots[index];
    }
    void wait(Slot& slot, bool release_inputs) {
        try {
            for (unsigned d = 0; d != 2; ++d) if (slot.completed[d]->recorded()) slot.completed[d]->synchronize();
        } catch (...) { slot.failed = true; throw; }
        if (release_inputs && !slot.inputs_retired) {
            slot.reads[0].reset(); slot.reads[1].reset();
            slot.inputs_retired = true; --active_readers;
        }
    }
    void status(Slot& slot) {
        if (slot.failed) throw std::logic_error("distributed evaluation previously failed");
        try {
            wait(slot, false);
            for (unsigned d = 0; d != 2; ++d) {
                std::uint32_t result = 0;
                slot.status[d]->download_async(&result, sizeof(result), devices[d]->stream);
                devices[d]->stream.synchronize();
                if (result)
                    throw std::runtime_error("packed distributed operation rejected invalid/nonfinite numerical contents");
            }
        } catch (...) {
            slot.failed = true;
            throw;
        }
    }
    op::PackedGatedProductInput<std::uint16_t> input(Slot& slot, unsigned d) {
        const auto& device = *devices[d];
        return {slot.reads[d]->view(), device.topology,
            static_cast<const std::uint64_t*>(device.argument_offsets.data()),
            static_cast<const std::uint64_t*>(device.coordinate_offsets.data()),
            static_cast<const float*>(device.regulators.data()), device.theta(), device.phi(), device.tau(),
            batch, options.beta, d == 0};
    }
    op::PackedGatedProductSaved saved(Slot& slot, unsigned d) {
        auto* p = static_cast<float*>(slot.saved[d]->data());
        return {p, p + product(batch, devices[d]->topology.group_count)};
    }
    op::PackedGatedProductCapacities capacities(const Slot& slot, unsigned d) const {
        const auto& device = *devices[d];
        op::PackedGatedProductCapacities size{};
        size.values = product(device.structure->view().stored_values, sizeof(std::uint16_t));
        size.groups = device.groups.bytes(); size.arguments = device.arguments.bytes(); size.outputs = device.outputs.bytes();
        size.argument_value_offsets = device.argument_offsets.bytes(); size.coordinate_value_offsets = device.coordinate_offsets.bytes();
        size.z = device.regulators.bytes(); size.theta = size.phi = size.tau = product(parameter_count, sizeof(float));
        size.output = slot.output[d]->bytes(); size.output_gradient = slot.cotangent[d]->bytes();
        size.gradient_x = product(value_count, sizeof(float)); size.gradient_z = product(condition_count, sizeof(float));
        size.gradient_theta = size.gradient_phi = size.gradient_tau = product(parameter_count, sizeof(float));
        size.gate_arguments = size.products = slot.saved[d]->bytes() / 2; size.status = slot.status[d]->bytes();
        return size;
    }
    op::PackedGatedProductGradients gradient(Slot& slot, unsigned d) {
        auto* p = static_cast<float*>(slot.gradient[d]->data());
        return {p, p + value_count, p + value_count + condition_count,
            p + value_count + condition_count + parameter_count,
            p + value_count + condition_count + 2 * parameter_count};
    }
    const gd::SourceDescriptor& actual_source(op::GatedProductSource kind) const {
        const auto& expected = semantics->source(kind);
        for (const auto& source : actual_sources)
            if (gd::same_source(source.reference, expected.reference)) return source;
        throw std::logic_error("missing retained source association");
    }
    DeviceExtentBinding state_binding(unsigned d) const {
        const auto& device = *devices[d];
        const auto& source = actual_source(op::GatedProductSource::state);
        // The retained projection has its own content generation. Its source
        // revisions are independently checked by projection_stamp below.
        const auto physical = options.physical_orders[d];
        const auto epoch = options.structure_epochs[d];
        return DeviceExtentBinding{device.values.view().values,
            product(device.structure->view().stored_values, sizeof(std::uint16_t)),
            {memory::domain::device, static_cast<std::int16_t>(device.stream.device()), -1, 0},
            source.axis.domain, source.axis.logical_order, physical, options.physical_structures[d],
            epoch, device.x_generation, 1, device.x_generation, device.stream.device()};
    }
    ExtentRequirement state_requirement(unsigned d, std::uint64_t generation) const {
        const auto& source = actual_source(op::GatedProductSource::state);
        return {source.axis.domain, source.axis.logical_order, options.physical_orders[d], options.physical_structures[d],
            options.structure_epochs[d], generation, 1,
            product(devices[d]->structure->view().stored_values, sizeof(std::uint16_t)),
            alignof(std::uint16_t), devices[d]->stream.device()};
    }
    void validate_local(unsigned d) {
        const auto actual = state_binding(d);
        const auto required = state_requirement(d, devices[d]->x_generation);
        require(bool(validate_extent(required, actual)), "invalid placed packed source");
    }
    PackedProgramLaunch program_launch(Slot& slot, unsigned d) {
        PackedProgramLaunch call{};
        call.source = &actual_source(op::GatedProductSource::state);
        call.state = state_binding(d); call.expected = state_requirement(d, slot.reads[d]->generation());
        call.read = &*slot.reads[d]; call.readiness = &devices[d]->x_ready; call.stream = &devices[d]->stream;
        call.input = input(slot, d); call.capacities = capacities(slot, d);
        call.output = static_cast<float*>(slot.output[d]->data());
        call.output_gradient = static_cast<const float*>(slot.cotangent[d]->data());
        call.gradients = gradient(slot, d); call.saved = saved(slot, d);
        call.status = static_cast<std::uint32_t*>(slot.status[d]->data());
        return call;
    }
    struct AuxiliaryReadiness { std::array<const mc::Event*, 6> events{}; std::size_t count = 0; };
    AuxiliaryReadiness validate_auxiliaries(Slot& slot, unsigned d, bool backward) {
        const auto& device = *devices[d];
        std::array<bool, 6> needed{};
        for (auto phase : {op::Phase::condition, op::Phase::primal, op::Phase::input_vjp,
                           op::Phase::condition_vjp, op::Phase::parameter_vjp}) {
            const bool derivative = phase == op::Phase::input_vjp || phase == op::Phase::condition_vjp || phase == op::Phase::parameter_vjp;
            if (derivative != backward) continue;
            for (const auto& read : semantics->phase(phase).reads) {
                bool matched = false;
                for (unsigned role = 0; role != 6; ++role) {
                    const auto kind = static_cast<op::GatedProductSource>(role);
                    if (gd::same_source(read.source, actual_source(kind).reference)) { needed[role] = true; matched = true; }
                }
                require(matched, "phase read has no actual source binding");
            }
        }
        AuxiliaryReadiness ready;
        for (unsigned role = 1; role != 6; ++role) {
            if (!needed[role]) continue;
            const auto kind = static_cast<op::GatedProductSource>(role);
            const auto& source = actual_source(kind);
            const void* pointer = nullptr; std::size_t bytes = 0;
            std::uint64_t current = 0, expected = 0;
            const mc::Event* event = nullptr;
            if (kind == op::GatedProductSource::regulators) {
                pointer = device.regulators.data(); bytes = device.regulators.bytes();
                current = device.z_generation; expected = slot.context_generation; event = &device.z_ready;
            } else if (kind == op::GatedProductSource::result) {
                require(backward, "primal phase cannot read an unpublished result");
                pointer = slot.cotangent[d]->data(); bytes = slot.cotangent[d]->bytes();
                current = expected = slot.incarnation; event = slot.cotangent_ready[d].get();
            } else {
                pointer = kind == op::GatedProductSource::theta ? device.theta()
                    : (kind == op::GatedProductSource::phi ? device.phi() : device.tau());
                bytes = product(parameter_count, sizeof(float));
                current = device.parameter_generation; expected = slot.parameter_generation; event = &device.parameter_ready;
            }
            require(event && event->recorded() && event->device() == device.stream.device(), "unpublished auxiliary source readiness");
            DeviceExtentBinding actual{pointer, bytes,
                {memory::domain::device, static_cast<std::int16_t>(device.stream.device()), -1, 0},
                source.axis.domain, source.axis.logical_order, source.axis.logical_order,
                options.physical_structures[d], options.structure_epochs[d], current, 1, current, device.stream.device()};
            ExtentRequirement required{source.axis.domain, source.axis.logical_order, source.axis.logical_order,
                options.physical_structures[d], options.structure_epochs[d], expected, 1, bytes, alignof(float), device.stream.device()};
            require(bool(validate_extent(required, actual)), "stale or misplaced auxiliary source");
            ready.events[ready.count++] = event;
        }
        return ready;
    }
    void prepare_sources() {
        auto& home = *devices[0]; auto& remote = *devices[1];
        if (home.z_generation != z_generation) {
            z_transfer->enqueue(remote.regulators.view(), home.regulators.view(), home.regulators.bytes(), remote.stream, home.stream);
            home.z_ready.record(home.stream); home.z_generation = z_generation;
            accounting.condition_bytes += home.regulators.bytes();
        }
        if (remote.parameter_generation != parameter_generation) {
            parameter_transfer->enqueue(home.parameters.view(), remote.parameters.view(), home.parameters.bytes(), home.stream, remote.stream);
            remote.parameter_ready.record(remote.stream); remote.parameter_generation = parameter_generation;
            accounting.parameter_bytes += home.parameters.bytes();
        }
        const std::array stamp{x_generation, z_generation, gate_parameter_generation};
        if (projection_stamp != stamp) {
            mc::DeviceGuard guard(home.stream.device());
            projection_reads.zero_async(home.stream);
            if (projection.bytes()) {
                pack_remote<<<blocks(projection.bytes() / sizeof(std::uint16_t)), 256, 0, home.stream.get()>>>(
                    home.values.view().values, static_cast<std::uint16_t*>(projection.data()),
                    static_cast<const std::uint64_t*>(projection_source_offsets.data()),
                    static_cast<const std::uint64_t*>(projection_rows.data()),
                    static_cast<const std::uint64_t*>(projection_coordinates.data()),
                    static_cast<const std::uint64_t*>(user_offsets.data()), static_cast<const std::uint32_t*>(users.data()),
                    static_cast<const op::GatedProductGroup*>(projection_groups.data()),
                    static_cast<const float*>(home.regulators.data()), home.phi(), home.tau(),
                    options.topology.conditions, projection.bytes() / sizeof(std::uint16_t),
                    static_cast<std::uint32_t*>(projection_reads.data()));
                mc::check(cudaGetLastError(), "condition-first remote input packing");
            }
            projection_reads.download_async(projection_read_stage.data(), sizeof(std::uint32_t), home.stream);
            home.stream.synchronize(); // Explicit condition-summary cost, before participant transfer.
            const auto count = *static_cast<const std::uint32_t*>(projection_read_stage.data());
            accounting.projected_participant_values += count; ++accounting.projection_refreshes;
            if (count) {
                auto edit = remote.values.edit();
                x_transfer->enqueue(projection.view(), edit.buffer(), projection.bytes(), home.stream, remote.stream);
                edit.publish(++remote.x_generation); remote.x_ready.record(remote.stream);
                accounting.participant_bytes += projection.bytes();
            } else ++accounting.omitted_participant_transfers;
            projection_stamp = stamp;
        }
        validate_local(0); validate_local(1);
    }
};
}

DistributedProductTape::DistributedProductTape(std::shared_ptr<detail::DistributedProductState> state,
    std::size_t slot, std::uint64_t incarnation) : state_(std::move(state)), slot_(slot), incarnation_(incarnation) {}
DistributedProductTape::DistributedProductTape(DistributedProductTape&& other) noexcept
    : state_(std::move(other.state_)), slot_(other.slot_), incarnation_(other.incarnation_) {}
DistributedProductTape& DistributedProductTape::operator=(DistributedProductTape&& other) noexcept {
    if (this != &other) { release(); state_ = std::move(other.state_); slot_ = other.slot_; incarnation_ = other.incarnation_; }
    return *this;
}
DistributedProductTape::~DistributedProductTape() noexcept { release(); }
void DistributedProductTape::release() noexcept {
    if (!state_) return;
    try {
        auto& slot = state_->checked(slot_, incarnation_);
        state_->wait(slot, true); slot.occupied = false;
    } catch (...) { std::terminate(); }
    state_.reset();
}
void DistributedProductTape::wait() {
    if (!state_) throw std::logic_error("empty distributed tape");
    auto& slot = state_->checked(slot_, incarnation_); state_->wait(slot, slot.consumed);
}
bool DistributedProductTape::consumed() const {
    if (!state_) throw std::logic_error("empty distributed tape");
    return state_->checked(slot_, incarnation_).consumed;
}
std::uint64_t DistributedProductTape::parameter_generation() const {
    if (!state_) throw std::logic_error("empty distributed tape");
    return state_->checked(slot_, incarnation_).parameter_generation;
}

PreparedDistributedGatedProduct::PreparedDistributedGatedProduct(DistributedGatedProductOptions options,
    std::span<const std::uint16_t> x, std::span<const float> z,
    std::span<const float> theta, std::span<const float> phi, std::span<const float> tau)
    : state_(std::make_shared<detail::DistributedProductState>(std::move(options), x, z, theta, phi, tau)) {}
PreparedDistributedGatedProduct::~PreparedDistributedGatedProduct() = default;
PreparedDistributedGatedProduct::PreparedDistributedGatedProduct(PreparedDistributedGatedProduct&&) noexcept = default;
PreparedDistributedGatedProduct& PreparedDistributedGatedProduct::operator=(PreparedDistributedGatedProduct&&) noexcept = default;

DistributedProductTape PreparedDistributedGatedProduct::forward() {
    auto& state = *state_;
    if (state.poisoned) throw std::logic_error("distributed owner poisoned after failed publication");
    std::size_t index = 0;
    while (index < state.slots.size() && state.slots[index]->occupied) ++index;
    if (index == state.slots.size()) throw std::logic_error("distributed forward tape capacity exceeded");
    state.prepare_sources();
    auto& slot = *state.slots[index];
    slot.occupied = true; slot.consumed = false; slot.updated = false; slot.inputs_retired = false; slot.failed = false;
    ++slot.incarnation; slot.parameter_generation = state.parameter_generation;
    slot.context_generation = state.z_generation; slot.state_generation = state.x_generation;
    ++state.active_readers;
    try {
        std::array<PackedProgramLaunch, 2> calls;
        std::array<PackedProgramInvocation, 2> invocations;
        std::array<detail::DistributedProductState::AuxiliaryReadiness, 2> auxiliary;
        // Admit the complete distributed launch before either callback runs.
        // A late-invalid second binding must never leave a partial operation.
        for (unsigned d = 0; d != 2; ++d) {
            slot.reads[d].emplace(state.devices[d]->values.retain(state.devices[d]->x_generation));
            calls[d] = state.program_launch(slot, d);
            auxiliary[d] = state.validate_auxiliaries(slot, d, false);
            invocations[d] = {state.devices[d]->program.get(), &calls[d]};
        }
        require(admit_packed_programs(invocations, false) == program::program_status::success,
            "distributed prepared forward admission failed");
        for (unsigned d = 0; d != 2; ++d) {
            auto& device = *state.devices[d];
            slot.output[d]->zero_async(device.stream); slot.status[d]->zero_async(device.stream);
            for (std::size_t i = 0; i != auxiliary[d].count; ++i) device.stream.wait(*auxiliary[d].events[i]);
        }
        require(execute_packed_programs(invocations, false) == program::program_status::success,
            "distributed prepared forward failed");
        state.accounting.prepared_program_forwards += invocations.size();
        for (unsigned d = 0; d != 2; ++d) slot.completed[d]->record(state.devices[d]->stream);
        slot.output_transfer->enqueue(slot.output[1]->view(), slot.remote_output.view(), slot.remote_output.bytes(),
            state.devices[1]->stream, state.devices[0]->stream);
        add(slot.output[0]->view(), slot.remote_output.view(), state.value_count, slot.status[0]->view(), state.devices[0]->stream);
        slot.completed[0]->record(state.devices[0]->stream);
        state.accounting.output_bytes += slot.remote_output.bytes();
    } catch (...) {
        for (auto& device : state.devices) device->stream.synchronize();
        state.wait(slot, true); slot.occupied = false; throw;
    }
    return DistributedProductTape(state_, index, slot.incarnation);
}

std::vector<float> PreparedDistributedGatedProduct::values(DistributedProductTape& tape) const {
    require(tape.state_.get() == state_.get(), "foreign distributed tape");
    auto& slot = state_->checked(tape.slot_, tape.incarnation_); state_->status(slot);
    std::vector<float> values(state_->value_count);
    slot.output[0]->download_async(values.data(), product(values.size(), sizeof(float)), state_->devices[0]->stream);
    state_->devices[0]->stream.synchronize(); return values;
}

void PreparedDistributedGatedProduct::backward(DistributedProductTape& tape, std::span<const float> dy) {
    require(tape.state_.get() == state_.get(), "foreign distributed tape");
    auto& state = *state_; auto& slot = state.checked(tape.slot_, tape.incarnation_);
    if (state.poisoned) throw std::logic_error("distributed owner poisoned after failed publication");
    require(dy.size() == state.value_count, "distributed cotangent extent mismatch");
    if (slot.failed) throw std::logic_error("distributed backward previously failed");
    if (slot.consumed || slot.inputs_retired) throw std::logic_error("distributed tape already consumed");
    require(slot.parameter_generation == state.parameter_generation, "stale distributed parameter generation");
    require(slot.context_generation == state.z_generation && slot.state_generation == state.x_generation,
        "stale distributed forward source contents");
    for (auto value : dy) require(std::isfinite(value), "nonfinite distributed cotangent");
    state.status(slot);
    try {
        std::memcpy(slot.cotangent_stage.data(), dy.data(), slot.cotangent_stage.bytes());
        slot.cotangent[0]->upload_async(slot.cotangent_stage.data(), slot.cotangent_stage.bytes(), state.devices[0]->stream);
        slot.cotangent_ready[0]->record(state.devices[0]->stream);
        slot.cotangent_transfer->enqueue(slot.cotangent[0]->view(), slot.cotangent[1]->view(), slot.cotangent_stage.bytes(),
            state.devices[0]->stream, state.devices[1]->stream);
        slot.cotangent_ready[1]->record(state.devices[1]->stream);
        std::array<PackedProgramLaunch, 2> calls;
        std::array<PackedProgramInvocation, 2> invocations;
        std::array<detail::DistributedProductState::AuxiliaryReadiness, 2> auxiliary;
        for (unsigned d = 0; d != 2; ++d) {
            calls[d] = state.program_launch(slot, d);
            auxiliary[d] = state.validate_auxiliaries(slot, d, true);
            invocations[d] = {state.devices[d]->program.get(), &calls[d]};
        }
        require(admit_packed_programs(invocations, true) == program::program_status::success,
            "distributed prepared backward admission failed");
        for (unsigned d = 0; d != 2; ++d) {
            auto& device = *state.devices[d]; slot.gradient[d]->zero_async(device.stream);
            for (std::size_t i = 0; i != auxiliary[d].count; ++i) device.stream.wait(*auxiliary[d].events[i]);
        }
        require(execute_packed_programs(invocations, true) == program::program_status::success,
            "distributed prepared backward failed");
        state.accounting.prepared_program_backwards += invocations.size();
        for (unsigned d = 0; d != 2; ++d) slot.completed[d]->record(state.devices[d]->stream);
        const auto offset = product(state.value_count, sizeof(float));
        const auto bytes = product(state.condition_count, sizeof(float));
        const auto parameter_bytes = product(product(state.parameter_count, 3), sizeof(float));
        const auto parameter_offset = slot.gradient[1]->bytes() - parameter_bytes;
        slot.gradient_transfer->enqueue(slot.gradient[1]->view().subview(0, offset),
            slot.remote_gradient.view().subview(0, offset), offset,
            state.devices[1]->stream, state.devices[0]->stream);
        slot.parameter_gradient_transfer->enqueue(slot.gradient[1]->view().subview(parameter_offset, parameter_bytes),
            slot.remote_gradient.view().subview(offset, parameter_bytes), parameter_bytes,
            state.devices[1]->stream, state.devices[0]->stream);
        slot.condition_gradient_transfer->enqueue(slot.gradient[0]->view().subview(offset, bytes),
            slot.remote_condition_gradient.view(), bytes, state.devices[0]->stream, state.devices[1]->stream);
        add(slot.gradient[0]->view().subview(0, offset), slot.remote_gradient.view().subview(0, offset),
            state.value_count, slot.status[0]->view(), state.devices[0]->stream);
        add(slot.gradient[0]->view().subview(parameter_offset, parameter_bytes),
            slot.remote_gradient.view().subview(offset, parameter_bytes),
            product(state.parameter_count, 3), slot.status[0]->view(), state.devices[0]->stream);
        add(slot.gradient[1]->view().subview(offset, bytes), slot.remote_condition_gradient.view(),
            state.condition_count, slot.status[1]->view(), state.devices[1]->stream);
        slot.completed[0]->record(state.devices[0]->stream); slot.completed[1]->record(state.devices[1]->stream);
        state.accounting.backward_bytes += 2 * slot.cotangent_stage.bytes() + offset + parameter_bytes + bytes;
        // Publish backward only after both programs and all owner-directed
        // reductions have been queued. Completion/status are checked by every
        // gradient/update consumer before the contents become observable.
        slot.consumed = true;
    } catch (...) {
        slot.failed = true;
        for (auto& device : state.devices) device->stream.synchronize(); state.wait(slot, true); throw;
    }
}

op::GatedProductHostBackward<float> PreparedDistributedGatedProduct::gradients(DistributedProductTape& tape) const {
    require(tape.state_.get() == state_.get(), "foreign distributed tape");
    auto& state = *state_; auto& slot = state.checked(tape.slot_, tape.incarnation_);
    if (state.poisoned) throw std::logic_error("distributed owner poisoned after failed publication");
    require(!slot.failed && slot.consumed, "distributed backward has not succeeded");
    state.status(slot); tape.wait();
    std::vector<float> all(state.gradient_count);
    slot.gradient[0]->download_async(all.data(), product(all.size(), sizeof(float)), state.devices[0]->stream);
    state.devices[0]->stream.synchronize();
    op::GatedProductHostBackward<float> out;
    auto at = all.begin();
    out.dx.assign(at, at + state.value_count); at += state.value_count;
    out.dz.resize(state.condition_count);
    slot.gradient[1]->download_async(out.dz.data(), product(out.dz.size(), sizeof(float)), state.devices[1]->stream,
        product(state.value_count, sizeof(float)));
    state.devices[1]->stream.synchronize(); at += state.condition_count;
    out.dtheta.assign(at, at + state.parameter_count); at += state.parameter_count;
    out.dphi.assign(at, at + state.parameter_count); at += state.parameter_count;
    out.dtau.assign(at, at + state.parameter_count);
    return out;
}

void PreparedDistributedGatedProduct::update(DistributedProductTape& tape, float rate) {
    update(tape, ParameterLearningRates{rate, 0.0f, 0.0f});
}

void PreparedDistributedGatedProduct::update(DistributedProductTape& tape, ParameterLearningRates rates) {
    require(tape.state_.get() == state_.get(), "foreign distributed tape");
    auto& state = *state_; auto& slot = state.checked(tape.slot_, tape.incarnation_);
    if (state.poisoned) throw std::logic_error("distributed owner poisoned after failed publication");
    for (auto rate : {rates.theta, rates.phi, rates.tau})
        require(std::isfinite(rate) && rate >= 0, "invalid distributed learning rate");
    require(!slot.failed && slot.consumed, "update requires successful backward");
    state.status(slot); tape.wait();
    if (state.active_readers) throw std::logic_error("unresolved distributed forward prevents parameter update");
    if (slot.updated || slot.parameter_generation != state.parameter_generation)
        throw std::logic_error("duplicate or stale distributed update");
    if (state.parameter_generation == std::numeric_limits<std::uint64_t>::max())
        throw std::overflow_error("distributed parameter generation exhausted");
    const bool gate_update = rates.phi != 0.0f || rates.tau != 0.0f;
    if (gate_update && state.gate_parameter_generation == std::numeric_limits<std::uint64_t>::max())
        throw std::overflow_error("distributed gate generation exhausted");
    auto& home = *state.devices[0]; auto& remote = *state.devices[1];
    mc::DeviceGuard guard(home.stream.device());
    const auto count = product(state.parameter_count, 3);
    const auto* gradient = state.gradient(slot, 0).theta;
    try {
        // The remote-gradient packet is dead after reduction. Reuse its
        // parameter section for a proposal so numerical failure cannot partly
        // overwrite the canonical parameter version.
        auto proposal = slot.remote_gradient.view().subview(product(state.value_count, sizeof(float)), home.parameters.bytes());
        sgd_proposal<<<blocks(count), 256, 0, home.stream.get()>>>(
            static_cast<const float*>(home.parameters.data()), static_cast<float*>(proposal.data),
            gradient, state.parameter_count, rates, static_cast<std::uint32_t*>(slot.status[0]->data()));
        mc::check(cudaGetLastError(), "distributed parameter proposal");
        slot.completed[0]->record(home.stream);
        state.status(slot);
        mc::check(cudaMemcpyAsync(home.parameters.data(), proposal.data, home.parameters.bytes(),
            cudaMemcpyDeviceToDevice, home.stream.get()), "canonical distributed parameter publication");
        home.parameter_ready.record(home.stream); home.parameter_generation = ++state.parameter_generation;
        if (gate_update) ++state.gate_parameter_generation;
        state.parameter_transfer->enqueue(home.parameters.view(), remote.parameters.view(), home.parameters.bytes(), home.stream, remote.stream);
        remote.parameter_ready.record(remote.stream); remote.parameter_generation = state.parameter_generation;
        state.accounting.replica_refresh_bytes += home.parameters.bytes(); slot.updated = true;
    } catch (...) { state.poisoned = true; throw; }
}

std::array<std::vector<float>, 3> PreparedDistributedGatedProduct::parameters() const {
    auto& state = *state_; auto& home = *state.devices[0];
    std::vector<float> all(product(state.parameter_count, 3));
    home.parameters.download_async(all.data(), product(all.size(), sizeof(float)), home.stream); home.stream.synchronize();
    return {std::vector<float>(all.begin(), all.begin() + state.parameter_count),
        std::vector<float>(all.begin() + state.parameter_count, all.begin() + 2 * state.parameter_count),
        std::vector<float>(all.begin() + 2 * state.parameter_count, all.end())};
}
std::uint64_t PreparedDistributedGatedProduct::parameter_generation() const { return state_->parameter_generation; }
void PreparedDistributedGatedProduct::synchronize() const {
    for (const auto& device : state_->devices) device->stream.synchronize();
}

void PreparedDistributedGatedProduct::upload_values(std::span<const std::uint16_t> values, std::uint64_t generation) {
    auto& state = *state_; auto& home = *state.devices[0];
    if (state.poisoned) throw std::logic_error("distributed owner poisoned after failed publication");
    require(values.size_bytes() == state.x_stage.bytes() && generation > state.x_generation, "invalid state update extent/generation");
    if (state.active_readers) throw std::logic_error("retained distributed forward prevents state write");
    home.x_ready.synchronize(); std::memcpy(state.x_stage.data(), values.data(), values.size_bytes());
    auto edit = home.values.edit(); mc::DeviceGuard guard(home.stream.device());
    mc::check(cudaMemcpyAsync(edit.buffer().data, state.x_stage.data(), state.x_stage.bytes(), cudaMemcpyHostToDevice, home.stream.get()), "packed state update");
    edit.publish(generation); home.x_ready.record(home.stream);
    home.x_generation = state.x_generation = generation;
}
void PreparedDistributedGatedProduct::upload_regulators(std::span<const float> values, std::uint64_t generation) {
    auto& state = *state_; auto& remote = *state.devices[1];
    if (state.poisoned) throw std::logic_error("distributed owner poisoned after failed publication");
    require(values.size_bytes() == state.z_stage.bytes() && generation > state.z_generation, "invalid regulator update extent/generation");
    if (state.active_readers) throw std::logic_error("retained distributed forward prevents regulator write");
    remote.z_ready.synchronize(); std::memcpy(state.z_stage.data(), values.data(), values.size_bytes());
    remote.regulators.upload_async(state.z_stage.data(), state.z_stage.bytes(), remote.stream);
    remote.z_ready.record(remote.stream); remote.z_generation = state.z_generation = generation;
}
DistributedTransferStats PreparedDistributedGatedProduct::stats() const { return state_->accounting; }

} // namespace cellerator::execution::dependency
