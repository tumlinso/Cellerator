#include "gated_product_fixture.hh"
#include <Cellerator/execution/dependency/packed_program.cuh>

#include <charconv>
#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace ce = cellerator;
namespace fixture = ce::test::execution_dependency;
namespace dep = ce::execution::dependency;
namespace pg = ce::execution::program;
namespace bm = ce::tensor::block_masked;
namespace mc = ce::memory::cuda;
namespace op = ce::ops::mechanism;

namespace {
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
using Buffer = std::unique_ptr<mc::DeviceBuffer>;
Buffer allocate(std::size_t n, int device) { return std::make_unique<mc::DeviceBuffer>(n, device); }
template<class T> Buffer upload(std::span<const T> values, const mc::Stream& stream) {
    auto result = allocate(values.size_bytes(), stream.device());
    if (!values.empty()) result->upload_async(values.data(), values.size_bytes(), stream);
    return result;
}

struct Local {
    mc::Stream stream;
    std::shared_ptr<const bm::DeviceStructure> structure;
    bm::NativeTensor<std::uint16_t> values;
    mc::Event ready, unrecorded;
    std::optional<bm::RetainedRead<std::uint16_t>> lease;
    Buffer groups, arguments, outputs, argument_offsets, coordinate_offsets;
    Buffer z, theta, phi, tau, y, dy, dx, dz, dtheta, dphi, dtau, gate, products, status;
    op::PackedGatedProductInput<std::uint16_t> input{};
    op::PackedGatedProductCapacities cap{};
    op::PackedGatedProductSaved saved{};
    op::PackedGatedProductGradients gradients{};
    std::unique_ptr<dep::PreparedPackedProgram> program;
    dep::PackedProgramLaunch call{};

    Local(fixture::GatedProductFixture& f, int device, unsigned slot)
      : stream(device), structure(std::make_shared<bm::DeviceStructure>(f.structure, stream)),
        values(structure), ready(device), unrecorded(device) {
        auto topology = f.semantics->topology();
        std::vector<std::uint64_t> indices;
        std::vector<op::GatedProductGroup> host_groups;
        for (std::size_t i = 0; i < f.group_slots.size(); ++i)
            if (f.group_slots[i] == slot) { indices.push_back(i); host_groups.push_back(topology.groups[i]); }
        auto local_topology = topology;
        local_topology.groups = host_groups.data(); local_topology.group_count = host_groups.size();
        auto argument_map = op::prepare_argument_value_offsets(structure->host(), local_topology);
        auto coordinate_map = slot == 0 ? op::prepare_coordinate_value_offsets(structure->host(), topology.coordinates)
                                         : std::vector<std::uint64_t>{};
        groups = upload<op::GatedProductGroup>(host_groups, stream);
        arguments = upload<std::uint64_t>({topology.arguments, static_cast<std::size_t>(topology.argument_count)}, stream);
        outputs = upload<op::OutputIncidence>({topology.outputs, static_cast<std::size_t>(topology.output_count)}, stream);
        argument_offsets = upload<std::uint64_t>(argument_map, stream);
        coordinate_offsets = upload<std::uint64_t>(coordinate_map, stream);
        const auto coordinate_bytes = fixture::fixture_batch * topology.coordinates * sizeof(float);
        const auto condition_bytes = fixture::fixture_batch * topology.conditions * 2 * sizeof(float);
        const auto parameter_bytes = topology.parameters * sizeof(float);
        const auto saved_bytes = fixture::fixture_batch * host_groups.size() * sizeof(float);
        z = allocate(condition_bytes, device); theta = allocate(parameter_bytes, device);
        phi = allocate(parameter_bytes, device); tau = allocate(parameter_bytes, device);
        y = allocate(coordinate_bytes, device); dy = allocate(coordinate_bytes, device);
        dx = allocate(coordinate_bytes, device); dz = allocate(condition_bytes, device);
        dtheta = allocate(parameter_bytes, device); dphi = allocate(parameter_bytes, device);
        dtau = allocate(parameter_bytes, device); gate = allocate(saved_bytes, device);
        products = allocate(saved_bytes, device); status = allocate(sizeof(std::uint32_t), device);
        auto edit = values.edit();
        {
            const auto destination = edit.buffer();
            mc::DeviceGuard guard(device);
            mc::check(cudaMemcpyAsync(destination.data, f.packed_values.data(), destination.bytes,
                cudaMemcpyHostToDevice, stream.get()), "initialize packed admission fixture");
        }
        z->upload_async(f.regulators.data(), z->bytes(), stream);
        theta->upload_async(f.theta.data(), theta->bytes(), stream);
        phi->upload_async(f.phi.data(), phi->bytes(), stream);
        tau->upload_async(f.tau.data(), tau->bytes(), stream);
        dy->upload_async(f.cotangents.data(), dy->bytes(), stream);
        std::vector<float> saved_gate(saved_bytes / sizeof(float), 0.75f);
        std::vector<float> saved_product(saved_bytes / sizeof(float), 1.0f);
        gate->upload_async(saved_gate.data(), saved_bytes, stream);
        products->upload_async(saved_product.data(), saved_bytes, stream);
        edit.publish(1); ready.record(stream);
        lease.emplace(values.retain(1));
        input.x = lease->view();
        input.topology = {static_cast<const op::GatedProductGroup*>(groups->data()), host_groups.size(),
            static_cast<const std::uint64_t*>(arguments->data()), topology.argument_count,
            static_cast<const op::OutputIncidence*>(outputs->data()), topology.output_count,
            topology.coordinates, topology.conditions, topology.parameters};
        input.argument_value_offsets = static_cast<const std::uint64_t*>(argument_offsets->data());
        input.coordinate_value_offsets = static_cast<const std::uint64_t*>(coordinate_offsets->data());
        input.z = static_cast<const float*>(z->data()); input.theta = static_cast<const float*>(theta->data());
        input.phi = static_cast<const float*>(phi->data()); input.tau = static_cast<const float*>(tau->data());
        input.batch = fixture::fixture_batch; input.beta = 0.25f; input.include_baseline = slot == 0;
        cap.values = structure->view().stored_values * sizeof(std::uint16_t);
        cap.groups = groups->bytes(); cap.arguments = arguments->bytes(); cap.outputs = outputs->bytes();
        cap.argument_value_offsets = argument_offsets->bytes(); cap.coordinate_value_offsets = coordinate_offsets->bytes();
        cap.z = z->bytes(); cap.theta = theta->bytes(); cap.phi = phi->bytes(); cap.tau = tau->bytes();
        cap.output = y->bytes(); cap.output_gradient = dy->bytes(); cap.gradient_x = dx->bytes(); cap.gradient_z = dz->bytes();
        cap.gradient_theta = dtheta->bytes(); cap.gradient_phi = dphi->bytes(); cap.gradient_tau = dtau->bytes();
        cap.gate_arguments = gate->bytes(); cap.products = products->bytes(); cap.status = status->bytes();
        saved = {static_cast<float*>(gate->data()), static_cast<float*>(products->data())};
        gradients = {static_cast<float*>(dx->data()), static_cast<float*>(dz->data()), static_cast<float*>(dtheta->data()),
            static_cast<float*>(dphi->data()), static_cast<float*>(dtau->data())};
        const auto options = f.options({device, device + 1}, dep::TransferMode::staged);
        const auto atom = slot ? fixture::packed_program_identity::state_projection() : fixture::packed_program_identity::state_home();
        const auto physical_order = options.physical_orders[slot];
        const auto physical_structure = options.physical_structures[slot];
        const auto epoch = options.structure_epochs[slot];
        program = std::make_unique<dep::PreparedPackedProgram>(dep::PackedProgramPreparation{
            f.semantics, structure, indices, atom, physical_structure, physical_order, epoch, 1, input});
        const auto& source = f.semantics->source(op::GatedProductSource::state);
        call.source = &source;
        call.state = {input.x.values, cap.values, {ce::memory::domain::device, static_cast<std::int16_t>(device), -1, 0},
            source.axis.domain, source.axis.logical_order, physical_order, physical_structure, epoch, 1, 1, 1, device};
        call.expected = {source.axis.domain, source.axis.logical_order, physical_order, physical_structure,
            epoch, 1, 1, cap.values, alignof(std::uint16_t), device};
        call.read = &*lease; call.readiness = &ready; call.stream = &stream; call.input = input; call.capacities = cap;
        call.output = static_cast<float*>(y->data()); call.output_gradient = static_cast<const float*>(dy->data());
        call.gradients = gradients; call.saved = saved; call.status = static_cast<std::uint32_t*>(status->data());
        stream.synchronize(); // uploaded topology is fully prepared before its host vectors die
    }
};

bool accepted(pg::program_status status) { return status == pg::program_status::success; }
void rejects_without_dispatch(Local& first, Local& second, const dep::PackedProgramLaunch& bad, const char* message) {
    const std::array<dep::PackedProgramInvocation, 2> calls{{
        {first.program.get(), &first.call}, {second.program.get(), &bad}}};
    require(!accepted(dep::admit_packed_programs(calls, false)), message);
    require(!accepted(dep::execute_packed_programs(calls, false)), message);
    // Also exercise the actual prepared_program_v2 admission pass.
    require(!accepted(second.program->forward(bad)), message);
}

int parse(int argc, char** argv, std::string_view name, int fallback) {
    for (int i = 1; i < argc; ++i) if (std::string_view(argv[i]) == name) {
        if (++i >= argc) throw std::invalid_argument("missing device ordinal");
        const std::string_view value(argv[i]); int result = -1;
        const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
        if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() || result < 0)
            throw std::invalid_argument("invalid device ordinal");
        return result;
    }
    return fallback;
}
} // namespace

int main(int argc, char** argv) {
    try {
        const int device0 = parse(argc, argv, "--device0", 0);
        const int device1 = parse(argc, argv, "--device1", 1);
        require(device0 != device1, "distinct devices required");
        fixture::GatedProductFixture data;
        Local first(data, device0, 0), second(data, device1, 1);
        require(accepted(first.program->admit_forward(first.call)) && accepted(second.program->admit_forward(second.call)),
            "valid prepared forward bindings must pass admission");
        require(accepted(first.program->admit_backward(first.call)) && accepted(second.program->admit_backward(second.call)),
            "valid prepared backward bindings must pass admission");
        require(first.program->forward_fragment().program && second.program->backward_fragment().program,
            "real prepared fragments are attached");
        const auto home_coverage = first.program->coverage_coordinates();
        const auto remote_coverage = second.program->coverage_coordinates();
        const std::array<std::uint64_t, 4> remote_expected{{1, 2, 3, 4}};
        require(home_coverage.size() == fixture::fixture_coordinates && home_coverage.front() == 0
            && home_coverage.back() == fixture::fixture_coordinates - 1,
            "baseline atom does not cover the full state axis");
        require(std::equal(remote_coverage.begin(), remote_coverage.end(),
                           remote_expected.begin(), remote_expected.end()),
            "remote atom coverage does not match local group participants");
        std::vector<float> expected_output(fixture::fixture_batch * fixture::fixture_coordinates, 17.0f);
        const std::uint32_t expected_status = 0;
        for (Local* local : {&first, &second}) {
            local->y->upload_async(expected_output.data(), local->y->bytes(), local->stream);
            local->dx->upload_async(expected_output.data(), local->dx->bytes(), local->stream);
            local->status->upload_async(&expected_status, sizeof(expected_status), local->stream);
            local->stream.synchronize();
        }
        auto bad = second.call;
        auto source = *bad.source; ++source.reference.identity.low; bad.source = &source;
        rejects_without_dispatch(first, second, bad, "foreign source accepted");
        bad = second.call; ++bad.state.domain.low;
        rejects_without_dispatch(first, second, bad, "wrong domain accepted");
        bad = second.call; ++bad.state.where.device_ordinal;
        rejects_without_dispatch(first, second, bad, "wrong device accepted");
        bad = second.call; ++bad.state.structure_epoch;
        rejects_without_dispatch(first, second, bad, "stale structure epoch accepted");
        bad = second.call; ++bad.state.value_generation; ++bad.state.ready_generation;
        rejects_without_dispatch(first, second, bad, "stale value generation accepted");
        bad = second.call; ++bad.expected.placement_revision;
        rejects_without_dispatch(first, second, bad, "stale placement revision accepted");
        bad = second.call; bad.readiness = &second.unrecorded;
        rejects_without_dispatch(first, second, bad, "unrecorded readiness accepted");
        bad = second.call; bad.read = first.call.read;
        rejects_without_dispatch(first, second, bad, "foreign read lease accepted");
        bad = second.call; --bad.capacities.values;
        rejects_without_dispatch(first, second, bad, "short state extent accepted");
        bad = second.call; bad.capacities.output = 0;
        rejects_without_dispatch(first, second, bad, "short output extent accepted");
        bad = second.call; bad.output = const_cast<float*>(bad.input.z);
        rejects_without_dispatch(first, second, bad, "overlapping output accepted");
        bad = second.call; bad.capacities.gradient_theta = 0;
        require(!accepted(second.program->admit_backward(bad)) && !accepted(second.program->backward(bad)),
            "short backward gradient accepted");
        bad = second.call; bad.saved = {};
        require(!accepted(second.program->admit_backward(bad)) && !accepted(second.program->backward(bad)),
            "missing backward saved context accepted");
        const std::array<dep::PackedProgramInvocation, 2> backward_calls{{
            {first.program.get(), &first.call}, {second.program.get(), &bad}}};
        require(!accepted(dep::execute_packed_programs(backward_calls, true)),
            "late invalid backward batch accepted");
        for (Local* local : {&first, &second}) {
            std::vector<float> actual_output(expected_output.size());
            std::vector<float> actual_dx(expected_output.size());
            std::uint32_t actual_status = 0;
            local->y->download_async(actual_output.data(), local->y->bytes(), local->stream);
            local->dx->download_async(actual_dx.data(), local->dx->bytes(), local->stream);
            local->status->download_async(&actual_status, sizeof(actual_status), local->stream);
            local->stream.synchronize();
            require(actual_output == expected_output && actual_dx == expected_output && actual_status == expected_status,
                "rejected batch admission changed output or status");
        }
        // Positive controls prove sentinel sensitivity and exercise actual
        // batch dispatch across devices, including host-device switching.
        for (Local* local : {&first, &second}) {
            local->dz->zero_async(local->stream); local->dtheta->zero_async(local->stream);
            local->dphi->zero_async(local->stream); local->dtau->zero_async(local->stream);
        }
        const std::array<dep::PackedProgramInvocation, 2> valid_calls{{
            {first.program.get(), &first.call}, {second.program.get(), &second.call}}};
        require(accepted(dep::execute_packed_programs(valid_calls, false)), "valid forward batch failed");
        require(accepted(dep::execute_packed_programs(valid_calls, true)), "valid backward batch failed");
        for (Local* local : {&first, &second}) {
            std::vector<float> actual_output(expected_output.size()), actual_dx(expected_output.size());
            std::uint32_t actual_status = 1;
            local->y->download_async(actual_output.data(), local->y->bytes(), local->stream);
            local->dx->download_async(actual_dx.data(), local->dx->bytes(), local->stream);
            local->status->download_async(&actual_status, sizeof(actual_status), local->stream);
            local->stream.synchronize();
            require(actual_status == 0 && actual_output != expected_output && actual_dx != expected_output,
                "valid program did not change forward/backward sentinels");
        }
        std::cout << "prepared packed program admission passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
