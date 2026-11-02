#include <Cellerator/memory/cuda/resources.hh>
#include <Cellerator/ops/mechanism/packed_gated_product.cuh>
#include <Cellerator/tensor/block_masked/owner.hh>

#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace mc = cellerator::memory::cuda;
namespace bm = cellerator::tensor::block_masked;
namespace op = cellerator::ops::mechanism;
namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
template<class Function> void rejects(Function&& function) {
    bool failed = false;
    try { function(); } catch (const std::exception&) { failed = true; }
    require(failed, "invalid owner operation was accepted");
}
template<class T> mc::DeviceBuffer upload(const std::vector<T>& values, const mc::Stream& stream) {
    mc::DeviceBuffer result(values.size() * sizeof(T), stream.device());
    result.upload_async(values.data(), result.bytes(), stream); return result;
}
std::uint16_t half_bits(float value) {
    const auto half = __float2half_rn(value);
    std::uint16_t result; std::memcpy(&result, &half, sizeof(result)); return result;
}
void compare(const std::vector<float>& actual, const std::vector<double>& expected, const char* label) {
    require(actual.size() == expected.size(), "reference shape mismatch");
    for (std::size_t i = 0; i < actual.size(); ++i)
        if (!std::isfinite(actual[i]) || std::abs(actual[i] - expected[i]) > 3e-5 * (1 + std::abs(expected[i])))
            throw std::runtime_error(std::string(label) + " differs at " + std::to_string(i));
}
void check_host_buffer_admission() {
    const auto structure = bm::Structure::from_rows(1, {{0}});
    float values[20]{};
    op::GatedProductGroup group{0, 1, 0, 1, 0, 0, 1};
    std::uint64_t argument = 0, offset = 0;
    op::OutputIncidence incidence{0, 1};
    std::uint32_t status = 0x12345678;
    const op::GatedProductTopologyView topology{&group, 1, &argument, 1, &incidence, 1, 1, 1, 1};
    op::PackedGatedProductInput<float> input{{structure.view(), values}, topology, &offset, &offset,
        values + 1, values + 3, values + 4, values + 5, 1, .25f, true};
    const op::PackedGatedProductSaved saved{values + 14, values + 15};
    op::PackedGatedProductGradients gradients{values + 8, values + 9, values + 11, values + 12, values + 13};
    op::PackedGatedProductCapacities capacity;
    capacity.values = sizeof(float); capacity.groups = sizeof(group);
    capacity.arguments = sizeof(argument); capacity.outputs = sizeof(incidence);
    capacity.argument_value_offsets = capacity.coordinate_value_offsets = sizeof(offset);
    capacity.z = 2 * sizeof(float); capacity.theta = capacity.phi = capacity.tau = sizeof(float);
    capacity.output = capacity.output_gradient = capacity.gradient_x = sizeof(float);
    capacity.gradient_z = 2 * sizeof(float);
    capacity.gradient_theta = capacity.gradient_phi = capacity.gradient_tau = sizeof(float);
    capacity.gate_arguments = capacity.products = sizeof(float); capacity.status = sizeof(status);
    require(op::validate_packed_forward_buffers(input, capacity, values + 6, saved, &status),
            "valid host forward admission rejected");
    require(op::validate_packed_backward_buffers(input, capacity, values + 7, gradients, saved, &status),
            "valid host backward admission rejected");
    auto shortened = capacity; --shortened.output;
    require(!op::validate_packed_forward_buffers(input, shortened, values + 6, saved, &status),
            "host admission accepted short output");
    require(!op::validate_packed_forward_buffers(input, capacity, values, saved, &status),
            "host admission accepted aliased input/output");
    shortened = capacity; --shortened.gradient_z;
    require(!op::validate_packed_backward_buffers(input, shortened, values + 7, gradients, saved, &status),
            "host admission accepted short VJP");
    gradients.theta = gradients.x;
    require(!op::validate_packed_backward_buffers(input, capacity, values + 7, gradients, saved, &status),
            "host admission accepted aliased VJPs");
    auto invalid = input; invalid.argument_value_offsets = nullptr;
    require(!op::validate_packed_forward_buffers(invalid, capacity, values + 6, saved, &status),
            "host admission skipped input validation");
    require(!op::validate_packed_forward_buffers(input, capacity, values + 6, {saved.gate_arguments, nullptr}, &status),
            "host admission accepted incomplete saved values");
    require(status == 0x12345678, "host admission modified status");
}
struct Reference { std::vector<double> y, dx, dz, theta, phi, tau; };
// Deliberately ordinary double arithmetic, canonical indexing and O(arity^2)
// exclusive products; independent of packed addressing and device scans.
Reference reference(op::GatedProductTopologyView topology, std::uint64_t batch,
                    const std::vector<float>& x, const std::vector<float>& z,
                    const std::vector<float>& theta, const std::vector<float>& phi,
                    const std::vector<float>& tau, const std::vector<float>& dy, float beta) {
    Reference out{std::vector<double>(x.size()), std::vector<double>(x.size()),
        std::vector<double>(z.size()), std::vector<double>(theta.size()),
        std::vector<double>(theta.size()), std::vector<double>(theta.size())};
    for (std::size_t i = 0; i < x.size(); ++i) {
        out.y[i] = beta == 0 ? 0 : -double(beta) * x[i]; out.dx[i] = -double(beta) * dy[i];
    }
    for (std::uint64_t row = 0; row < batch; ++row)
        for (std::uint64_t m = 0; m < topology.group_count; ++m) {
            const auto group = topology.groups[m]; const auto k = group.parameter_index;
            const auto c = 2 * (row * topology.conditions + group.condition_index);
            const double u = double(z[c]) + double(phi[k]) * z[c + 1] - tau[k];
            if (u <= 0) continue;
            double product = 1, upstream = 0;
            for (std::uint64_t i = 0; i < group.argument_count; ++i)
                product *= x[row * topology.coordinates + topology.arguments[group.argument_begin + i]];
            for (std::uint64_t i = 0; i < group.output_count; ++i) {
                const auto output = topology.outputs[group.output_begin + i];
                out.y[row * topology.coordinates + output.coordinate] += output.scale * theta[k] * u * u * product;
                upstream += output.scale * dy[row * topology.coordinates + output.coordinate];
            }
            for (std::uint64_t i = 0; i < group.argument_count; ++i) {
                double exclusive = 1;
                for (std::uint64_t j = 0; j < group.argument_count; ++j)
                    if (i != j) exclusive *= x[row * topology.coordinates + topology.arguments[group.argument_begin + j]];
                out.dx[row * topology.coordinates + topology.arguments[group.argument_begin + i]] += upstream * theta[k] * u * u * exclusive;
            }
            const auto condition = upstream * theta[k] * product * 2 * u;
            out.dz[c] += condition; out.dz[c + 1] += condition * phi[k];
            out.theta[k] += upstream * u * u * product;
            out.phi[k] += condition * z[c + 1]; out.tau[k] -= condition;
        }
    return out;
}
void check_saved_backward_phase_reads(int device, const mc::Stream& stream) {
    const auto structure = std::make_shared<const bm::Structure>(bm::Structure::from_rows(3, {{0, 1, 2}}));
    const bm::DeviceStructure device_structure(structure, stream);
    const std::vector<float> x{2, 3, 4}, z{1, .5f, -2, .25f};
    const std::vector<float> theta{2, .5f}, phi{.5f, .25f}, tau{.25f, .5f}, dy{1, -.5f, 2};
    // Coordinate 2 participates only in the baseline, never any product.
    // The second module is inactive and must retain that saved branch decision.
    const std::vector<std::uint64_t> arguments{0, 1, 1};
    const std::vector<op::GatedProductGroup> groups{{0, 2, 0, 1, 0, 0, 1}, {2, 1, 1, 1, 1, 1, 2}};
    const std::vector<op::OutputIncidence> outputs{{2, 1}, {0, 1}};
    const op::GatedProductTopologyView host{groups.data(), groups.size(), arguments.data(), arguments.size(),
        outputs.data(), outputs.size(), 3, 2, 2};
    const auto argument_offsets = op::prepare_argument_value_offsets(*structure, host);
    const auto coordinate_offsets = op::prepare_coordinate_value_offsets(*structure, 3);
    std::vector<std::uint16_t> packed(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) packed[coordinate_offsets[i]] = half_bits(x[i]);
    auto device_x = upload(packed, stream), device_z = upload(z, stream), device_theta = upload(theta, stream),
        device_phi = upload(phi, stream), device_tau = upload(tau, stream), device_dy = upload(dy, stream),
        device_groups = upload(groups, stream), device_arguments = upload(arguments, stream),
        device_outputs = upload(outputs, stream), device_argument_offsets = upload(argument_offsets, stream),
        device_coordinate_offsets = upload(coordinate_offsets, stream);
    mc::DeviceBuffer y(x.size() * sizeof(float), device), dx(y.bytes(), device), dz(z.size() * sizeof(float), device),
        dtheta(theta.size() * sizeof(float), device), dphi(dtheta.bytes(), device), dtau(dtheta.bytes(), device),
        gates(groups.size() * sizeof(float), device), products(gates.bytes(), device), status(sizeof(std::uint32_t), device);
    const op::PackedGatedProductInput<std::uint16_t> input{
        {device_structure.view(), static_cast<const std::uint16_t*>(device_x.data())},
        {static_cast<const op::GatedProductGroup*>(device_groups.data()), groups.size(),
         static_cast<const std::uint64_t*>(device_arguments.data()), arguments.size(),
         static_cast<const op::OutputIncidence*>(device_outputs.data()), outputs.size(), 3, 2, 2},
        static_cast<const std::uint64_t*>(device_argument_offsets.data()),
        static_cast<const std::uint64_t*>(device_coordinate_offsets.data()), static_cast<const float*>(device_z.data()),
        static_cast<const float*>(device_theta.data()), static_cast<const float*>(device_phi.data()),
        static_cast<const float*>(device_tau.data()), 1, .25f, true};
    const op::PackedGatedProductSaved saved{static_cast<float*>(gates.data()), static_cast<float*>(products.data())};
    const op::PackedGatedProductGradients gradients{static_cast<float*>(dx.data()), static_cast<float*>(dz.data()),
        static_cast<float*>(dtheta.data()), static_cast<float*>(dphi.data()), static_cast<float*>(dtau.data())};
    op::PackedGatedProductCapacities capacity;
    capacity.values = device_x.bytes(); capacity.groups = device_groups.bytes();
    capacity.arguments = device_arguments.bytes(); capacity.outputs = device_outputs.bytes();
    capacity.argument_value_offsets = device_argument_offsets.bytes(); capacity.coordinate_value_offsets = device_coordinate_offsets.bytes();
    capacity.z = device_z.bytes(); capacity.theta = device_theta.bytes(); capacity.phi = device_phi.bytes(); capacity.tau = device_tau.bytes();
    capacity.output = y.bytes(); capacity.output_gradient = device_dy.bytes();
    capacity.gradient_x = dx.bytes(); capacity.gradient_z = dz.bytes(); capacity.gradient_theta = dtheta.bytes();
    capacity.gradient_phi = dphi.bytes(); capacity.gradient_tau = dtau.bytes();
    capacity.gate_arguments = gates.bytes(); capacity.products = products.bytes(); capacity.status = status.bytes();
    y.zero_async(stream); status.zero_async(stream);
    mc::check(op::launch_packed_forward_checked(input, capacity, static_cast<float*>(y.data()), saved,
        static_cast<std::uint32_t*>(status.data()), stream.get()), "phase-read forward");
    std::uint32_t bits = 0; status.download_async(&bits, sizeof(bits), stream); stream.synchronize();
    require(bits == 0, "phase-read forward failed");
    const auto expected = reference(host, 1, x, z, theta, phi, tau, dy, .25f);
    // Deliberate raw-launch diagnostic after completed forward. Production tapes
    // retain immutable inputs; poisoning only undeclared backward reads proves
    // this lower-level kernel respects the same phase-specific numerical inputs.
    auto poisoned_z = z, poisoned_tau = tau;
    poisoned_z[0] = poisoned_z[2] = std::numeric_limits<float>::quiet_NaN();
    std::fill(poisoned_tau.begin(), poisoned_tau.end(), std::numeric_limits<float>::quiet_NaN());
    packed[coordinate_offsets[2]] = half_bits(std::numeric_limits<float>::quiet_NaN());
    device_x.upload_async(packed.data(), device_x.bytes(), stream);
    device_z.upload_async(poisoned_z.data(), device_z.bytes(), stream);
    device_tau.upload_async(poisoned_tau.data(), device_tau.bytes(), stream);
    for (auto* buffer : {&dx, &dz, &dtheta, &dphi, &dtau, &status}) buffer->zero_async(stream);
    mc::check(op::launch_packed_backward_checked(input, capacity, static_cast<const float*>(device_dy.data()),
        gradients, saved, static_cast<std::uint32_t*>(status.data()), stream.get()), "saved phase-read backward");
    status.download_async(&bits, sizeof(bits), stream); stream.synchronize();
    require(bits == 0, "saved backward read z0, tau or baseline-only x");
    auto check = [&](const mc::DeviceBuffer& buffer, const std::vector<double>& values, const char* label) {
        std::vector<float> actual(values.size()); buffer.download_async(actual.data(), buffer.bytes(), stream);
        stream.synchronize(); compare(actual, values, label);
    };
    check(dx, expected.dx, "saved phase-read input VJP"); check(dz, expected.dz, "saved phase-read condition VJP");
    check(dtheta, expected.theta, "saved phase-read theta VJP"); check(dphi, expected.phi, "saved phase-read phi VJP");
    check(dtau, expected.tau, "saved phase-read tau VJP");
    status.zero_async(stream);
    mc::check(op::launch_packed_backward_checked(input, capacity, static_cast<const float*>(device_dy.data()),
        gradients, {}, static_cast<std::uint32_t*>(status.data()), stream.get()), "unsaved phase-read backward");
    status.download_async(&bits, sizeof(bits), stream); stream.synchronize();
    require(bits & op::packed_nonfinite, "unsaved backward failed to validate recomputed gate inputs");
    // Restore the forward inputs before any later evaluation.
    device_z.upload_async(z.data(), device_z.bytes(), stream); device_tau.upload_async(tau.data(), device_tau.bytes(), stream);
    packed[coordinate_offsets[2]] = half_bits(x[2]); device_x.upload_async(packed.data(), device_x.bytes(), stream);
    for (auto* buffer : {&dx, &dz, &dtheta, &dphi, &dtau, &status}) buffer->zero_async(stream);
    mc::check(op::launch_packed_backward_checked(input, capacity, static_cast<const float*>(device_dy.data()),
        gradients, {}, static_cast<std::uint32_t*>(status.data()), stream.get()), "restored unsaved backward");
    status.download_async(&bits, sizeof(bits), stream); stream.synchronize();
    require(bits == 0, "unsaved backward rejected restored gate inputs");
    check(dx, expected.dx, "unsaved input VJP"); check(dz, expected.dz, "unsaved condition VJP");
    check(dtheta, expected.theta, "unsaved theta VJP"); check(dphi, expected.phi, "unsaved phi VJP");
    check(dtau, expected.tau, "unsaved tau VJP");
}
}

int main(int argc, char** argv) {
    try {
        check_host_buffer_admission();
        if (argc > 1 && std::string(argv[1]) == "--host-only") {
            const auto structure = bm::Structure::from_rows(4, {{0, 1, 2, 3}, {0, 1, 2, 3}}, 2, 2,
                                                            {2, 0, 3, 1}, {1, 0});
            const auto offsets = op::prepare_coordinate_value_offsets(structure, 4);
            require(offsets.size() == 8 && offsets[0] != 0, "canonical map was inferred from position");
            const auto partition=bm::Structure::from_rows(1,{{0},{0}},1,2,{0},{101,100});
            require(op::prepare_coordinate_value_offsets(partition,1)[0]==1,"global row mapping failed");
            require(op::prepare_coordinate_value_offsets(partition,1,{101}).size()==1,"row subset rejected");
            rejects([&] { (void)op::prepare_coordinate_value_offsets(partition,1,{999}); });
            std::cout << "packed host preparation passed (no CUDA calls)\n"; return 0;
        }
        const int device = argc > 2 && std::string(argv[1]) == "--device" ? std::stoi(argv[2]) : 0;
        mc::DeviceGuard guard(device); mc::Stream stream(device);
        check_saved_backward_phase_reads(device, stream);
        static_assert(!std::is_move_constructible<bm::DeviceStructure>::value,
                      "device metadata owners must remain pinned");
        {
            auto mutable_structure=std::make_shared<bm::Structure>(bm::Structure::from_rows(2,{{0,1}},2,1,{1,0}));
            const bm::DeviceStructure replica(mutable_structure,stream);
            *mutable_structure=bm::Structure::from_rows(2,{{0,1}},2,1,{0,1});
            require(replica.host().canonical_features()[0]==1,"mutable host alias changed uploaded identity");
            auto moved=std::move(*mutable_structure); (void)moved;
            require(replica.host().canonical_features().size()==2,"host move invalidated uploaded metadata");
        }
        const std::uint64_t batch = 2, coordinates = 4;
        const auto structure = std::make_shared<const bm::Structure>(bm::Structure::from_rows(
            coordinates, {{0, 1, 2, 3}, {0, 1, 2, 3}}, 2, 2, {2, 0, 3, 1}, {1, 0}));
        const auto device_structure = std::make_shared<const bm::DeviceStructure>(structure, stream);
        std::vector<float> x{0, 2, 1, 4, 2, -1, 1, 0.5f};
        const std::vector<float> z{1, 0.5f, -1, 0, 1, 0.5f, 1, 0};
        const std::vector<float> theta{2, 0.5f}, phi{0.5f, -0.25f}, tau{0.25f, 0};
        const std::vector<float> dy{1, -0.5f, 2, 0.25f, -1, 0.5f, 1, -2};
        std::vector<std::uint64_t> arguments{0, 1, 1, 0, 0, 3};
        arguments.insert(arguments.end(), 32, 2);
        const std::vector<op::GatedProductGroup> groups{{0,3,0,2,0,0,1}, {3,3,2,1,0,1,2}, {6,32,3,1,1,0,3}};
        const std::vector<op::OutputIncidence> outputs{{2,1}, {3,0.5f}, {2,-0.25f}, {1,0.125f}};
        const op::GatedProductTopologyView host{groups.data(),groups.size(),arguments.data(),arguments.size(),
            outputs.data(),outputs.size(),coordinates,2,2};
        const auto argument_offsets = op::prepare_argument_value_offsets(*structure, host);
        const auto coordinate_offsets = op::prepare_coordinate_value_offsets(*structure, coordinates);
        auto device_groups = upload(groups,stream), device_arguments = upload(arguments,stream),
            device_outputs = upload(outputs,stream), device_argument_offsets = upload(argument_offsets,stream),
            device_coordinate_offsets = upload(coordinate_offsets,stream), device_z = upload(z,stream),
            device_theta = upload(theta,stream), device_phi = upload(phi,stream), device_tau = upload(tau,stream),
            device_dy = upload(dy,stream);
        mc::DeviceBuffer y(x.size()*sizeof(float),device), dx(y.bytes(),device), dz(z.size()*sizeof(float),device),
            dtheta(theta.size()*sizeof(float),device), dphi(dtheta.bytes(),device), dtau(dtheta.bytes(),device),
            gates(batch*groups.size()*sizeof(float),device), products(gates.bytes(),device), status(sizeof(std::uint32_t),device);
        bm::NativeTensor<std::uint16_t> tensor(device_structure);
        op::PackedGatedProductCapacities capacity;
        capacity.values=structure->view().stored_values*sizeof(std::uint16_t);
        capacity.groups=device_groups.bytes(); capacity.arguments=device_arguments.bytes(); capacity.outputs=device_outputs.bytes();
        capacity.argument_value_offsets=device_argument_offsets.bytes(); capacity.coordinate_value_offsets=device_coordinate_offsets.bytes();
        capacity.z=device_z.bytes(); capacity.theta=device_theta.bytes(); capacity.phi=device_phi.bytes(); capacity.tau=device_tau.bytes();
        capacity.output=y.bytes(); capacity.output_gradient=device_dy.bytes();
        capacity.gradient_x=dx.bytes(); capacity.gradient_z=dz.bytes(); capacity.gradient_theta=dtheta.bytes();
        capacity.gradient_phi=dphi.bytes(); capacity.gradient_tau=dtau.bytes();
        capacity.gate_arguments=gates.bytes(); capacity.products=products.bytes(); capacity.status=status.bytes();
        auto publish = [&](std::uint64_t generation) {
            std::vector<std::uint16_t> packed(structure->view().stored_values);
            for (std::uint64_t row=0; row<batch; ++row)
                for (std::uint64_t coordinate=0; coordinate<coordinates; ++coordinate)
                    packed[coordinate_offsets[row*coordinates+coordinate]] = half_bits(x[row*coordinates+coordinate]);
            auto edit=tensor.edit();
            mc::check(cudaMemcpyAsync(edit.buffer().data,packed.data(),packed.size()*sizeof(std::uint16_t),cudaMemcpyHostToDevice,stream.get()),"upload packed values");
            stream.synchronize(); edit.publish(generation);
        };
        publish(1);
        for (unsigned trial=0; trial<2; ++trial) {
            if (trial) { x[3]=std::numeric_limits<float>::quiet_NaN(); publish(2); }
            const float beta=0.25f;
            auto retained=tensor.retain(trial+1);
            rejects([&] { auto edit=tensor.edit(); });
            rejects([&] { auto wrong=tensor.retain(999); });
            const op::GatedProductTopologyView topology{static_cast<const op::GatedProductGroup*>(device_groups.data()),groups.size(),
                static_cast<const std::uint64_t*>(device_arguments.data()),arguments.size(),
                static_cast<const op::OutputIncidence*>(device_outputs.data()),outputs.size(),coordinates,2,2};
            const op::PackedGatedProductInput<std::uint16_t> input{retained.view(),topology,
                static_cast<const std::uint64_t*>(device_argument_offsets.data()),
                static_cast<const std::uint64_t*>(device_coordinate_offsets.data()),static_cast<const float*>(device_z.data()),
                static_cast<const float*>(device_theta.data()),static_cast<const float*>(device_phi.data()),
                static_cast<const float*>(device_tau.data()),batch,beta,!trial};
            for (auto* buffer : {&y,&dx,&dz,&dtheta,&dphi,&dtau,&status}) buffer->zero_async(stream);
            const op::PackedGatedProductSaved saved{static_cast<float*>(gates.data()),static_cast<float*>(products.data())};
            auto short_output=capacity; --short_output.output;
            require(op::launch_packed_forward_checked(input,short_output,static_cast<float*>(y.data()),saved,
                static_cast<std::uint32_t*>(status.data()),stream.get())==cudaErrorInvalidValue,"short output accepted");
            auto short_map=capacity; --short_map.argument_value_offsets;
            require(op::launch_packed_forward_checked(input,short_map,static_cast<float*>(y.data()),saved,
                static_cast<std::uint32_t*>(status.data()),stream.get())==cudaErrorInvalidValue,"short map accepted");
            auto short_saved=capacity; --short_saved.products;
            require(op::launch_packed_forward_checked(input,short_saved,static_cast<float*>(y.data()),saved,
                static_cast<std::uint32_t*>(status.data()),stream.get())==cudaErrorInvalidValue,"short saved buffer accepted");
            require(op::launch_packed_forward_checked(input,capacity,reinterpret_cast<float*>(const_cast<std::uint16_t*>(input.x.values)),saved,
                static_cast<std::uint32_t*>(status.data()),stream.get())==cudaErrorInvalidValue,"output/input alias accepted");
            mc::check(op::launch_packed_forward_checked(input,capacity,static_cast<float*>(y.data()),saved,static_cast<std::uint32_t*>(status.data()),stream.get()),"packed forward");
            const op::PackedGatedProductGradients gradients{static_cast<float*>(dx.data()),static_cast<float*>(dz.data()),
                static_cast<float*>(dtheta.data()),static_cast<float*>(dphi.data()),static_cast<float*>(dtau.data())};
            auto short_grad=capacity; --short_grad.gradient_z;
            require(op::launch_packed_backward_checked(input,short_grad,static_cast<const float*>(device_dy.data()),gradients,saved,
                static_cast<std::uint32_t*>(status.data()),stream.get())==cudaErrorInvalidValue,"short VJP buffer accepted");
            auto alias=gradients; alias.theta=alias.x;
            require(op::launch_packed_backward_checked(input,capacity,static_cast<const float*>(device_dy.data()),alias,saved,
                static_cast<std::uint32_t*>(status.data()),stream.get())==cudaErrorInvalidValue,"gradient alias accepted");
            mc::check(op::launch_packed_backward_checked(input,capacity,static_cast<const float*>(device_dy.data()),gradients,saved,
                static_cast<std::uint32_t*>(status.data()),stream.get()),"packed backward");
            std::uint32_t bits=0; status.download_async(&bits,sizeof(bits),stream); stream.synchronize();
            require(bits==0,"packed numerical status failure");
            const auto expected=reference(host,batch,x,z,theta,phi,tau,dy,trial?0.0f:beta);
            auto check_buffer=[&](const mc::DeviceBuffer& buffer,const std::vector<double>& expected,const char* label) {
                std::vector<float> actual(expected.size()); buffer.download_async(actual.data(),actual.size()*sizeof(float),stream);
                stream.synchronize(); compare(actual,expected,label);
            };
            check_buffer(y,expected.y,"output"); check_buffer(dx,expected.dx,"input VJP");
            check_buffer(dz,expected.dz,"condition VJP"); check_buffer(dtheta,expected.theta,"theta VJP");
            check_buffer(dphi,expected.phi,"phi VJP"); check_buffer(dtau,expected.tau,"tau VJP");
            if (trial) {
                auto owned_zero=input; owned_zero.beta=0; owned_zero.include_baseline=true;
                status.zero_async(stream);
                mc::check(op::launch_packed_forward_checked(owned_zero,capacity,static_cast<float*>(y.data()),saved,
                    static_cast<std::uint32_t*>(status.data()),stream.get()),"owned zero baseline");
                status.download_async(&bits,sizeof(bits),stream); stream.synchronize();
                require(bits & op::packed_nonfinite,"owned beta zero incorrectly skipped NaN input");
            }
        }
        x[2]=65504; x[6]=65504; publish(3);
        {
            auto retained=tensor.retain(3);
            op::PackedGatedProductInput<std::uint16_t> overflow{retained.view(),
                {static_cast<const op::GatedProductGroup*>(device_groups.data()),groups.size(),
                 static_cast<const std::uint64_t*>(device_arguments.data()),arguments.size(),
                 static_cast<const op::OutputIncidence*>(device_outputs.data()),outputs.size(),coordinates,2,2},
                static_cast<const std::uint64_t*>(device_argument_offsets.data()),
                static_cast<const std::uint64_t*>(device_coordinate_offsets.data()),static_cast<const float*>(device_z.data()),
                static_cast<const float*>(device_theta.data()),static_cast<const float*>(device_phi.data()),
                static_cast<const float*>(device_tau.data()),batch,0.25f,false};
            status.zero_async(stream); y.zero_async(stream);
            const op::PackedGatedProductSaved saved{static_cast<float*>(gates.data()),static_cast<float*>(products.data())};
            mc::check(op::launch_packed_forward_checked(overflow,capacity,static_cast<float*>(y.data()),saved,
                static_cast<std::uint32_t*>(status.data()),stream.get()),"overflow status");
            std::uint32_t bits=0; status.download_async(&bits,sizeof(bits),stream); stream.synchronize();
            require(bits & op::packed_nonfinite,"nonfinite computation was published without status");
        }
        std::optional<bm::RetainedRead<std::uint16_t>> after_owner;
        {
            bm::NativeTensor<std::uint16_t> temporary(device_structure);
            auto edit=temporary.edit();
            mc::check(cudaMemsetAsync(edit.buffer().data,0,edit.buffer().bytes,stream.get()),"retained owner zero");
            stream.synchronize(); edit.publish(1); after_owner.emplace(temporary.retain(1));
        }
        std::uint16_t first=1;
        mc::check(cudaMemcpyAsync(&first,after_owner->view().values,sizeof(first),cudaMemcpyDeviceToHost,stream.get()),"read after owner destruction");
        stream.synchronize(); require(first==0,"retained allocation did not survive owner destruction");
        after_owner.reset();
        { auto edit=tensor.edit(); } // An abandoned edit cannot publish unknown contents.
        rejects([&] { auto read=tensor.retain(); });
        std::cout << "packed owner, canonical maps, direct FP16 forward/VJP and inactive no-read passed\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
