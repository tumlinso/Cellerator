#include <Cellerator/bindings/mechanism_handle.hh>

#include <cuda_runtime_api.h>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace ix = cellerator::compute::operation::indexed;

namespace {
void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

cellerator::execution::persistent_axis_identity axis(std::uint64_t base) {
    cellerator::execution::persistent_axis_identity value{};
    value.header.schema_version = cellerator::execution::biological_abi_version;
    value.header.kind = cellerator::execution::serialized_record_kind::persistent_axis_identity;
    value.header.byte_count = sizeof(value);
    value.domain = {base, 1}; value.order = {base + 1, 1};
    value.geometry = {base + 2, 1}; value.partition = {base + 3, 1};
    return value;
}
}

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        std::cerr << "CUDA indexed mechanism test requires a CUDA device\n";
        return 1;
    }

    ix::mechanism_declaration declaration;
    declaration.input = {axis(10), 2};
    declaration.output = {axis(20), 1};
    declaration.coefficients = {axis(30), 1};
    declaration.coefficient_ids = {{101, 1}};
    declaration.max_batch = 2;
    declaration.max_live_forwards = 2;
    ix::product_mechanism mechanism;
    mechanism.instance = {201, 1};
    mechanism.coefficient = 0;
    mechanism.arguments = {{0, {301, 1}, 0, 0}, {1, {302, 1}, 0, 1}};
    ix::output_index output{};
    output.slot = 0; output.role = {401, 1}; output.index = 0;
    output.assembly_owner = {501, 1};
    output.effect.update = cellerator::execution::output_update_kind::accumulate;
    output.effect.requires_initialized_destination = true;
    mechanism.outputs.push_back({output, 1.0f});
    declaration.mechanisms.push_back(mechanism);

    const float initial[] = {0.5f};
    cellerator::bindings::MechanismHandle handle(declaration, std::span(initial), 0);
    const std::uint64_t input_values[] = {0};
    (void)input_values;
    float *input = nullptr, *result = nullptr, *dy = nullptr, *dx = nullptr, *dk = nullptr;
    check(cudaMalloc(reinterpret_cast<void**>(&input), 4 * sizeof(float)));
    check(cudaMalloc(reinterpret_cast<void**>(&result), 2 * sizeof(float)));
    check(cudaMalloc(reinterpret_cast<void**>(&dy), 2 * sizeof(float)));
    check(cudaMalloc(reinterpret_cast<void**>(&dx), 4 * sizeof(float)));
    check(cudaMalloc(reinterpret_cast<void**>(&dk), sizeof(float)));
    const float host_input[] = {2.0f, 3.0f, 4.0f, 5.0f};
    const float host_dy[] = {1.0f, 1.0f};
    check(cudaMemcpy(input, host_input, sizeof(host_input), cudaMemcpyHostToDevice));
    check(cudaMemcpy(dy, host_dy, sizeof(host_dy), cudaMemcpyHostToDevice));
    auto tape = handle.program()->forward(input, false, 2, declaration.input.identity, result, nullptr);
    bool rejected_live_tape_write = false;
    try { handle.preflight_write(); } catch (const std::exception&) { rejected_live_tape_write = true; }
    require(rejected_live_tape_write, "owner accepted a write while a tape was live");
    const float changed_input[] = {-10.0f, -11.0f, -12.0f, -13.0f};
    check(cudaMemcpy(input, changed_input, sizeof(changed_input), cudaMemcpyHostToDevice));
    handle.program()->backward(*tape, dy, dx, dk, nullptr);
    check(cudaDeviceSynchronize());
    float host_result[2]{}, host_dx[4]{}, host_dk = 0;
    check(cudaMemcpy(host_result, result, sizeof(host_result), cudaMemcpyDeviceToHost));
    check(cudaMemcpy(host_dx, dx, sizeof(host_dx), cudaMemcpyDeviceToHost));
    check(cudaMemcpy(&host_dk, dk, sizeof(float), cudaMemcpyDeviceToHost));
    require(std::abs(host_result[0] - 3.0f) < 1e-6f, "native forward output mismatch");
    require(std::abs(host_result[1] - 10.0f) < 1e-6f, "native forward output mismatch");
    require(std::abs(host_dx[0] - 1.5f) < 1e-6f && std::abs(host_dx[1] - 1.0f) < 1e-6f,
            "native saved-primal input gradient mismatch");
    require(std::abs(host_dx[2] - 2.5f) < 1e-6f && std::abs(host_dx[3] - 2.0f) < 1e-6f,
            "native saved-primal input gradient mismatch");
    require(std::abs(host_dk - 26.0f) < 1e-6f, "native coefficient gradient mismatch");
    bool rejected_tape_replay = false;
    try { handle.program()->backward(*tape, dy, dx, dk, nullptr); }
    catch (const std::exception&) { rejected_tape_replay = true; }
    require(rejected_tape_replay, "native mechanism accepted tape replay");

    const auto generation = handle.owner()->generation();
    handle.begin_write(nullptr);
    const float updated[] = {0.25f};
    check(cudaMemcpy(handle.owner()->data(), updated, sizeof(updated), cudaMemcpyHostToDevice));
    handle.publish_write(nullptr);
    require(handle.owner()->generation() == generation + 1, "owner generation did not publish");
    auto snapshot = handle.snapshot(nullptr);
    require(snapshot.size() == 1 && std::abs(snapshot[0] - 0.25f) < 1e-6f,
            "native owner snapshot mismatch after update");
    handle.restore(std::span(initial), nullptr);
    require(handle.owner()->generation() == generation + 2, "owner generation did not restore");
    require(std::abs(handle.snapshot(nullptr)[0] - 0.5f) < 1e-6f,
            "native owner snapshot mismatch after restore");

    cudaFree(input); cudaFree(result); cudaFree(dy); cudaFree(dx); cudaFree(dk);
    std::cout << "native mechanism owner/program/tape lifecycle passed\n";
}
