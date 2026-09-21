#include <Cellerator/compute/operation/native_foundation_contract.hh>
#include <cuda_runtime_api.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace ex = cellerator::execution;
namespace nf = cellerator::compute::operation::nf1;
namespace pg = cellerator::execution::program;

void require(bool value) { if (!value) std::abort(); }
void ok(cudaError_t value) { if (value != cudaSuccess) std::abort(); }

struct external_state { std::uint64_t count{}; float scale{}, bias{}; };
struct external_payload { const float *input{}; float *output{}; std::uint64_t count{}; };

__global__ void external_affine_kernel(const float *input, float *output, std::uint64_t count,
                                       float scale, float bias) {
    const auto index = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) output[index] = input[index] * scale + bias;
}

pg::program_status external_forward(const void *prepared_state,
                                    const pg::launch_binding_v2 &binding,
                                    void *caller_stream) noexcept {
    const auto *state = static_cast<const external_state *>(prepared_state);
    const auto *payload = static_cast<const external_payload *>(binding.input);
    if (!state || !payload || binding.output || binding.values || binding.workspace
        || binding.workspace_bytes || !caller_stream || !payload->input || !payload->output
        || !state->count || payload->count != state->count) return pg::program_status::invalid_argument;
    external_affine_kernel<<<static_cast<unsigned>((state->count + 255u) / 256u), 256, 0,
                            static_cast<cudaStream_t>(caller_stream)>>>(
        payload->input, payload->output, state->count, state->scale, state->bias);
    return cudaGetLastError() == cudaSuccess ? pg::program_status::success
                                             : pg::program_status::launch_failed;
}

int main() {
    ok(cudaSetDevice(0));
    cudaDeviceProp device{};
    ok(cudaGetDeviceProperties(&device, 0));
    require(device.major == 7 && device.minor == 0);

    constexpr std::size_t count = 33;
    constexpr float scale = 1.5f, bias = .25f;
    cudaStream_t stream{};
    ok(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    std::vector<float> host_input(count), host_output(count);
    for (std::size_t index = 0; index < count; ++index) host_input[index] = .2f + .01f * float(index);
    float *input{}, *output{};
    ok(cudaMalloc(&input, count * sizeof(float)));
    ok(cudaMalloc(&output, count * sizeof(float)));
    ok(cudaMemcpyAsync(input, host_input.data(), count * sizeof(float), cudaMemcpyHostToDevice, stream));

    std::array<nf::operand_signature, 1> inputs{{{{10, 1}, {}, count, ex::numeric_type::f32}}};
    nf::output_signature output_signature{};
    output_signature.operand = {{12, 1}, {}, count, ex::numeric_type::f32};
    output_signature.assembly_owner = {13, 1};
    nf::operation_contract contract{};
    contract.definition = {501, 1};
    contract.arguments = {inputs.data(), inputs.size()};
    contract.outputs = {&output_signature, 1};
    contract.numeric = {ex::numeric_type::f32, ex::numeric_type::f32, ex::numeric_type::f32,
                        ex::numeric_type::f32, ex::numeric_type::f32, ex::numeric_type::f32};
    contract.capabilities = nf::forward;

    const external_state state{count, scale, bias};
    nf::compiled_block custom{};
    custom.contract = contract;
    custom.effects = {true, true, true, true};
    custom.forward_launch = external_forward;
    require(nf::validate_compiled_block(custom) == nf::status::success);

    pg::prepared_stage_v2 stage{};
    require(nf::bind_compiled_stage(custom, nf::forward, &state, 1, 1, 0, stage) == nf::status::success);
    pg::prepared_program_v2 program{2, 0, &stage, 1, nullptr, 0};
    const external_payload payload{input, output, count};
    pg::launch_binding_v2 binding{};
    binding.input = &payload;
    require(pg::execute_prepared_program_v2(program, &binding, 1, stream) == pg::program_status::success);
    ok(cudaMemcpyAsync(host_output.data(), output, count * sizeof(float), cudaMemcpyDeviceToHost, stream));
    ok(cudaStreamSynchronize(stream));
    for (std::size_t index = 0; index < count; ++index)
        require(std::abs(host_output[index] - (host_input[index] * scale + bias)) < 2e-6f);

    pg::prepared_stage_v2 rejected{};
    require(nf::bind_compiled_stage(custom, nf::jvp, &state, 2, 2, 0, rejected)
            == nf::status::unsupported_derivative);
    std::cout << "external consumer-defined CUDA forward block executed; derivative binding refused before launch\n";
    ok(cudaFree(output));
    ok(cudaFree(input));
    ok(cudaStreamDestroy(stream));
}
