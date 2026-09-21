#include <Cellerator/compute/operation/indexed_mechanism/evaluators.hh>

#include <cuda_runtime_api.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace ix = cellerator::compute::operation::indexed;
void check(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
void cuda_check(cudaError_t status, const char* message) { check(status == cudaSuccess, message); }
int main() try {
    ix::registered_block block{{400, 1}, ix::evaluator_opcode::first_minus_product_tail, 3};
    const std::array<float, 5> input{10.0f, 2.0f, 3.0f, 2.0f, 1.0f};
    const std::array<float, 5> invalid_input{std::numeric_limits<float>::quiet_NaN(), 2.0f, 3.0f, 2.0f, 1.0f};
    float* device_input = nullptr; float* device_invalid = nullptr; float* device_output = nullptr;
    cuda_check(cudaMalloc(&device_input, sizeof(input)), "allocate input");
    cuda_check(cudaMalloc(&device_invalid, sizeof(invalid_input)), "allocate invalid input");
    cuda_check(cudaMalloc(&device_output, sizeof(float)), "allocate output");
    cuda_check(cudaMemcpy(device_input, input.data(), sizeof(input), cudaMemcpyHostToDevice), "copy input");
    cuda_check(cudaMemcpy(device_invalid, invalid_input.data(), sizeof(invalid_input), cudaMemcpyHostToDevice), "copy invalid input");
    const float sentinel = 17.0f;
    cuda_check(cudaMemcpy(device_output, &sentinel, sizeof(sentinel), cudaMemcpyHostToDevice), "set predicate sentinel");
    check(ix::evaluate_cuda_f32(block, false, device_invalid, invalid_input.size(), device_output) == ix::evaluation_status::success, "device predicate launch");
    cuda_check(cudaDeviceSynchronize(), "synchronize excluded predicate");
    float output = 0.0f;
    cuda_check(cudaMemcpy(&output, device_output, sizeof(output), cudaMemcpyDeviceToHost), "copy excluded output");
    check(output == sentinel, "false device predicate neither evaluates nonfinite operand nor writes destination");
    check(ix::evaluate_cuda_f32(block, true, device_invalid, invalid_input.size(), device_output) == ix::evaluation_status::success, "active nonfinite launch");
    cuda_check(cudaDeviceSynchronize(), "synchronize active nonfinite");
    cuda_check(cudaMemcpy(&output, device_output, sizeof(output), cudaMemcpyDeviceToHost), "copy active nonfinite output");
    check(std::isnan(output), "active nonfinite propagates rather than being sanitized");
    cuda_check(cudaMemcpy(device_output, &sentinel, sizeof(sentinel), cudaMemcpyHostToDevice), "reset predicate sentinel");
    check(ix::evaluate_cuda_f32(block, false, nullptr, input.size(), device_output) == ix::evaluation_status::success, "null input predicate control");
    cuda_check(cudaDeviceSynchronize(), "synchronize null predicate control");
    cuda_check(cudaMemcpy(&output, device_output, sizeof(output), cudaMemcpyDeviceToHost), "copy null predicate control");
    check(output == sentinel, "false device predicate short-circuits null input");
    check(ix::evaluate_cuda_f32(block, true, device_input, input.size(), device_output) == ix::evaluation_status::success, "device FP32 launch");
    cuda_check(cudaDeviceSynchronize(), "synchronize FP32");
    cuda_check(cudaMemcpy(&output, device_output, sizeof(output), cudaMemcpyDeviceToHost), "copy FP32 output");
    check(output == -2.0f, "runtime-width CUDA triad result");
    ix::prepared_evaluator_stage stage{block, input.size(), 9};
    ix::evaluator_stage_values stage_values{ix::evaluator_stage_values::magic, 9, true};
    auto prepared = ix::make_prepared_stage(1, 2, stage, 0);
    cellerator::execution::program::launch_binding_v2 binding{device_input, device_output, &stage_values};
    check(prepared.launch(prepared.prepared_state, binding, nullptr)
              == cellerator::execution::program::program_status::success,
          "CE program-v2 stage callback accepts generation-matched resident binding");
    cuda_check(cudaDeviceSynchronize(), "synchronize program-v2 stage");
    cuda_check(cudaMemcpy(&output, device_output, sizeof(output), cudaMemcpyDeviceToHost), "copy program-v2 output");
    check(output == -2.0f, "program-v2 stage result");
    stage_values.value_generation = 10;
    check(prepared.launch(prepared.prepared_state, binding, nullptr)
              == cellerator::execution::program::program_status::invalid_argument,
          "stale value generation rejected before device launch");

    const std::array<std::uint16_t, 3> half_input{0x4900u, 0x4000u, 0x4600u};
    std::uint16_t* device_half_input = nullptr; std::uint16_t* device_half_output = nullptr;
    cuda_check(cudaMalloc(&device_half_input, sizeof(half_input)), "allocate half input");
    cuda_check(cudaMalloc(&device_half_output, sizeof(std::uint16_t)), "allocate half output");
    cuda_check(cudaMemcpy(device_half_input, half_input.data(), sizeof(half_input), cudaMemcpyHostToDevice), "copy half input");
    check(ix::evaluate_cuda_f16(block, true, device_half_input, half_input.size(), device_half_output) == ix::evaluation_status::success, "device FP16 launch");
    cuda_check(cudaDeviceSynchronize(), "synchronize FP16");
    std::uint16_t half_output = 0;
    cuda_check(cudaMemcpy(&half_output, device_half_output, sizeof(half_output), cudaMemcpyDeviceToHost), "copy half output");
    check(half_output == 0xc000u, "device FP16 RNE result");
    cudaFree(device_half_output); cudaFree(device_half_input); cudaFree(device_output); cudaFree(device_invalid); cudaFree(device_input);
    std::cout << "H03 CUDA FP32/FP16 runtime-width route and predicate exclusion passed\n";
} catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
