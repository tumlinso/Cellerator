#include <Cellerator/modeling/cuda/factorized_interaction.hh>

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>

namespace {

int device_query_count = 0;
int memset_count = 0;

[[noreturn]] void fail(const std::string& message) {
    std::fprintf(stderr, "host contract failure: %s\n", message.c_str());
    std::exit(1);
}

void require(bool condition, const std::string& message) {
    if (!condition) fail(message);
}

void* address(std::uintptr_t value) {
    return reinterpret_cast<void*>(value);
}

using namespace cellerator::modeling::cuda;
namespace cuda_memory = cellerator::memory::cuda;

cuda_memory::ConstDeviceView const_view(std::uintptr_t pointer,
                                        std::size_t bytes) {
    return {reinterpret_cast<const void*>(pointer), bytes, 0};
}

cuda_memory::DeviceView device_view(std::uintptr_t pointer,
                                    std::size_t bytes) {
    return {address(pointer), bytes, 0};
}

struct Fixture {
    cellerator::layout::FactorizedInteractionView plan{};
    FactorizedInteractionProblem problem{1, 1};
    FactorizedInteractionPrecision precision{
        cuda_memory::DType::float32,
        cuda_memory::DType::float32,
        cuda_memory::DType::float32};
    FactorizedInteractionValues values{};
    cuda_memory::DeviceView output{};
    cuda_memory::DeviceView tape{};
    cuda_memory::DeviceView scratch{};
    cuda_memory::ConstDeviceView dy{};
    FactorizedInteractionGradients gradients{};

    Fixture() {
        using namespace cellerator::layout;
        plan.schema = factorized_interaction_schema;
        plan.mechanism_count = 1;
        plan.term_count = 1;
        plan.extents.inputs = 2;
        plan.extents.contexts = 1;
        plan.extents.outputs = 1;
        plan.extents.alpha_rows = 1;
        plan.extents.biases = 1;
        plan.extents.gammas = 1;
        plan.mechanisms = reinterpret_cast<const FactorizedInteractionRecord*>(0x10000);
        plan.terms = reinterpret_cast<const FactorizedInteractionTermRecord*>(0x11000);

        values = {
            const_view(0x2000, sizeof(float) * 2), // x
            const_view(0x3000, sizeof(float)),     // z
            const_view(0x4000, sizeof(float)),     // alpha
            const_view(0x5000, sizeof(float)),     // bias
            const_view(0x6000, sizeof(float))};    // gamma
        output = device_view(0x7000, sizeof(float));
        tape = device_view(0x8000, sizeof(float));
        scratch = device_view(0x9000, sizeof(float));
        dy = const_view(0xA000, sizeof(float));
        gradients = {
            device_view(0xB000, sizeof(float) * 2), // dx
            device_view(0xC000, sizeof(float)),     // dz
            device_view(0xD000, sizeof(float)),     // dalpha
            device_view(0xE000, sizeof(float)),     // dbias
            device_view(0xF000, sizeof(float))};    // dgamma
    }
};

void expect_early_invalid(const std::string& label, cudaError_t status) {
    require(status == cudaErrorInvalidValue,
            label + " did not return cudaErrorInvalidValue");
    require(device_query_count == 0,
            label + " queried a CUDA device before rejecting the view");
    require(memset_count == 0,
            label + " issued a CUDA memset before rejecting the view");
}

void test_workspace_contract() {
    using namespace cellerator::layout;
    FactorizedInteractionView plan{};
    plan.schema = factorized_interaction_schema;
    plan.mechanism_count = 2;
    plan.extents.inputs = 4;
    plan.extents.contexts = 3;
    plan.extents.outputs = 5;
    plan.extents.alpha_rows = 2;
    plan.extents.biases = 2;
    plan.extents.gammas = 3;
    FactorizedInteractionWorkspace workspace{};
    auto status = factorized_interaction_direct_workspace(plan, {7, 9}, &workspace);
    require(status == cudaSuccess, "valid direct workspace query failed");
    require(workspace.tape_bytes == 2u * 2u * 7u * sizeof(float),
            "direct tape extent changed");
    require(workspace.forward_bytes == 5u * 7u * 9u * sizeof(float),
            "direct forward scratch extent changed");
    require(workspace.backward_bytes == 2u * 7u * sizeof(float),
            "direct backward scratch extent changed");

    plan.max_output_mechanisms = 65;
    FactorizedInteractionBackend backend{};
    status = factorized_interaction_backend(
        plan, {7, 9}, FactorizedInteractionRoute::direct, &backend);
    require(status == cudaSuccess && backend == FactorizedInteractionBackend::stable,
            "large-fanin plan did not select the stable backend");
    status = factorized_interaction_direct_workspace(plan, {7, 9}, &workspace);
    require(status == cudaSuccess && workspace.forward_bytes == 0,
            "stable direct forward unexpectedly requires packet scratch");
    require(workspace.backward_bytes == 2u * 7u * sizeof(float),
            "stable direct backward scratch extent changed");
    plan.max_output_mechanisms = 0;

    status = factorized_interaction_direct_workspace(
        plan, {std::numeric_limits<std::uint64_t>::max(), 9}, &workspace);
    require(status == cudaErrorInvalidValue, "overflowing workspace was accepted");
    require(factorized_interaction_direct_workspace(plan, {1, 1}, nullptr) ==
                cudaErrorInvalidValue,
            "null workspace result was accepted");

    plan.extents.biases = std::numeric_limits<std::uint64_t>::max() / sizeof(float) + 1;
    require(factorized_interaction_direct_workspace(plan, {0, 0}, &workspace) ==
                cudaErrorInvalidValue,
            "overflowing scalar-bank extent was accepted");
}

void test_overlap_guards() {
    Fixture f;
    const auto shared = reinterpret_cast<const void*>(0x2000);
    f.values.x = {shared, sizeof(float) * 2, 0};
    f.output = {const_cast<void*>(shared), sizeof(float), 0};
    expect_early_invalid("forward primal/output overlap",
        factorized_interaction_direct_forward(
            f.plan, f.problem, f.precision, f.values, f.output, f.tape,
            f.scratch, nullptr));

    f = Fixture{};
    f.gradients.x = {const_cast<void*>(reinterpret_cast<const void*>(0x2000)),
                     sizeof(float) * 2, 0};
    expect_early_invalid("backward primal/gradient overlap",
        factorized_interaction_direct_backward(
            f.plan, f.problem, f.precision, f.values, f.dy, f.tape,
            f.gradients, f.scratch,
            FactorizedInteractionGradientMode::overwrite, nullptr));

    f = Fixture{};
    f.scratch.data = address(0x2000);
    expect_early_invalid("backward primal/scratch overlap",
        factorized_interaction_direct_backward(
            f.plan, f.problem, f.precision, f.values, f.dy, f.tape,
            f.gradients, f.scratch,
            FactorizedInteractionGradientMode::overwrite, nullptr));
}

void test_forward_alignment_guards() {
    // One failure for each independently typed primal and each writable/tape
    // payload. All fake allocations are otherwise sufficiently sized and
    // disjoint, so only the selected alignment rule can reject the call.
    for (int field = 0; field != 8; ++field) {
        Fixture f;
        const auto misaligned = std::uintptr_t{0x20001} +
                                static_cast<std::uintptr_t>(field) * 0x1000;
        switch (field) {
        case 0: f.values.x.data = reinterpret_cast<const void*>(misaligned); break;
        case 1: f.values.z.data = reinterpret_cast<const void*>(misaligned); break;
        case 2: f.values.alpha.data = reinterpret_cast<const void*>(misaligned); break;
        case 3: f.values.bias.data = reinterpret_cast<const void*>(misaligned); break;
        case 4: f.values.gamma.data = reinterpret_cast<const void*>(misaligned); break;
        case 5: f.output.data = address(misaligned); break;
        case 6: f.tape.data = address(misaligned); break;
        case 7: f.scratch.data = address(misaligned); break;
        }
        expect_early_invalid("forward misaligned payload " + std::to_string(field),
            factorized_interaction_direct_forward(
                f.plan, f.problem, f.precision, f.values, f.output, f.tape,
                f.scratch, nullptr));
    }
}

void test_backward_alignment_guards() {
    // The contract applies to all five primals, dy, all five VJPs, tape, and
    // scratch. The fixture's one-mechanism, one-cell backward workspace is
    // one FP32 value, so this nonempty view is sufficient for guard checks.
    for (int field = 0; field != 13; ++field) {
        Fixture f;
        f.scratch.bytes = sizeof(float);
        const auto misaligned = std::uintptr_t{0x20001} +
                                static_cast<std::uintptr_t>(field) * 0x1000;
        switch (field) {
        case 0: f.values.x.data = reinterpret_cast<const void*>(misaligned); break;
        case 1: f.values.z.data = reinterpret_cast<const void*>(misaligned); break;
        case 2: f.values.alpha.data = reinterpret_cast<const void*>(misaligned); break;
        case 3: f.values.bias.data = reinterpret_cast<const void*>(misaligned); break;
        case 4: f.values.gamma.data = reinterpret_cast<const void*>(misaligned); break;
        case 5: f.dy.data = reinterpret_cast<const void*>(misaligned); break;
        case 6: f.gradients.x.data = address(misaligned); break;
        case 7: f.gradients.z.data = address(misaligned); break;
        case 8: f.gradients.alpha.data = address(misaligned); break;
        case 9: f.gradients.bias.data = address(misaligned); break;
        case 10: f.gradients.gamma.data = address(misaligned); break;
        case 11: f.tape.data = address(misaligned); break;
        case 12: f.scratch.data = address(misaligned); break;
        }
        expect_early_invalid("backward misaligned payload " + std::to_string(field),
            factorized_interaction_direct_backward(
                f.plan, f.problem, f.precision, f.values, f.dy, f.tape,
                f.gradients, f.scratch,
                FactorizedInteractionGradientMode::overwrite, nullptr));
    }
}

} // namespace

extern "C" cudaError_t CUDARTAPI __wrap_cudaGetDevice(int* device) {
    ++device_query_count;
    if (device) *device = 0;
    // Fail closed if validation ever reaches the runtime. This keeps the
    // regression from launching a kernel on its deliberately fake addresses.
    return cudaErrorUnknown;
}

extern "C" cudaError_t CUDARTAPI __wrap_cudaMemsetAsync(
    void*, int, std::size_t, cudaStream_t) {
    ++memset_count;
    return cudaErrorUnknown;
}

int main() {
    test_workspace_contract();
    test_overlap_guards();
    test_forward_alignment_guards();
    test_backward_alignment_guards();
    std::puts("host contract guards passed; no CUDA query or memset was issued");
}
