#include <Cellerator/compute/operation/active_projection/active_projection_v1.cuh>
#include <Cellerator/compute/operation/differential/local_arithmetic.hh>
#include <Cellerator/compute/operation/edge/dynamic_support_mask_v1.cuh>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace ap = cellerator::compute::operation::active_projection;
namespace df = cellerator::compute::differential;
namespace edge = cellerator::compute::operation::edge;
namespace ex = cellerator::execution;
namespace nf = cellerator::compute::operation::nf1;
namespace nn = cellerator::compute::native_numeric;
namespace pg = cellerator::execution::program;
constexpr unsigned repetitions = 7;

void ok(cudaError_t result) { if (result != cudaSuccess) std::abort(); }
void require(bool result) { if (!result) std::abort(); }
double host_time(auto &&call) {
    const auto begin = std::chrono::steady_clock::now(); call();
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - begin).count();
}
double gpu_time(cudaStream_t stream, auto &&call) {
    cudaEvent_t begin{}, end{}; ok(cudaEventCreate(&begin)); ok(cudaEventCreate(&end));
    ok(cudaEventRecord(begin, stream)); call(); ok(cudaEventRecord(end, stream)); ok(cudaEventSynchronize(end));
    float milliseconds{}; ok(cudaEventElapsedTime(&milliseconds, begin, end)); cudaEventDestroy(begin); cudaEventDestroy(end);
    return milliseconds * 1000.0;
}
double median(std::array<double, repetitions> values) { std::sort(values.begin(), values.end()); return values[repetitions / 2]; }
void print_array(const std::array<double, repetitions> &values) {
    std::printf("["); for (unsigned i = 0; i < repetitions; ++i) std::printf(i ? ",%.3f" : "%.3f", values[i]); std::printf("]");
}

struct resident {
    nn::resident_vector value{};
    resident() = default;
    ~resident() { ok(nn::release(&value)); }
    resident(const resident &) = delete;
    void allocate(std::size_t count) { ok(nn::allocate(&value, count, nn::device_representation::f32, 0)); }
    void upload(const std::vector<float> &host, std::uint64_t generation, cudaStream_t stream) {
        ok(nn::upload(value, host.data(), host.size(), {generation}, stream));
    }
};
nf::generation_stamp stamp(std::uint64_t identity, std::uint64_t generation) { return {{identity, 1}, {generation}}; }
nf::primal_record primal(nf::identity definition, std::uint64_t generation) {
    nf::primal_record result{}; result.instance.prepared = {definition, {2, 1}, {1}, {3, 1}};
    result.instance.state = stamp(4, generation); result.instance.parameters = stamp(5, generation); return result;
}

struct support_binding { const float *input{}; float *output{}; const std::uint8_t *mask{}; std::uint64_t count{}; };
pg::program_status support_admit(const void *state, const pg::launch_binding_v2 &binding, void *stream) noexcept {
    const auto *support = static_cast<const support_binding *>(binding.input);
    return stream && support && support->input && support->output && support->mask && support->count == *static_cast<const std::uint64_t *>(state)
        ? pg::program_status::success : pg::program_status::invalid_argument;
}
pg::program_status support_launch(const void *state, const pg::launch_binding_v2 &binding, void *stream) noexcept {
    if (support_admit(state, binding, stream) != pg::program_status::success) return pg::program_status::invalid_argument;
    const auto *support = static_cast<const support_binding *>(binding.input);
    const auto result = edge::enqueue_dynamic_support_mask_v1({{17, static_cast<std::uint32_t>(*static_cast<const std::uint64_t *>(state))},
        support->input, support->output, support->mask, edge::mask_encoding_v1::byte_per_edge, 17, 1, 1, 1, 1, 0,
        static_cast<cudaStream_t>(stream)});
    return result == edge::status_v1::success ? pg::program_status::success : pg::program_status::launch_failed;
}

struct program_fixture {
    cudaStream_t stream{}; std::size_t count{}; resident left, right, left_direction, right_direction, response, observed;
    std::uint8_t *device_mask{}; std::vector<float> host_left, host_right, host_direction, host_output;
    std::vector<std::uint8_t> host_mask; std::vector<std::uint32_t> primal_indices, derivative_indices;
    std::array<nf::operand_signature, 2> inputs{}; nf::output_signature output{}; nf::operation_contract contract{};
    df::local_block response_block{}; df::response_binding<float> response_request{}; support_binding support_request{};
    std::array<pg::prepared_stage_v2, 2> stages{}; std::array<std::uint64_t, 1> dependencies{{0}}; pg::prepared_program_v2 program{};
    program_fixture(std::size_t width, bool sparse) : count(width), host_left(width), host_right(width), host_direction(width, .05f),
        host_output(width), host_mask(width, 1u), primal_indices(width), derivative_indices(width) {
        for (std::size_t i = 0; i < width; ++i) { host_left[i] = .2f + .01f * float(i); host_right[i] = -.3f + .02f * float(i); if (sparse && i % 2u) host_mask[i] = 0; }
    }
    ~program_fixture() { if (device_mask) cudaFree(device_mask); if (stream) cudaStreamDestroy(stream); }
    void allocate() {
        ok(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        for (auto *item : {&left, &right, &left_direction, &right_direction, &response, &observed}) item->allocate(count);
        ok(cudaMalloc(&device_mask, count));
    }
    void initial_upload() {
        left.upload(host_left, 1, stream); right.upload(host_right, 1, stream); left_direction.upload(host_direction, 1, stream);
        right_direction.upload(host_direction, 1, stream); response.upload(host_output, 1, stream); observed.upload(host_output, 1, stream); ok(cudaStreamSynchronize(stream));
    }
    void prepare() {
        inputs = {nf::operand_signature{{10, 1}, {}, count, ex::numeric_type::f32}, nf::operand_signature{{11, 1}, {}, count, ex::numeric_type::f32}};
        output.operand = {{12, 1}, {}, count, ex::numeric_type::f32}; output.assembly_owner = {13, 1};
        contract.definition = {200, 1}; contract.arguments = {inputs.data(), inputs.size()}; contract.outputs = {&output, 1};
        contract.numeric = {ex::numeric_type::f32, ex::numeric_type::f32, ex::numeric_type::f32, ex::numeric_type::f32, ex::numeric_type::f32, ex::numeric_type::f32};
        contract.capabilities = nf::forward | nf::jvp | nf::vjp | nf::second_direction;
        df::local_primal_owners owners{&left.value, &right.value, &left.value, &right.value, stream, primal(contract.definition, 1)};
        require(df::make_local_device_block(nn::local_operation::multiply, contract, owners, response_block) == nf::status::success);
        require(df::make_local_device_stage(response_block, nf::jvp, 1, 1, 0, stages[0]) == nf::status::success);
        stages[1] = {2, 2, &count, support_launch, 0, 1, 1, 0, support_admit}; program = {2, 0, stages.data(), 2, dependencies.data(), 1};
        ap::projection_map_v1 primal_map{}, derivative_map{};
        require(ap::build_exact_projection_v1({{17, 1, {ap::activity_scope_v1::per_instance, 9, 1}}, host_mask.data(), host_mask.data(),
            static_cast<std::uint32_t>(count), primal_indices.data(), static_cast<std::uint32_t>(count), derivative_indices.data(), static_cast<std::uint32_t>(count)},
            &primal_map, &derivative_map) == ap::status_v1::success);
        require(primal_map.active_count == derivative_map.active_count);
    }
    double refresh(unsigned generation) { host_left[0] = .2f + .001f * generation; return gpu_time(stream, [&] { left.upload(host_left, generation, stream); }); }
    double upload_support() { return gpu_time(stream, [&] { ok(cudaMemcpyAsync(device_mask, host_mask.data(), count, cudaMemcpyHostToDevice, stream)); }); }
    double execute(unsigned generation, double &host_launch) {
        response_request = {}; response_request.left_direction = &left_direction.value; response_request.right_direction = &right_direction.value;
        response_request.output = &response.value; response_request.count = count; response_request.request.action = nf::jvp;
        response_request.request.primal = primal(contract.definition, generation); response_request.request.primal.instance.parameters = stamp(5, 1); response_request.request.direction_domain = inputs[0]; response_request.request.response_domain = output.operand;
        response_request.direction = inputs[0]; response_request.response = output.operand; support_request = {static_cast<const float *>(response.value.data), static_cast<float *>(observed.value.data), device_mask, count};
        std::array<pg::launch_binding_v2, 2> bindings{}; bindings[0].input = &response_request; bindings[1].input = &support_request;
        cudaEvent_t begin{}, end{}; ok(cudaEventCreate(&begin)); ok(cudaEventCreate(&end)); ok(cudaEventRecord(begin, stream));
        host_launch = host_time([&] { require(pg::execute_prepared_program_v2(program, bindings.data(), bindings.size(), stream) == pg::program_status::success); });
        ok(cudaEventRecord(end, stream)); ok(cudaEventSynchronize(end)); float milliseconds{}; ok(cudaEventElapsedTime(&milliseconds, begin, end)); cudaEventDestroy(begin); cudaEventDestroy(end); return milliseconds * 1000.0;
    }
    double observe() { return gpu_time(stream, [&] { ok(nn::download(observed.value, host_output.data(), count, stream)); }); }
};

int main() {
    ok(cudaSetDevice(0)); cudaDeviceProp device{}; ok(cudaGetDeviceProperties(&device, 0)); if (device.major != 7 || device.minor != 0) return 2;
    for (const auto width : {16u, 33u, 65u}) for (const auto sparse : {false, true}) {
        std::array<double, repetitions> cold{}, resident_cost{}, amortized{}, allocation{}, preparation{}, initial_upload{}, refresh{}, support{}, response{}, launch{}, observation{};
        for (unsigned i = 0; i < repetitions; ++i) {
            program_fixture item(width, sparse); allocation[i] = host_time([&] { item.allocate(); }); initial_upload[i] = gpu_time(item.stream, [&] { item.initial_upload(); });
            preparation[i] = host_time([&] { item.prepare(); }); refresh[i] = item.refresh(i + 2); support[i] = item.upload_support(); response[i] = item.execute(i + 2, launch[i]); observation[i] = item.observe();
            cold[i] = allocation[i] + preparation[i] + initial_upload[i] + refresh[i] + support[i] + response[i] + launch[i] + observation[i];
        }
        program_fixture item(width, sparse); item.allocate(); item.initial_upload(); item.prepare(); const double setup = median(allocation) + median(preparation) + median(initial_upload);
        for (unsigned i = 0; i < repetitions; ++i) {
            const double current_refresh = item.refresh(i + 20), current_support = item.upload_support(); double current_launch{};
            const double current_response = item.execute(i + 20, current_launch), current_observation = item.observe();
            resident_cost[i] = current_support + current_response + current_launch + current_observation;
            amortized[i] = setup / repetitions + current_refresh + current_support + current_response + current_launch + current_observation;
        }
        std::printf("{\"width\":%u,\"support\":\"%s\",\"support_route\":\"persistent_exact_mask_after_typed_jvp\",\"repeat_count\":7,\"cold_us\":", width, sparse ? "exact_active_half" : "dense_full"); print_array(cold);
        std::printf(",\"resident_us\":"); print_array(resident_cost); std::printf(",\"amortized_us\":"); print_array(amortized);
        std::printf(",\"phase_medians_us\":{\"allocation\":%.3f,\"prepare\":%.3f,\"initial_upload\":%.3f,\"value_refresh\":%.3f,\"support\":%.3f,\"response_completion\":%.3f,\"program_launch_host\":%.3f,\"observation_download\":%.3f},\"medians_us\":{\"cold\":%.3f,\"resident\":%.3f,\"amortized\":%.3f},\"launches_per_program\":2,\"device_bytes\":%llu,\"compact_response_route\":\"unsupported_no_typed_compact_local_response_binding\",\"fp16_response_route\":\"unsupported\"}\n",
            median(allocation), median(preparation), median(initial_upload), median(refresh), median(support), median(response), median(launch), median(observation), median(cold), median(resident_cost), median(amortized), static_cast<unsigned long long>(6ull * width * sizeof(float) + width));
    }
}
