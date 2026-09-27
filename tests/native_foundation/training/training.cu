#include <Cellerator/compute/operation/indexed_mechanism/training.hh>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

namespace ix = cellerator::compute::operation::indexed;
namespace ex = cellerator::execution;

namespace {
__global__ void overwrite_coefficients(float* values) {
    const auto i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < 2) values[i] = 42.0f + static_cast<float>(i);
}
void CUDART_CB delay_stream(void*) {
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
}
void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void cuda_check(cudaError_t status, const char* message) {
    if (status != cudaSuccess) throw std::runtime_error(std::string(message) + ": " + cudaGetErrorString(status));
}
template<class T> struct device_buffer {
    T* value = nullptr;
    explicit device_buffer(std::size_t count) {
        if (count) cuda_check(cudaMalloc(reinterpret_cast<void**>(&value), count * sizeof(T)), "cudaMalloc");
    }
    ~device_buffer() { if (value) cudaFree(value); }
    device_buffer(const device_buffer&) = delete;
    device_buffer& operator=(const device_buffer&) = delete;
};
template<class T> void upload(T* dst, const std::vector<T>& src) {
    cuda_check(cudaMemcpy(dst, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
}
template<class T> std::vector<T> download(const T* src, std::size_t count) {
    std::vector<T> dst(count);
    cuda_check(cudaMemcpy(dst.data(), src, count * sizeof(T), cudaMemcpyDeviceToHost), "download");
    return dst;
}
ex::persistent_axis_identity axis(std::uint64_t base) {
    return {{ex::biological_abi_version, ex::serialized_record_kind::persistent_axis_identity,
             sizeof(ex::persistent_axis_identity)},
            {base, 1}, {base + 1, 1}, {base + 2, 1}, {base + 3, 1}};
}
ix::output_index destination(std::uint64_t slot, std::uint64_t index, std::uint64_t role) {
    ix::output_index out{slot, {role, 1}, 0, index, {501, 1}};
    out.effect.update = ex::output_update_kind::accumulate;
    out.effect.requires_initialized_destination = true;
    return out;
}
struct reference_result { std::vector<float> y, dx, dk; };
reference_result reference(const ix::mechanism_declaration& d,
        const std::vector<float>& input, const std::vector<float>& coeff,
        const std::vector<float>& dy, std::uint64_t batch, bool mixed) {
    auto stored = [mixed](float x) {
        return mixed ? __half2float(__float2half_rn(x)) : x;
    };
    reference_result r{{}, std::vector<float>(input.size(), 0), std::vector<float>(coeff.size(), 0)};
    r.y.assign(batch * d.output.extent, 0);
    for (std::uint64_t b = 0; b < batch; ++b) {
        for (const auto& m : d.mechanisms) {
            std::vector<double> factors;
            for (const auto& a : m.arguments) factors.push_back(stored(input[b * d.input.extent + a.index]));
            double product = 1;
            for (double x : factors) product *= x;
            const auto k = stored(coeff[m.coefficient]);
            double upstream = 0;
            for (const auto& out : m.outputs) {
                r.y[b * d.output.extent + out.destination.index] +=
                    static_cast<float>(out.scale * k * product);
                upstream += out.scale * dy[b * d.output.extent + out.destination.index];
            }
            r.dk[m.coefficient] += static_cast<float>(upstream * product);
            for (std::size_t j = 0; j < m.arguments.size(); ++j) {
                double other = 1;
                for (std::size_t q = 0; q < m.arguments.size(); ++q) if (q != j) other *= factors[q];
                r.dx[b * d.input.extent + m.arguments[j].index] +=
                    static_cast<float>(upstream * k * other);
            }
        }
    }
    return r;
}
void near(const std::vector<float>& actual, const std::vector<float>& expected,
          float tolerance, const char* message) {
    check(actual.size() == expected.size(), message);
    for (std::size_t i = 0; i < actual.size(); ++i)
        check(std::isfinite(actual[i]) && std::abs(actual[i] - expected[i]) <= tolerance,
              message);
}
ix::mechanism_declaration declaration(ix::training_precision precision) {
    ix::mechanism_declaration d;
    d.input = {axis(100), 3}; d.output = {axis(200), 2}; d.coefficients = {axis(300), 2};
    d.coefficient_ids = {{701, 1}, {702, 1}};
    d.max_batch = 2; d.max_live_forwards = 2; d.precision = precision;
    // The repeated x0 slot is intentionally retained. The first two mechanisms
    // share k0, and all contributions to y0 declare one additive owner.
    d.mechanisms = {
        {{801, 1}, {{0, {901,1}, 0, 0}, {1, {902,1}, 0, 0}, {2, {903,1}, 0, 1}},
            {{destination(0, 0, 911), 1.0f}}, 0},
        {{802, 1}, {{0, {904,1}, 0, 2}},
            {{destination(0, 0, 912), -0.75f}}, 0},
        {{803, 1}, {{0, {905,1}, 0, 1}, {1, {906,1}, 0, 2}},
            {{destination(0, 1, 913), 0.5f}}, 1}
    };
    return d;
}
void run_precision(ix::training_precision precision, int device, cudaStream_t stream,
                   cudaStream_t stream2) {
    const bool mixed = precision == ix::training_precision::mixed_f16;
    auto d = declaration(precision);
    const std::vector<float> initial_coeff{0.33331f, -0.81273f};
    auto owner = std::make_shared<ix::mechanism_parameter_owner>(
        d.coefficients, d.coefficient_ids, initial_coeff, device, 8);
    const auto prepare_start = std::chrono::steady_clock::now();
    auto program = std::make_shared<ix::prepared_mechanism_program>(d, owner);
    const auto prepare_us = std::chrono::duration<double, std::micro>(
        std::chrono::steady_clock::now() - prepare_start).count();

    const std::vector<float> x{1.375f, -0.625f, 2.25f, 1.25f, 0.0f, 0.875f};
    const std::vector<float> dy{0.75f, -1.125f, -0.25f, 2.0f};
    device_buffer<float> dx(6), dk(2), dy_dev(4), y(4), x_dev(6);
    upload(x_dev.value, x); upload(dy_dev.value, dy);
    const auto fwd_start = std::chrono::steady_clock::now();
    auto tape = program->forward(x_dev.value, false, 2, d.input.identity, y.value, stream);
    cuda_check(cudaStreamSynchronize(stream), "forward synchronize");
    const auto fwd_us = std::chrono::duration<double, std::micro>(
        std::chrono::steady_clock::now() - fwd_start).count();
    const auto actual_y = download(y.value, 4);
    const auto expected = reference(d, x, initial_coeff, dy, 2, mixed);
    near(actual_y, expected.y, mixed ? 2e-6f : 2e-6f, "native forward differs from independent double reference");
    device_buffer<float> y2(4);
    auto tape2 = program->forward(x_dev.value, false, 2, d.input.identity, y2.value, stream2);
    cuda_check(cudaStreamSynchronize(stream2), "second forward synchronize");
    bool rejected = false;
    try { owner->begin_write(stream); } catch (const std::exception&) { rejected = true; }
    check(rejected, "outstanding tape must reject parameter writes");
    rejected = false;
    try { (void)program->forward(x_dev.value, false, 3, d.input.identity, y.value, stream); }
    catch (const std::exception&) { rejected = true; }
    check(rejected, "batch above reserved capacity must reject");
    auto wrong_axis = d.input.identity; wrong_axis.geometry.low++;
    rejected = false;
    try { (void)program->forward(x_dev.value, false, 1, wrong_axis, y.value, stream); }
    catch (const std::exception&) { rejected = true; }
    check(rejected, "same-shaped wrong biological axis must reject");

    const auto bwd_start = std::chrono::steady_clock::now();
    program->backward(*tape, dy_dev.value, dx.value, dk.value, stream);
    cuda_check(cudaStreamSynchronize(stream), "backward synchronize");
    const auto bwd_us = std::chrono::duration<double, std::micro>(
        std::chrono::steady_clock::now() - bwd_start).count();
    near(download(dx.value, 6), expected.dx, mixed ? 3e-6f : 3e-6f,
         "native input VJP differs from independent double reference");
    near(download(dk.value, 2), expected.dk, mixed ? 3e-6f : 3e-6f,
         "native shared coefficient VJP differs from independent double reference");
    rejected = false;
    try { owner->begin_write(stream); } catch (const std::exception&) { rejected = true; }
    check(rejected, "second outstanding tape must continue to prevent an update");
    device_buffer<float> dx2_dev(6), dk2_dev(2);
    program->backward(*tape2, dy_dev.value, dx2_dev.value, dk2_dev.value, stream2);
    cuda_check(cudaStreamSynchronize(stream2), "second backward synchronize");
    near(download(dx2_dev.value, 6), expected.dx, mixed ? 3e-6f : 3e-6f,
         "second concurrent tape input VJP differs from reference");
    near(download(dk2_dev.value, 2), expected.dk, mixed ? 3e-6f : 3e-6f,
         "second concurrent tape coefficient VJP differs from reference");
    rejected = false;
    try { program->backward(*tape, dy_dev.value, dx.value, dk.value, stream); }
    catch (const std::exception&) { rejected = true; }
    check(rejected, "a mechanism tape must reject backward replay");

    // An abandoned tape releases its read ticket, while the completion event
    // still orders reuse of the associated saved-operand slot.
    auto abandoned = program->forward(x_dev.value, false, 2, d.input.identity, y.value, stream2);
    abandoned.reset();
    const std::vector<float> next_coeff{0.25f, -0.5f};
    owner->begin_write(stream);
    cuda_check(cudaMemcpyAsync(owner->data(), next_coeff.data(), next_coeff.size() * sizeof(float),
                               cudaMemcpyHostToDevice, stream), "optimizer-like parameter write");
    owner->publish_write(stream);
    cuda_check(cudaStreamSynchronize(stream), "parameter publication synchronize");
    check(owner->generation() == 2, "sanctioned publication must advance generation exactly once");
    if (mixed) {
        const auto half = download(owner->half_data(), 2);
        check(half[0] == __half_as_ushort(__float2half_rn(next_coeff[0])) &&
              half[1] == __half_as_ushort(__float2half_rn(next_coeff[1])),
              "half plane must refresh from the FP32 master with RNE");
    }
    owner->begin_write(stream2);
    cuda_check(cudaLaunchHostFunc(stream2, delay_stream, nullptr), "queue failed-writer delay");
    overwrite_coefficients<<<1, 32, 0, stream2>>>(owner->data());
    cuda_check(cudaGetLastError(), "queue partial optimizer write");
    rejected = false;
    try { owner->restore(next_coeff, stream); } catch (const std::exception&) { rejected = true; }
    check(rejected, "checkpoint restore must reject an active, nonpoisoned writer");
    check(owner->generation() == 2 && !owner->poisoned(),
          "rejected restore must leave the admitted writer and generation unchanged");
    owner->poison();
    rejected = false;
    try { (void)program->forward(x_dev.value, false, 1, d.input.identity, y.value, stream); }
    catch (const std::exception&) { rejected = true; }
    check(rejected, "poisoned coefficient state must reject new forwards");
    owner->restore(next_coeff, stream);
    cuda_check(cudaStreamSynchronize(stream2), "failed writer stream synchronize");
    check(owner->generation() == 3, "checkpoint restore republishes one new generation");
    near(owner->snapshot(stream), next_coeff, 0.0f, "checkpoint restore must restore canonical coefficients");
    std::cout << (mixed ? "mixed" : "f32") << " prepare_us=" << prepare_us
              << " reserved_bytes=" << program->reserved_bytes()
              << " forward_us=" << fwd_us << " backward_us=" << bwd_us << '\n';
}

void rejects_incompatible_assembly(int device) {
    auto d = declaration(ix::training_precision::f32);
    auto ids = d.coefficient_ids;
    const std::vector<float> values{0.2f, 0.3f};
    auto owner = std::make_shared<ix::mechanism_parameter_owner>(d.coefficients, ids, values, device);
    d.mechanisms[1].outputs[0].destination.assembly_owner = {999, 1};
    bool rejected = false;
    try { auto p = std::make_shared<ix::prepared_mechanism_program>(std::move(d), owner); }
    catch (const std::exception&) { rejected = true; }
    check(rejected, "incompatible additive assembly owner must reject preparation");
}
} // namespace

int main() try {
    int device = 0;
    cuda_check(cudaGetDevice(&device), "cudaGetDevice");
    cudaStream_t stream{};
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream create");
    cudaStream_t stream2{};
    cuda_check(cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking), "second stream create");
    rejects_incompatible_assembly(device);
    run_precision(ix::training_precision::f32, device, stream, stream2);
    run_precision(ix::training_precision::mixed_f16, device, stream, stream2);
    cuda_check(cudaStreamDestroy(stream2), "second stream destroy");
    cuda_check(cudaStreamDestroy(stream), "stream destroy");
    std::cout << "native product training forward/VJP, identity, capacity, lifetime, assembly, and publication checks passed\n";
    return 0;
} catch (const std::exception& e) {
    std::cerr << "ce_ml2_native_training: " << e.what() << '\n';
    return 1;
}
