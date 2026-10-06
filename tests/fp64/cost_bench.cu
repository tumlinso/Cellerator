#include <Cellerator/compute/candidate/sparse/project.hh>
#include <Cellerator/compute/operation/device_elementwise.hh>
#include <Cellerator/compute/operation/prepared_relation.hh>
#include <Cellerator/runtime/stream.cuh>

#include <cuda_runtime.h>
#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <type_traits>
#include <vector>

namespace ce = cellerator::compute::relation;
namespace ex = cellerator::execution;
namespace project = cellerator::compute::sparse::project;
namespace runtime = cellerator::runtime;

namespace {

constexpr std::uint32_t kRows = 2048;
constexpr std::uint32_t kCols = 2048;
constexpr std::uint32_t kDegree = 16;
constexpr std::uint32_t kWidth = 65;
constexpr int kWarmups = 20;
constexpr int kRepetitions = 200;

void cuda_check(cudaError_t result) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA: " << cudaGetErrorString(result) << '\n';
        std::abort();
    }
}

void relation_check(ce::status result) {
    if (!result) {
        std::cerr << "prepared relation: " << result.message << " (status "
                  << unsigned(result.code) << ")\n";
        std::abort();
    }
}

template<class T> T* allocate(std::uint64_t count) {
    T* result = nullptr;
    if (count) cuda_check(cudaMalloc(reinterpret_cast<void**>(&result), count * sizeof(T)));
    return result;
}

template<class T> T* allocate_pinned(std::uint64_t count) {
    T* result = nullptr;
    if (count) cuda_check(cudaHostAlloc(reinterpret_cast<void**>(&result), count * sizeof(T), cudaHostAllocDefault));
    return result;
}

float elapsed_ms(cudaEvent_t begin, cudaEvent_t end) {
    float result = 0.0f;
    cuda_check(cudaEventElapsedTime(&result, begin, end));
    return result;
}

struct timing {
    double batch_ms = 0.0;
    double per_call_ms = 0.0;
};

template<class F> timing time_calls(cudaStream_t stream, int count, F&& call) {
    cudaEvent_t begin{}, end{};
    cuda_check(cudaEventCreate(&begin));
    cuda_check(cudaEventCreate(&end));
    cuda_check(cudaEventRecord(begin, stream));
    for (int i = 0; i < count; ++i) call();
    cuda_check(cudaEventRecord(end, stream));
    cuda_check(cudaEventSynchronize(end));
    const double batch = elapsed_ms(begin, end);
    cuda_check(cudaEventDestroy(begin));
    cuda_check(cudaEventDestroy(end));
    return {batch, batch / count};
}

ce::axis_descriptor make_axis(std::uint32_t base, std::uint32_t extent) {
    ce::axis_descriptor result{};
    result.extent = extent;
    result.identity.header = {1, ex::serialized_record_kind::persistent_axis_identity,
                              sizeof(result.identity)};
    result.identity.domain = {base, 11};
    result.identity.order = {base + 1, 12};
    result.identity.geometry = {base + 2, 13};
    result.identity.partition = {base + 3, 14};
    return result;
}

template<class T> constexpr ex::numeric_type numeric_type() {
    return std::is_same_v<T, float> ? ex::numeric_type::f32 : ex::numeric_type::f64;
}

template<class W, class X, class Y>
void run_relation_case(const char* label, cudaStream_t stream, int device_id) {
    constexpr std::uint64_t nnz = std::uint64_t(kRows) * kDegree;
    constexpr std::uint64_t feature_count = std::uint64_t(kCols) * kWidth;
    constexpr std::uint64_t output_count = std::uint64_t(kRows) * kWidth;

    std::vector<std::uint32_t> offsets(kRows + 1);
    std::vector<std::uint32_t> indices(nnz);
    for (std::uint32_t row = 0; row <= kRows; ++row) offsets[row] = row * kDegree;
    for (std::uint32_t row = 0; row < kRows; ++row) {
        for (std::uint32_t edge = 0; edge < kDegree; ++edge) {
            indices[std::uint64_t(row) * kDegree + edge] =
                (row * 17u + edge * 131u + edge * edge * 7u) % kCols;
        }
    }

    ce::operation_descriptor forward{};
    forward.topology.identity = {0x16001u, 0x26001u};
    forward.topology.epoch = {1};
    forward.topology.source = make_axis(0x16100u, kCols);
    forward.topology.destination = make_axis(0x16200u, kRows);
    forward.topology.logical_edge_order = {0x36001u, 0x46001u};
    forward.topology.edge_count = nnz;
    forward.dense_width = kWidth;
    forward.arithmetic.relation_storage = numeric_type<W>();
    forward.arithmetic.input_storage = numeric_type<X>();
    forward.arithmetic.multiply = numeric_type<Y>();
    forward.arithmetic.accumulation = numeric_type<Y>();
    forward.arithmetic.output_storage = numeric_type<Y>();
    forward.arithmetic.permit_fma = true;
    forward.arithmetic.permit_reassociation = true;
    forward.update = ce::output_update::overwrite;
    auto transpose = forward;
    transpose.direction = ce::orientation::transpose;

    auto prepare_pair = [&](ce::prepared_relation_pair** destination) {
        relation_check(ce::prepare_relation_pair(forward, transpose,
            {offsets.data(), offsets.size(), indices.data(), indices.size()},
            {device_id, 1ull << 32, false}, stream, destination));
    };
    auto cold_prepare = [&] {
        ce::prepared_relation_pair* temporary = nullptr;
        prepare_pair(&temporary);
        ce::destroy(temporary);
    };
    for (int i = 0; i < kWarmups; ++i) cold_prepare();
    std::vector<double> preparation_samples;
    preparation_samples.reserve(kRepetitions);
    for (int i = 0; i < kRepetitions; ++i) {
        ce::prepared_relation_pair* temporary = nullptr;
        const auto begin = std::chrono::steady_clock::now();
        prepare_pair(&temporary);
        const auto end = std::chrono::steady_clock::now();
        preparation_samples.push_back(std::chrono::duration<double, std::milli>(end - begin).count());
        ce::destroy(temporary);
    }
    auto sorted_preparation = preparation_samples;
    std::sort(sorted_preparation.begin(), sorted_preparation.end());
    const double preparation_mean_ms = std::accumulate(preparation_samples.begin(), preparation_samples.end(), 0.0) / kRepetitions;
    const double preparation_median_ms = sorted_preparation[kRepetitions / 2];

    ce::prepared_relation_pair* pair = nullptr;
    prepare_pair(&pair);

    std::vector<W> weights(nnz);
    for (std::uint64_t edge = 0; edge < nnz; ++edge) {
        const int centered = static_cast<int>((edge * 13u + edge / 7u) % 31u) - 15;
        weights[edge] = static_cast<W>(centered / 32.0);
    }
    std::vector<X> features(feature_count);
    for (std::uint64_t i = 0; i < feature_count; ++i) {
        const int centered = static_cast<int>((i * 19u + i / 11u) % 127u) - 63;
        features[i] = static_cast<X>(centered / 64.0);
    }
    W* device_weights = allocate<W>(nnz);
    X* device_features = allocate<X>(feature_count);
    Y* device_output = allocate<Y>(output_count);
    Y* element_a = allocate<Y>(output_count);
    Y* element_b = allocate<Y>(output_count);
    Y* element_output = allocate<Y>(output_count);
    std::vector<Y> host_a(output_count), host_b(output_count);
    for (std::uint64_t i = 0; i < output_count; ++i) {
        host_a[i] = static_cast<Y>((static_cast<int>(i % 101u) - 50) / 32.0);
        host_b[i] = static_cast<Y>((static_cast<int>((i * 7u) % 97u) - 48) / 64.0);
    }
    W* pinned_weights = allocate_pinned<W>(nnz);
    X* pinned_features = allocate_pinned<X>(feature_count);
    Y* pinned_a = allocate_pinned<Y>(output_count);
    Y* pinned_b = allocate_pinned<Y>(output_count);
    Y* pinned_output = allocate_pinned<Y>(output_count);
    std::copy(weights.begin(), weights.end(), pinned_weights);
    std::copy(features.begin(), features.end(), pinned_features);
    std::copy(host_a.begin(), host_a.end(), pinned_a);
    std::copy(host_b.begin(), host_b.end(), pinned_b);

    std::uint64_t generation = 0;
    auto transfer_call = [&] {
        ++generation;
        cuda_check(cudaMemcpyAsync(device_weights, pinned_weights, nnz * sizeof(W), cudaMemcpyHostToDevice, stream));
        cuda_check(cudaMemcpyAsync(device_features, pinned_features, feature_count * sizeof(X), cudaMemcpyHostToDevice, stream));
        cuda_check(cudaMemcpyAsync(element_a, pinned_a, output_count * sizeof(Y), cudaMemcpyHostToDevice, stream));
        cuda_check(cudaMemcpyAsync(element_b, pinned_b, output_count * sizeof(Y), cudaMemcpyHostToDevice, stream));
        if constexpr (std::is_same_v<W, float>) {
            ce::device_f32_values_binding binding{device_weights, nnz, forward.topology.identity,
                forward.topology.epoch, forward.topology.logical_edge_order, {generation}, device_id};
            relation_check(ce::publish_f32_values(*pair, binding, stream));
        } else {
            ce::device_f64_values_binding binding{device_weights, nnz, forward.topology.identity,
                forward.topology.epoch, forward.topology.logical_edge_order, {generation}, device_id};
            relation_check(ce::publish_f64_values(*pair, binding, stream));
        }
    };
    for (int i = 0; i < kWarmups; ++i) transfer_call();
    cuda_check(cudaStreamSynchronize(stream));
    cudaEvent_t upload_begin{}, upload_end{};
    cuda_check(cudaEventCreate(&upload_begin));
    cuda_check(cudaEventCreate(&upload_end));
    cuda_check(cudaEventRecord(upload_begin, stream));
    for (int i = 0; i < kRepetitions; ++i) transfer_call();
    cuda_check(cudaEventRecord(upload_end, stream));
    cuda_check(cudaEventSynchronize(upload_end));
    const double upload_batch_ms = elapsed_ms(upload_begin, upload_end);
    const double upload_ms = upload_batch_ms / kRepetitions;
    cuda_check(cudaEventDestroy(upload_begin));
    cuda_check(cudaEventDestroy(upload_end));

    ce::preparation_report report{};
    relation_check(ce::inspect(*pair, &report));
    const std::uint64_t provider_persistent = report.shared_structural_bytes + report.instance_value_bytes;
    const std::uint64_t estimated_host_topology = nnz * 20;
    const std::uint64_t estimated_device_persistent = provider_persistent >= estimated_host_topology
        ? provider_persistent - estimated_host_topology : 0;
    const std::uint64_t caller_buffer_bytes = nnz * sizeof(W) + feature_count * sizeof(X) +
        output_count * sizeof(Y) * 4;
    const std::uint64_t pinned_staging_bytes = nnz * sizeof(W) + feature_count * sizeof(X) +
        output_count * sizeof(Y) * 3;
    const std::uint64_t host_to_device_bytes = nnz * sizeof(W) + feature_count * sizeof(X) +
        output_count * sizeof(Y) * 2;
    constexpr std::uint64_t provider_temporary_bytes = 0; // no temporary planes in forward/transpose apply

    ce::device_state_view input{device_features, feature_count, forward.topology.source, device_id, numeric_type<X>()};
    ce::device_result_view output{device_output, output_count, forward.topology.destination, device_id, numeric_type<Y>()};
    ce::device_state_view transpose_input{device_features, feature_count,
        forward.topology.destination, device_id, numeric_type<X>()};
    ce::device_result_view transpose_output{device_output, output_count,
        forward.topology.source, device_id, numeric_type<Y>()};
    runtime::execution_context context{device_id, stream, false};
    const ex::value_generation current_generation{generation};
    auto apply = [&] { relation_check(ce::enqueue(*pair, forward, input, output, current_generation, stream)); };
    auto apply_transpose = [&] {
        relation_check(ce::enqueue(*pair, transpose, transpose_input, transpose_output,
                                   current_generation, stream));
    };
    auto element_pipeline = [&] {
        auto status = ce::multiply(context, element_a, element_b, element_output, output_count);
        if (!status) { std::cerr << status.message << '\n'; std::abort(); }
        status = ce::square(context, element_output, element_output, output_count);
        if (!status) { std::cerr << status.message << '\n'; std::abort(); }
        status = ce::axpby(context, Y(1.0000001), element_a, Y(-0.25), element_b, element_output, output_count);
        if (!status) { std::cerr << status.message << '\n'; std::abort(); }
    };
    for (int i = 0; i < kWarmups; ++i) { apply(); apply_transpose(); element_pipeline(); }
    cuda_check(cudaStreamSynchronize(stream));
    const timing relation_time = time_calls(stream, kRepetitions, apply);
    const timing transpose_time = time_calls(stream, kRepetitions, apply_transpose);
    const timing element_time = time_calls(stream, kRepetitions, element_pipeline);

    auto download = [&] {
        cuda_check(cudaMemcpyAsync(pinned_output, device_output, output_count * sizeof(Y),
                                   cudaMemcpyDeviceToHost, stream));
    };
    for (int i = 0; i < kWarmups; ++i) download();
    cuda_check(cudaStreamSynchronize(stream));
    cudaEvent_t d2h_begin{}, d2h_end{};
    cuda_check(cudaEventCreate(&d2h_begin));
    cuda_check(cudaEventCreate(&d2h_end));
    cuda_check(cudaEventRecord(d2h_begin, stream));
    for (int i = 0; i < kRepetitions; ++i) download();
    cuda_check(cudaEventRecord(d2h_end, stream));
    cuda_check(cudaEventSynchronize(d2h_end));
    const double d2h_batch_ms = elapsed_ms(d2h_begin, d2h_end);
    const double d2h_ms = d2h_batch_ms / kRepetitions;
    cuda_check(cudaEventDestroy(d2h_begin));
    cuda_check(cudaEventDestroy(d2h_end));

    auto whole_transaction = [&] {
        ce::prepared_relation_pair* transaction_pair = nullptr;
        prepare_pair(&transaction_pair);
        cuda_check(cudaMemcpyAsync(device_weights, pinned_weights, nnz * sizeof(W), cudaMemcpyHostToDevice, stream));
        cuda_check(cudaMemcpyAsync(device_features, pinned_features, feature_count * sizeof(X), cudaMemcpyHostToDevice, stream));
        cuda_check(cudaMemcpyAsync(element_a, pinned_a, output_count * sizeof(Y), cudaMemcpyHostToDevice, stream));
        cuda_check(cudaMemcpyAsync(element_b, pinned_b, output_count * sizeof(Y), cudaMemcpyHostToDevice, stream));
        if constexpr (std::is_same_v<W, float>) {
            ce::device_f32_values_binding binding{device_weights, nnz, forward.topology.identity,
                forward.topology.epoch, forward.topology.logical_edge_order, {1}, device_id};
            relation_check(ce::publish_f32_values(*transaction_pair, binding, stream));
        } else {
            ce::device_f64_values_binding binding{device_weights, nnz, forward.topology.identity,
                forward.topology.epoch, forward.topology.logical_edge_order, {1}, device_id};
            relation_check(ce::publish_f64_values(*transaction_pair, binding, stream));
        }
        ce::device_state_view transaction_input{device_features, feature_count,
            forward.topology.source, device_id, numeric_type<X>()};
        ce::device_result_view transaction_output{device_output, output_count,
            forward.topology.destination, device_id, numeric_type<Y>()};
        relation_check(ce::enqueue(*transaction_pair, forward, transaction_input,
                                   transaction_output, {1}, stream));
        auto transpose_input = transaction_input;
        transpose_input.axis = forward.topology.destination;
        auto transpose_output = transaction_output;
        transpose_output.axis = forward.topology.source;
        relation_check(ce::enqueue(*transaction_pair, transpose, transpose_input,
                                   transpose_output, {1}, stream));
        auto status = ce::multiply(context, element_a, element_b, element_output, output_count);
        if (!status) { std::cerr << status.message << '\n'; std::abort(); }
        status = ce::square(context, element_output, element_output, output_count);
        if (!status) { std::cerr << status.message << '\n'; std::abort(); }
        status = ce::axpby(context, Y(1.0000001), element_a, Y(-0.25), element_b,
                           element_output, output_count);
        if (!status) { std::cerr << status.message << '\n'; std::abort(); }
        download();
        cuda_check(cudaStreamSynchronize(stream));
        ce::destroy(transaction_pair);
    };
    for (int i = 0; i < kWarmups; ++i) whole_transaction();
    std::vector<double> end_to_end_samples;
    end_to_end_samples.reserve(kRepetitions);
    for (int i = 0; i < kRepetitions; ++i) {
        const auto begin = std::chrono::steady_clock::now();
        whole_transaction();
        const auto end = std::chrono::steady_clock::now();
        end_to_end_samples.push_back(std::chrono::duration<double, std::milli>(end - begin).count());
    }
    auto sorted_end_to_end = end_to_end_samples;
    std::sort(sorted_end_to_end.begin(), sorted_end_to_end.end());
    const double end_to_end_mean_ms = std::accumulate(end_to_end_samples.begin(), end_to_end_samples.end(), 0.0) / kRepetitions;
    const double end_to_end_median_ms = sorted_end_to_end[kRepetitions / 2];
    std::cout << std::setprecision(10)
        << "{\"kind\":\"prepared_cost\",\"case\":\"" << label
        << "\",\"rows\":" << kRows << ",\"cols\":" << kCols
        << ",\"nnz\":" << nnz << ",\"width\":" << kWidth
        << ",\"warmups\":" << kWarmups << ",\"repetitions\":" << kRepetitions
        << ",\"prepare_mean_ms\":" << preparation_mean_ms
        << ",\"prepare_median_ms\":" << preparation_median_ms
        << ",\"upload_and_publish_batch_ms\":" << upload_batch_ms
        << ",\"upload_and_publish_per_call_ms\":" << upload_ms
        << ",\"host_to_device_bytes_per_call\":" << host_to_device_bytes
        << ",\"device_value_publication_bytes_per_call\":" << nnz * sizeof(W)
        << ",\"relation_resident_batch_ms\":" << relation_time.batch_ms
        << ",\"relation_resident_per_call_ms\":" << relation_time.per_call_ms
        << ",\"transpose_resident_batch_ms\":" << transpose_time.batch_ms
        << ",\"transpose_resident_per_call_ms\":" << transpose_time.per_call_ms
        << ",\"elementwise_resident_batch_ms\":" << element_time.batch_ms
        << ",\"elementwise_pipeline_per_iteration_ms\":" << element_time.per_call_ms
        << ",\"download_batch_ms\":" << d2h_batch_ms
        << ",\"download_per_call_ms\":" << d2h_ms
        << ",\"device_to_host_bytes_per_call\":" << output_count * sizeof(Y)
        << ",\"end_to_end_mean_ms\":" << end_to_end_mean_ms
        << ",\"end_to_end_median_ms\":" << end_to_end_median_ms
        << ",\"provider_persistent_bytes\":" << provider_persistent
        << ",\"estimated_host_topology_bytes\":" << estimated_host_topology
        << ",\"estimated_requested_device_persistent_bytes\":" << estimated_device_persistent
        << ",\"provider_shared_structural_bytes\":" << report.shared_structural_bytes
        << ",\"provider_instance_value_bytes\":" << report.instance_value_bytes
        << ",\"provider_temporary_device_bytes\":" << provider_temporary_bytes
        << ",\"provider_temporary_scope\":\"resident apply and elementwise execution only\""
        << ",\"caller_buffer_bytes\":" << caller_buffer_bytes
        << ",\"pinned_host_staging_bytes\":" << pinned_staging_bytes
        << ",\"reported_forward_candidate\":\"" << (report.forward_candidate ? report.forward_candidate : "")
        << "\",\"reported_transpose_candidate\":\"" << (report.transpose_candidate ? report.transpose_candidate : "")
        << "\"}\n";

    cudaFree(device_weights); cudaFree(device_features); cudaFree(device_output);
    cudaFree(element_a); cudaFree(element_b); cudaFree(element_output);
    cudaFreeHost(pinned_weights); cudaFreeHost(pinned_features); cudaFreeHost(pinned_a);
    cudaFreeHost(pinned_b); cudaFreeHost(pinned_output);
    ce::destroy(pair);
}

using old_csr_fn = void (*)(const runtime::execution_context*, const std::uint32_t*,
    const std::uint32_t*, const float*, std::uint32_t, std::uint32_t, const float*,
    std::int64_t, std::int64_t, float*, std::int64_t, const std::uint32_t*, float, float);

void run_direct_f32(const char* label, old_csr_fn baseline, cudaStream_t stream, int device_id) {
    constexpr std::uint64_t nnz = std::uint64_t(kRows) * kDegree;
    constexpr std::uint64_t feature_count = std::uint64_t(kCols) * kWidth;
    constexpr std::uint64_t output_count = std::uint64_t(kRows) * kWidth;
    std::vector<std::uint32_t> offsets(kRows + 1), indices(nnz);
    for (std::uint32_t row = 0; row <= kRows; ++row) offsets[row] = row * kDegree;
    for (std::uint32_t row = 0; row < kRows; ++row)
        for (std::uint32_t edge = 0; edge < kDegree; ++edge)
            indices[std::uint64_t(row) * kDegree + edge] = (row * 17u + edge * 131u + edge * edge * 7u) % kCols;
    std::vector<float> weights(nnz), features(feature_count);
    for (std::uint64_t edge = 0; edge < nnz; ++edge) {
        const int centered = static_cast<int>((edge * 13u + edge / 7u) % 31u) - 15;
        weights[edge] = static_cast<float>(centered / 32.0);
    }
    for (std::uint64_t i = 0; i < feature_count; ++i) {
        const int centered = static_cast<int>((i * 19u + i / 11u) % 127u) - 63;
        features[i] = static_cast<float>(centered / 64.0);
    }
    float* d_weights = allocate<float>(nnz);
    float* d_features = allocate<float>(feature_count);
    float* d_output = allocate<float>(output_count);
    std::uint32_t* d_offsets = allocate<std::uint32_t>(offsets.size());
    std::uint32_t* d_indices = allocate<std::uint32_t>(indices.size());
    cuda_check(cudaMemcpyAsync(d_offsets, offsets.data(), offsets.size() * sizeof(std::uint32_t), cudaMemcpyHostToDevice, stream));
    cuda_check(cudaMemcpyAsync(d_indices, indices.data(), indices.size() * sizeof(std::uint32_t), cudaMemcpyHostToDevice, stream));
    cuda_check(cudaMemcpyAsync(d_weights, weights.data(), nnz * sizeof(float), cudaMemcpyHostToDevice, stream));
    cuda_check(cudaMemcpyAsync(d_features, features.data(), feature_count * sizeof(float), cudaMemcpyHostToDevice, stream));
    cuda_check(cudaStreamSynchronize(stream));
    runtime::execution_context context{device_id, stream, false};
    auto current_call = [&] { project::csr_spmm_fwd_f32(context, d_offsets, d_indices, d_weights,
        kRows, kCols, d_features, kWidth, kWidth, d_output, kWidth); };
    auto old_call = [&] { baseline(&context, d_offsets, d_indices, d_weights, kRows,
        kCols, d_features, kWidth, kWidth, d_output, kWidth, nullptr, 1.0f, 0.0f); };
    for (int i = 0; i < kWarmups; ++i) { current_call(); old_call(); }
    cuda_check(cudaStreamSynchronize(stream));
    std::vector<float> current_output(output_count), baseline_output(output_count);
    current_call(); cuda_check(cudaStreamSynchronize(stream));
    cuda_check(cudaMemcpy(current_output.data(), d_output, output_count * sizeof(float), cudaMemcpyDeviceToHost));
    old_call(); cuda_check(cudaStreamSynchronize(stream));
    cuda_check(cudaMemcpy(baseline_output.data(), d_output, output_count * sizeof(float), cudaMemcpyDeviceToHost));
    double max_absolute_error = 0.0, max_relative_error = 0.0;
    for (std::uint64_t i = 0; i < output_count; ++i) {
        const double absolute = std::abs(double(current_output[i]) - baseline_output[i]);
        const double relative = absolute / std::max(1.0, std::abs(double(baseline_output[i])));
        max_absolute_error = std::max(max_absolute_error, absolute);
        max_relative_error = std::max(max_relative_error, relative);
    }
    const bool outputs_match = max_absolute_error <= 2.0e-6 *
        std::max(1.0, std::abs(double(*std::max_element(baseline_output.begin(), baseline_output.end(),
            [](float a, float b) { return std::abs(a) < std::abs(b); }))));
    if (!outputs_match) {
        std::cerr << "historical FP32 CSR baseline output mismatch: max_abs=" << max_absolute_error << '\n';
        std::abort();
    }
    auto timed_group = [&](auto&& call) {
        cudaEvent_t begin{}, end{};
        cuda_check(cudaEventCreate(&begin)); cuda_check(cudaEventCreate(&end));
        cuda_check(cudaEventRecord(begin, stream));
        for (int i = 0; i < 10; ++i) call();
        cuda_check(cudaEventRecord(end, stream)); cuda_check(cudaEventSynchronize(end));
        const double ms = elapsed_ms(begin, end);
        cuda_check(cudaEventDestroy(begin)); cuda_check(cudaEventDestroy(end));
        return ms;
    };
    double current_batch_ms = 0.0, baseline_batch_ms = 0.0;
    for (int block = 0; block < kRepetitions / 10; ++block) {
        if ((block % 2) == 0) {
            current_batch_ms += timed_group(current_call);
            baseline_batch_ms += timed_group(old_call);
        } else {
            baseline_batch_ms += timed_group(old_call);
            current_batch_ms += timed_group(current_call);
        }
    }
    const timing current{current_batch_ms, current_batch_ms / kRepetitions};
    const timing old{baseline_batch_ms, baseline_batch_ms / kRepetitions};
    const double regression = old.per_call_ms == 0.0 ? 0.0 :
        100.0 * (current.per_call_ms / old.per_call_ms - 1.0);
    std::cout << std::setprecision(10)
        << "{\"kind\":\"direct_f32_baseline\",\"case\":\"" << label
        << "\",\"rows\":" << kRows << ",\"cols\":" << kCols
        << ",\"nnz\":" << nnz << ",\"width\":" << kWidth
        << ",\"warmups\":" << kWarmups << ",\"repetitions\":" << kRepetitions
        << ",\"baseline_per_call_ms\":" << old.per_call_ms
        << ",\"current_per_call_ms\":" << current.per_call_ms
        << ",\"current_vs_baseline_percent\":" << regression
        << ",\"outputs_match\":" << (outputs_match ? "true" : "false")
        << ",\"max_absolute_output_error\":" << max_absolute_error
        << ",\"max_relative_output_error\":" << max_relative_error
        << ",\"regression_threshold_percent\":5}\n";
    cudaFree(d_weights); cudaFree(d_features); cudaFree(d_output); cudaFree(d_offsets); cudaFree(d_indices);
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 3 || std::string(argv[1]) != "--baseline-library") {
        std::cerr << "usage: fp64_cost_bench --baseline-library PATH\n";
        return 2;
    }
    void* library = dlopen(argv[2], RTLD_NOW | RTLD_LOCAL);
    if (!library) { std::cerr << "dlopen: " << dlerror() << '\n'; return 2; }
    auto baseline = reinterpret_cast<old_csr_fn>(dlsym(library, "fp64_old_csr_spmm_fwd_f32"));
    if (!baseline) { std::cerr << "baseline symbol missing: " << dlerror() << '\n'; return 2; }

    int device_id = 0;
    cuda_check(cudaSetDevice(device_id));
    cudaDeviceProp properties{};
    cuda_check(cudaGetDeviceProperties(&properties, device_id));
    int runtime_version = 0, driver_version = 0;
    cuda_check(cudaRuntimeGetVersion(&runtime_version));
    cuda_check(cudaDriverGetVersion(&driver_version));
    cudaStream_t stream{};
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    std::cout << "{\"kind\":\"environment\",\"device_id\":" << device_id
        << ",\"device_name\":\"" << properties.name << "\",\"compute_major\":" << properties.major
        << ",\"compute_minor\":" << properties.minor
        << ",\"runtime_version\":" << runtime_version
        << ",\"driver_version\":" << driver_version
        << ",\"warmups\":" << kWarmups << ",\"repetitions\":" << kRepetitions << "}\n";
    run_relation_case<float, float, float>("f32_f32_f32", stream, device_id);
    run_relation_case<float, double, double>("f32_f64_f64", stream, device_id);
    run_relation_case<double, float, double>("f64_f32_f64", stream, device_id);
    run_relation_case<double, double, double>("f64_f64_f64", stream, device_id);
    run_direct_f32("old_project_source", baseline, stream, device_id);
    cuda_check(cudaStreamDestroy(stream));
    dlclose(library);
}
