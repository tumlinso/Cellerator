#include <Cellerator/compute/operation/device_elementwise.hh>
#include <Cellerator/compute/operation/prepared_relation.hh>
#include <Cellerator/compute/candidate/sparse/project.hh>

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

namespace ce = cellerator::compute::relation;
namespace ex = cellerator::execution;

static void gpu_impl(cudaError_t result, int line) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA call failed at fp64_qualification.cu:" << line
                  << ": " << cudaGetErrorString(result) << '\n';
        std::abort();
    }
}
static void check_impl(ce::status result, int line) {
    if (!result) {
        std::cerr << "native call failed at fp64_qualification.cu:" << line << ": "
                  << result.message << " (status " << unsigned(result.code) << ")\n";
        std::abort();
    }
}
#define gpu(expression) gpu_impl((expression), __LINE__)
#define check(expression) check_impl((expression), __LINE__)
template<class T> static T* device(std::size_t count) {
    T* pointer = nullptr;
    if (count) gpu(cudaMalloc(reinterpret_cast<void**>(&pointer), count * sizeof(T)));
    return pointer;
}
template<class T> static void upload(T* destination, const std::vector<T>& source,
                                     cudaStream_t stream) {
    if (!source.empty()) {
        const auto bytes = source.size() * sizeof(T);
        const auto result = cudaMemcpyAsync(destination, source.data(), bytes,
                                            cudaMemcpyHostToDevice, stream);
        if (result != cudaSuccess) {
            std::cerr << "upload failed: type=" << (std::is_same_v<T, float> ? "f32" : "f64")
                      << " bytes=" << bytes
                      << " host=" << static_cast<const void*>(source.data())
                      << " device=" << static_cast<const void*>(destination)
                      << " stream=" << static_cast<void*>(stream) << ": "
                      << cudaGetErrorString(result) << '\n';
            std::abort();
        }
    }
}
template<class T> static std::string dtype_name() {
    return std::is_same_v<T, float> ? "f32" : "f64";
}
template<class T> static ex::numeric_type dtype() {
    return std::is_same_v<T, float> ? ex::numeric_type::f32 : ex::numeric_type::f64;
}

static ce::axis_descriptor axis(std::uint32_t base, std::uint32_t extent) {
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

static void prepared_duplicate_rejection() {
    const std::vector<std::uint32_t> offsets{0, 2};
    const std::vector<std::uint32_t> indices{1, 1};
    ce::operation_descriptor forward{};
    forward.topology.identity = {0x901u, 0xa01u}; forward.topology.epoch = {1};
    forward.topology.source = axis(0x910u, 3); forward.topology.destination = axis(0x920u, 1);
    forward.topology.logical_edge_order = {0xb01u, 0xc01u}; forward.topology.edge_count = 2;
    auto transpose = forward; transpose.direction = ce::orientation::transpose;
    cudaStream_t stream{}; gpu(cudaStreamCreate(&stream));
    ce::prepared_relation_pair* pair = nullptr;
    const auto result = ce::prepare_relation_pair(forward, transpose,
        {offsets.data(), offsets.size(), indices.data(), indices.size()},
        {0, 1u << 20, false}, stream, &pair);
    if (result.code != ce::status_code::unsupported_semantics || pair != nullptr) {
        std::cerr << "prepared topology duplicate endpoints were not explicitly rejected\n";
        std::abort();
    }
    gpu(cudaStreamDestroy(stream));
}

static constexpr std::uint32_t rows = 5;
static constexpr std::uint32_t cols = 7;
static const std::vector<std::uint32_t> row_offsets{0, 3, 3, 6, 7, 9};
static const std::vector<std::uint32_t> source_indices{0, 3, 4, 1, 6, 5, 2, 5, 6};
static const std::vector<std::uint32_t> widths{1, 3, 15, 16, 17, 33, 65, 257};

template<class W, class X, class Y>
static void run_case(std::uint32_t width, bool dump) {
    ce::operation_descriptor forward{};
    forward.topology.identity = {0x101u, 0x201u};
    forward.topology.epoch = {7};
    forward.topology.source = axis(0x110u, cols);
    forward.topology.destination = axis(0x120u, rows);
    forward.topology.logical_edge_order = {0x301u, 0x401u};
    forward.topology.edge_count = source_indices.size();
    forward.dense_width = width;
    forward.arithmetic.relation_storage = dtype<W>();
    forward.arithmetic.input_storage = dtype<X>();
    forward.arithmetic.multiply = dtype<Y>();
    forward.arithmetic.accumulation = dtype<Y>();
    forward.arithmetic.output_storage = dtype<Y>();
    forward.arithmetic.permit_fma = true;
    forward.arithmetic.permit_reassociation = true;
    forward.update = ce::output_update::affine_accumulate;
    // 1 + 2^-25 is representable in binary64 and rounds to exactly 1 in f32.
    forward.input_scale = 1.0000000298023223876953125;
    forward.destination_scale = -1.0;
    auto transpose = forward;
    transpose.direction = ce::orientation::transpose;

    if constexpr (std::is_same_v<Y, double>) {
        auto unsupported = forward;
        unsupported.arithmetic.accumulation = ex::numeric_type::f32;
        auto unsupported_transpose = unsupported;
        unsupported_transpose.direction = ce::orientation::transpose;
        ce::prepared_relation_pair* rejected_pair = nullptr;
        const auto rejected = ce::prepare_relation_pair(unsupported, unsupported_transpose,
            {row_offsets.data(), row_offsets.size(), source_indices.data(), source_indices.size()},
            {0, 1u << 24, false}, nullptr, &rejected_pair);
        if (rejected.code != ce::status_code::unsupported_numeric_policy || rejected_pair != nullptr) {
            std::cerr << "unsupported numeric tuple was not explicitly rejected before preparation\n";
            std::abort();
        }
    }

    cudaStream_t stream{};
    gpu(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    ce::prepared_relation_pair* pair = nullptr;
    const auto prep = ce::prepare_relation_pair(forward, transpose,
        {row_offsets.data(), row_offsets.size(), source_indices.data(), source_indices.size()},
        {0, 1u << 24, false}, stream, &pair);
    check(prep);
    ce::prepared_relation_pair* sibling = nullptr;
    check(ce::create_relation_instance(*pair, stream, &sibling));
    ce::preparation_report owner_report{}, sibling_report{};
    check(ce::inspect(*pair, &owner_report));
    check(ce::inspect(*sibling, &sibling_report));
    if (owner_report.structural_preparation_id == 0
        || owner_report.structural_preparation_id != sibling_report.structural_preparation_id
        || owner_report.shared_structural_bytes != sibling_report.shared_structural_bytes) {
        std::cerr << "prepared instances did not reuse immutable topology\n"; std::abort();
    }
    ce::destroy(sibling);

    std::vector<W> weights(source_indices.size());
    for (std::size_t e = 0; e < weights.size(); ++e) {
        static constexpr double pattern[] = {1.0, -1.0, 0.5, -0.25, 0.125, -0.5, 1.0, 0.75, -0.25};
        weights[e] = static_cast<W>(pattern[e]);
    }
    std::vector<X> features(cols * width);
    for (std::uint32_t i = 0; i < cols; ++i) {
        for (std::uint32_t j = 0; j < width; ++j) {
            double value = (static_cast<int>((i * 7 + j * 3) % 13) - 6) / 8.0;
            if (j == 0) value += (i % 2 ? -1.0 : 1.0) * 1.0e8;
            features[std::size_t(i) * width + j] = static_cast<X>(value);
        }
    }
    W* d_weights = device<W>(weights.size());
    X* d_features = device<X>(features.size());
    upload(d_weights, weights, stream);
    upload(d_features, features, stream);
    Y* d_result = device<Y>(std::size_t(std::max(rows, cols)) * width);
    const Y sentinel = static_cast<Y>(-917.25);

    for (std::uint64_t generation = 1; generation <= 2; ++generation) {
        if (generation == 2) {
            for (auto& value : weights) value = static_cast<W>(value * W(0.75));
            upload(d_weights, weights, stream);
        }
        if constexpr (std::is_same_v<W, float>) {
            if (generation == 1) {
                ce::device_f64_values_binding wrong{reinterpret_cast<const double*>(d_weights),
                    weights.size(), forward.topology.identity, forward.topology.epoch,
                    forward.topology.logical_edge_order, {1}, 0};
                const auto rejected = ce::publish_f64_values(*pair, wrong, stream);
                if (rejected.code != ce::status_code::unsupported_numeric_policy) {
                    std::cerr << "wrong FP64 publication dtype was not rejected\n"; std::abort();
                }
            }
            ce::device_f32_values_binding binding{d_weights, weights.size(),
                forward.topology.identity, forward.topology.epoch,
                forward.topology.logical_edge_order, {generation}, 0};
            if (generation == 1) {
                gpu(cudaStreamSynchronize(stream));
                gpu(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
                const auto captured = ce::publish_f32_values(*pair, binding, stream);
                cudaGraph_t graph{}; gpu(cudaStreamEndCapture(stream, &graph));
                if (graph) gpu(cudaGraphDestroy(graph));
                if (captured.code != ce::status_code::unsupported_semantics) {
                    std::cerr << "FP32 value publication during capture was not rejected\n"; std::abort();
                }
            }
            check(ce::publish_f32_values(*pair, binding, stream));
        } else {
            if (generation == 1) {
                ce::device_f32_values_binding wrong{reinterpret_cast<const float*>(d_weights),
                    weights.size(), forward.topology.identity, forward.topology.epoch,
                    forward.topology.logical_edge_order, {1}, 0};
                const auto rejected = ce::publish_f32_values(*pair, wrong, stream);
                if (rejected.code != ce::status_code::unsupported_numeric_policy) {
                    std::cerr << "wrong FP32 publication dtype was not rejected\n"; std::abort();
                }
            }
            ce::device_f64_values_binding binding{d_weights, weights.size(),
                forward.topology.identity, forward.topology.epoch,
                forward.topology.logical_edge_order, {generation}, 0};
            if (generation == 1) {
                gpu(cudaStreamSynchronize(stream));
                gpu(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
                const auto captured = ce::publish_f64_values(*pair, binding, stream);
                cudaGraph_t graph{}; gpu(cudaStreamEndCapture(stream, &graph));
                if (graph) gpu(cudaGraphDestroy(graph));
                if (captured.code != ce::status_code::unsupported_semantics) {
                    std::cerr << "FP64 value publication during capture was not rejected\n"; std::abort();
                }
            }
            check(ce::publish_f64_values(*pair, binding, stream));
        }

        const auto report_before = [&] { ce::preparation_report r{}; check(ce::inspect(*pair, &r)); return r; }();
        if (report_before.value_refreshes != generation
            || report_before.latest_enqueued_generation.value != generation) {
            std::cerr << "value publication counter/generation mismatch\n"; std::abort();
        }
        for (auto direction : {ce::orientation::forward, ce::orientation::transpose}) {
            auto op = direction == ce::orientation::forward ? forward : transpose;
            const auto input_extent = direction == ce::orientation::forward ? cols : rows;
            const auto output_extent = direction == ce::orientation::forward ? rows : cols;
            const auto& in_axis = direction == ce::orientation::forward
                ? forward.topology.source : forward.topology.destination;
            const auto& out_axis = direction == ce::orientation::forward
                ? forward.topology.destination : forward.topology.source;
            std::vector<Y> initial(output_extent * width, sentinel);
            upload(d_result, initial, stream);
            ce::device_state_view input{d_features,
                std::uint64_t(input_extent) * width, in_axis, 0, dtype<X>()};
            ce::device_result_view output{d_result,
                std::uint64_t(output_extent) * width, out_axis, 0, dtype<Y>()};
            if (generation == 1 && direction == ce::orientation::forward) {
                const auto accepted_before = [&] { ce::preparation_report r{}; check(ce::inspect(*pair, &r)); return r.accepted_forward_launches; }();
                auto rejected = ce::enqueue(*pair, op, input, output, {0}, stream);
                if (rejected.code != ce::status_code::stale_generation) {
                    std::cerr << "stale generation was not rejected\n"; std::abort();
                }
                auto short_output = output;
                --short_output.count;
                rejected = ce::enqueue(*pair, op, input, short_output, {generation}, stream);
                if (rejected.code != ce::status_code::insufficient_capacity) {
                    std::cerr << "short output capacity was not rejected\n"; std::abort();
                }
                auto wrong_dtype = input;
                wrong_dtype.dtype = wrong_dtype.dtype == ex::numeric_type::f32
                    ? ex::numeric_type::f64 : ex::numeric_type::f32;
                rejected = ce::enqueue(*pair, op, wrong_dtype, output, {generation}, stream);
                if (rejected.code != ce::status_code::unsupported_numeric_policy) {
                    std::cerr << "feature dtype mismatch did not fail as unsupported numeric policy\n"; std::abort();
                }
                auto misaligned = input;
                misaligned.data = reinterpret_cast<const void*>(
                    reinterpret_cast<std::uintptr_t>(input.data) + 1);
                rejected = ce::enqueue(*pair, op, misaligned, output, {generation}, stream);
                if (rejected.code != ce::status_code::insufficient_capacity) {
                    std::cerr << "misaligned feature pointer was not rejected with insufficient_capacity\n"; std::abort();
                }
                cudaStream_t other{}; gpu(cudaStreamCreate(&other));
                rejected = ce::enqueue(*pair, op, input, output, {generation}, other);
                gpu(cudaStreamDestroy(other));
                if (rejected.code != ce::status_code::incompatible_stream) {
                    std::cerr << "stream mismatch was not rejected\n"; std::abort();
                }
                if constexpr (std::is_same_v<X, Y>) {
                    std::vector<X> input_before(features.size());
                    gpu(cudaMemcpy(input_before.data(), d_features, input_before.size() * sizeof(X), cudaMemcpyDeviceToHost));
                    auto alias = output;
                    alias.data = d_features;
                    rejected = ce::enqueue(*pair, op, input, alias, {generation}, stream);
                    if (rejected.code != ce::status_code::invalid_argument) {
                        std::cerr << "prepared input/output overlap was not rejected\n"; std::abort();
                    }
                    std::vector<X> input_after(features.size());
                    gpu(cudaMemcpy(input_after.data(), d_features, input_after.size() * sizeof(X), cudaMemcpyDeviceToHost));
                    if (input_before != input_after) {
                        std::cerr << "rejected prepared alias modified source storage\n"; std::abort();
                    }
                }
                gpu(cudaStreamSynchronize(stream));
                std::vector<Y> preserved(output_extent * width);
                gpu(cudaMemcpy(preserved.data(), d_result, preserved.size() * sizeof(Y), cudaMemcpyDeviceToHost));
                if (!std::all_of(preserved.begin(), preserved.end(), [=](Y value) { return value == sentinel; })) {
                    std::cerr << "admission failure modified caller output\n"; std::abort();
                }
                const auto accepted_after = [&] { ce::preparation_report r{}; check(ce::inspect(*pair, &r)); return r.accepted_forward_launches; }();
                if (accepted_before != accepted_after) {
                    std::cerr << "admission failure changed launch counters\n"; std::abort();
                }
            }
            check(ce::enqueue(*pair, op, input, output, {generation}, stream));
            gpu(cudaStreamSynchronize(stream));
            std::vector<Y> actual(output_extent * width);
            gpu(cudaMemcpy(actual.data(), d_result, actual.size() * sizeof(Y), cudaMemcpyDeviceToHost));

            if (dump) {
                std::cout << std::setprecision(17)
                    << "{\"weights\":\"" << dtype_name<W>() << "\",\"features\":\""
                    << dtype_name<X>() << "\",\"output\":\"" << dtype_name<Y>()
                    << "\",\"direction\":\""
                    << (direction == ce::orientation::forward ? "forward" : "transpose")
                    << "\",\"generation\":" << generation << ",\"rows\":" << rows
                    << ",\"cols\":" << cols << ",\"width\":" << width
                    << ",\"input_scale\":" << forward.input_scale
                    << ",\"destination_scale\":" << forward.destination_scale
                    << ",\"initial_output\":" << sentinel
                    << ",\"offsets\":[";
                for (std::size_t i = 0; i < row_offsets.size(); ++i)
                    std::cout << (i ? "," : "") << row_offsets[i];
                std::cout << "],\"indices\":[";
                for (std::size_t i = 0; i < source_indices.size(); ++i)
                    std::cout << (i ? "," : "") << source_indices[i];
                auto values = [&](const auto& xs) {
                    for (std::size_t i = 0; i < xs.size(); ++i)
                        std::cout << (i ? "," : "") << static_cast<double>(xs[i]);
                };
                std::cout << "],\"weight_values\":["; values(weights);
                std::cout << "],\"feature_values\":["; values(features);
                std::cout << "],\"actual\":["; values(actual); std::cout << "]}\n";
            }
            (void)report_before;
        }
    }
    ce::preparation_report report{};
    check(ce::inspect(*pair, &report));
    if (report.topology_preparations != 1 || report.value_refreshes != 2
        || report.latest_enqueued_generation.value != 2) {
        std::cerr << "topology or value generation reuse invariant failed\n";
        std::abort();
    }
    cudaFree(d_weights); cudaFree(d_features); cudaFree(d_result);
    ce::destroy(pair);
    gpu(cudaStreamDestroy(stream));
}

template<class W, class X, class Y>
static void run_type(bool dump) {
    for (auto width : widths) run_case<W, X, Y>(width, dump);
}

template<class W, class X, class Y>
static void shared_csr_duplicate_case(std::uint32_t width, ce::orientation direction) {
    constexpr std::uint32_t nr = 3, nc = 4;
    const std::vector<std::uint32_t> offsets{0, 3, 3, 6};
    const std::vector<std::uint32_t> indices{1, 1, 3, 0, 2, 2};
    const std::vector<double> original{1.0, -1.0, 0.5, -0.25, 0.125, 0.75};
    std::vector<W> weights(original.size());
    std::transform(original.begin(), original.end(), weights.begin(), [](double x) { return static_cast<W>(x); });
    std::vector<std::uint32_t> projected_offsets, projected_indices, value_indices;
    if (direction == ce::orientation::forward) {
        projected_offsets = offsets; projected_indices = indices;
        value_indices.resize(indices.size());
        for (std::uint32_t i = 0; i < value_indices.size(); ++i) value_indices[i] = i;
    } else {
        std::vector<std::vector<std::pair<std::uint32_t, std::uint32_t>>> by_destination(nc);
        for (std::uint32_t row = 0; row < nr; ++row)
            for (std::uint32_t edge = offsets[row]; edge < offsets[row + 1]; ++edge)
                by_destination[indices[edge]].push_back({row, edge});
        projected_offsets.push_back(0);
        for (const auto& row : by_destination) {
            for (const auto& edge : row) {
                projected_indices.push_back(edge.first);
                value_indices.push_back(edge.second);
            }
            projected_offsets.push_back(static_cast<std::uint32_t>(projected_indices.size()));
        }
    }
    const auto input_extent = direction == ce::orientation::forward ? nc : nr;
    const auto output_extent = direction == ce::orientation::forward ? nr : nc;
    std::vector<X> features(input_extent * width);
    for (std::uint32_t i = 0; i < input_extent; ++i)
        for (std::uint32_t j = 0; j < width; ++j) {
            double value = (int((i * 7 + j * 3) % 13) - 6) / 8.0;
            if (j == 0) value += i % 2 ? -1.0e8 : 1.0e8;
            features[std::size_t(i) * width + j] = static_cast<X>(value);
        }
    W* d_weights = device<W>(weights.size()); X* d_features = device<X>(features.size());
    Y* d_output = device<Y>(std::size_t(output_extent) * width);
    auto* d_offsets = device<std::uint32_t>(projected_offsets.size());
    auto* d_indices = device<std::uint32_t>(projected_indices.size());
    auto* d_value_indices = device<std::uint32_t>(value_indices.size());
    cudaStream_t stream{}; gpu(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    upload(d_weights, weights, stream); upload(d_features, features, stream);
    upload(d_offsets, projected_offsets, stream); upload(d_indices, projected_indices, stream);
    upload(d_value_indices, value_indices, stream);
    cellerator::runtime::execution_context context{0, stream, false};
    cellerator::compute::sparse::project::csr_spmm_fwd<W, X, Y, Y, Y, Y>(
        context, d_offsets, d_indices, d_weights,
        direction == ce::orientation::forward ? nr : nc,
        direction == ce::orientation::forward ? nc : nr, d_features, width, width,
        d_output, width, d_value_indices, Y(1), Y(0));
    gpu(cudaStreamSynchronize(stream));
    std::vector<Y> actual(std::size_t(output_extent) * width);
    gpu(cudaMemcpy(actual.data(), d_output, actual.size() * sizeof(Y), cudaMemcpyDeviceToHost));
    std::cout << std::setprecision(17)
        << "{\"kind\":\"shared_csr\",\"weights\":\"" << dtype_name<W>()
        << "\",\"features\":\"" << dtype_name<X>() << "\",\"output\":\""
        << dtype_name<Y>() << "\",\"direction\":\""
        << (direction == ce::orientation::forward ? "forward" : "transpose")
        << "\",\"rows\":" << nr << ",\"cols\":" << nc << ",\"width\":" << width
        << ",\"offsets\":[";
    auto emit_indices = [](const char* key, const std::vector<std::uint32_t>& values) {
        std::cout << "],\"" << key << "\":[";
        for (std::size_t i = 0; i < values.size(); ++i) std::cout << (i ? "," : "") << values[i];
    };
    for (std::size_t i = 0; i < projected_offsets.size(); ++i)
        std::cout << (i ? "," : "") << projected_offsets[i];
    emit_indices("indices", projected_indices); emit_indices("value_indices", value_indices);
    auto emit_values = [](const char* key, const auto& values) {
        std::cout << "],\"" << key << "\":[";
        for (std::size_t i = 0; i < values.size(); ++i) std::cout << (i ? "," : "") << static_cast<double>(values[i]);
    };
    emit_values("weight_values", weights); emit_values("feature_values", features);
    emit_values("actual", actual); std::cout << "]}\n";
    cudaFree(d_weights); cudaFree(d_features); cudaFree(d_output);
    cudaFree(d_offsets); cudaFree(d_indices); cudaFree(d_value_indices);
    gpu(cudaStreamDestroy(stream));
}

template<class W, class X, class Y>
static void shared_csr_duplicate_type() {
    for (auto width : widths)
        for (auto direction : {ce::orientation::forward, ce::orientation::transpose})
            shared_csr_duplicate_case<W, X, Y>(width, direction);
}

template<class T> static void elementwise_case(int device_id, cudaStream_t stream) {
    constexpr std::size_t n = 19;
    std::vector<T> a(n), b(n), expected(n);
    for (std::size_t i = 0; i < n; ++i) {
        a[i] = T(int(i) - 8) / T(4);
        b[i] = T(int(i % 5) - 2) / T(8);
        expected[i] = a[i] * b[i];
    }
    T* da = nullptr; T* db = nullptr; T* dout = nullptr;
    gpu(cudaMallocManaged(reinterpret_cast<void**>(&da), (n + 1) * sizeof(T)));
    gpu(cudaMallocManaged(reinterpret_cast<void**>(&db), n * sizeof(T)));
    gpu(cudaMallocManaged(reinterpret_cast<void**>(&dout), n * sizeof(T)));
    upload(da, a, stream); upload(db, b, stream);
    cellerator::runtime::execution_context context{device_id, stream, false};
    check(ce::multiply(context, da, db, dout, n));
    gpu(cudaStreamSynchronize(stream));
    std::vector<T> actual(n);
    gpu(cudaMemcpy(actual.data(), dout, n * sizeof(T), cudaMemcpyDeviceToHost));
    for (std::size_t i = 0; i < n; ++i) {
        constexpr T tolerance = std::is_same_v<T, double> ? T(1e-12) : T(2e-6);
        if (std::abs(actual[i] - a[i] * b[i]) > tolerance) {
            std::cerr << "elementwise multiply mismatch\n"; std::abort();
        }
    }
    std::fill_n(dout, n, T(311));
    const auto short_range = ce::multiply(context, da, db, dout, n + 1);
    if (short_range.code != ce::status_code::insufficient_capacity) {
        std::cerr << "elementwise capacity overrun was not rejected\n"; std::abort();
    }
    gpu(cudaStreamSynchronize(stream));
    if (!std::all_of(dout, dout + n, [](T value) { return value == T(311); })) {
        std::cerr << "elementwise capacity failure modified output\n"; std::abort();
    }
    std::fill_n(dout, n, T(419));
    const auto byte_overflow = ce::multiply(context, da, db, dout,
        std::numeric_limits<std::uint64_t>::max());
    if (byte_overflow.code != ce::status_code::insufficient_capacity) {
        std::cerr << "elementwise byte-size overflow was not rejected\n"; std::abort();
    }
    if (!std::all_of(dout, dout + n, [](T value) { return value == T(419); })) {
        std::cerr << "elementwise byte overflow modified output\n"; std::abort();
    }
    check(ce::square(context, da, da, n));
    constexpr double precise_alpha = 1.0000000000000002;
    check(ce::axpby(context, static_cast<T>(precise_alpha), da, T(-1), db, dout, n));
    gpu(cudaStreamSynchronize(stream));
    gpu(cudaMemcpy(actual.data(), dout, n * sizeof(T), cudaMemcpyDeviceToHost));
    for (std::size_t i = 0; i < n; ++i) {
        const T expected = static_cast<T>(precise_alpha) * a[i] * a[i] - b[i];
        constexpr T tolerance = std::is_same_v<T, double> ? T(1e-12) : T(2e-6);
        if (std::abs(actual[i] - expected) > tolerance) {
            std::cerr << "elementwise square/affine mismatch\n"; std::abort();
        }
    }
    const auto partial = ce::multiply(context, da, db, da + 1, n);
    if (partial.code != ce::status_code::invalid_argument) {
        std::cerr << "partial overlap was not rejected\n"; std::abort();
    }
    cudaFree(da); cudaFree(db); cudaFree(dout);
    (void)expected;
}

static void variance_pipeline(bool cancellation_case) {
    constexpr std::uint32_t vr = 4, vc = 3;
    const std::vector<std::uint32_t> offsets{0, 3, 6, 9, 12};
    const std::vector<std::uint32_t> indices{0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2};
    ce::operation_descriptor op{};
    op.topology.identity = {0x501u, 0x601u}; op.topology.epoch = {1};
    op.topology.source = axis(0x510u, vc); op.topology.destination = axis(0x520u, vr);
    op.topology.logical_edge_order = {0x701u, 0x801u}; op.topology.edge_count = indices.size();
    op.dense_width = 1;
    op.arithmetic.relation_storage = ex::numeric_type::f32;
    op.arithmetic.input_storage = ex::numeric_type::f64;
    op.arithmetic.multiply = ex::numeric_type::f64;
    op.arithmetic.accumulation = ex::numeric_type::f64;
    op.arithmetic.output_storage = ex::numeric_type::f64;
    auto transpose = op; transpose.direction = ce::orientation::transpose;
    cudaStream_t stream{}; gpu(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    ce::prepared_relation_pair* pair = nullptr;
    check(ce::prepare_relation_pair(op, transpose,
        {offsets.data(), offsets.size(), indices.data(), indices.size()},
        {0, 1u << 20, false}, stream, &pair));
    std::vector<float> weights(indices.size(), 0.0f);
    for (std::size_t row = 0; row < vr; ++row) {
        weights[row * 3] = 0.25f; weights[row * 3 + 1] = 0.5f; weights[row * 3 + 2] = 0.25f;
    }
    const std::vector<double> features = cancellation_case
        ? std::vector<double>{1.0e8, 1.0e8 + 1.0, 1.0e8 + 2.0}
        : std::vector<double>{1.0, 2.0, 3.0};
    float* d_weights = device<float>(weights.size());
    double* d_features = device<double>(features.size());
    upload(d_weights, weights, stream); upload(d_features, features, stream);
    ce::device_f32_values_binding binding{d_weights, weights.size(), op.topology.identity,
        op.topology.epoch, op.topology.logical_edge_order, {1}, 0};
    check(ce::publish_f32_values(*pair, binding, stream));

    double* d_squared = device<double>(vc);
    double* d_mean = device<double>(vr);
    double* d_second = device<double>(vr);
    double* d_mean_squared = device<double>(vr);
    double* d_variance = device<double>(vr);
    cellerator::runtime::execution_context context{0, stream, false};
    check(ce::square(context, d_features, d_squared, vc));
    ce::device_state_view feature_view{d_features, vc, op.topology.source, 0, ex::numeric_type::f64};
    ce::device_state_view squared_view{d_squared, vc, op.topology.source, 0, ex::numeric_type::f64};
    ce::device_result_view mean_view{d_mean, vr, op.topology.destination, 0, ex::numeric_type::f64};
    ce::device_result_view second_view{d_second, vr, op.topology.destination, 0, ex::numeric_type::f64};
    check(ce::enqueue(*pair, op, feature_view, mean_view, {1}, stream));
    check(ce::enqueue(*pair, op, squared_view, second_view, {1}, stream));
    check(ce::square(context, d_mean, d_mean_squared, vr));
    check(ce::axpby(context, 1.0, d_second, -1.0, d_mean_squared, d_variance, vr));
    gpu(cudaStreamSynchronize(stream));
    std::vector<double> mean(vr), second(vr), variance(vr);
    gpu(cudaMemcpy(mean.data(), d_mean, vr * sizeof(double), cudaMemcpyDeviceToHost));
    gpu(cudaMemcpy(second.data(), d_second, vr * sizeof(double), cudaMemcpyDeviceToHost));
    gpu(cudaMemcpy(variance.data(), d_variance, vr * sizeof(double), cudaMemcpyDeviceToHost));
    std::cout << std::setprecision(17)
        << "{\"kind\":\"variance\",\"regime\":\""
        << (cancellation_case ? "cancellation" : "well_conditioned")
        << "\",\"rows\":" << vr << ",\"cols\":" << vc
        << ",\"offsets\":[";
    for (std::size_t i = 0; i < offsets.size(); ++i) std::cout << (i ? "," : "") << offsets[i];
    std::cout << "],\"indices\":[";
    for (std::size_t i = 0; i < indices.size(); ++i) std::cout << (i ? "," : "") << indices[i];
    std::cout << "],\"weights\":[";
    for (std::size_t i = 0; i < weights.size(); ++i) std::cout << (i ? "," : "") << weights[i];
    auto emit = [](const char* name, const std::vector<double>& values) {
        std::cout << "],\"" << name << "\":[";
        for (std::size_t i = 0; i < values.size(); ++i) std::cout << (i ? "," : "") << values[i];
    };
    emit("features", features); emit("mean", mean); emit("second_moment", second);
    emit("variance_identity", variance); std::cout << "]}\n";
    cudaFree(d_weights); cudaFree(d_features); cudaFree(d_squared); cudaFree(d_mean);
    cudaFree(d_second); cudaFree(d_mean_squared); cudaFree(d_variance);
    ce::destroy(pair); gpu(cudaStreamDestroy(stream));
}

int main(int argc, char** argv) {
    gpu(cudaSetDevice(0));
    const bool dump = argc > 1 && std::string(argv[1]) == "--dump-jsonl";
    prepared_duplicate_rejection();
    run_type<float, float, float>(dump);
    run_type<float, double, double>(dump);
    run_type<double, float, double>(dump);
    run_type<double, double, double>(dump);
    if (dump) {
        shared_csr_duplicate_type<float, float, float>();
        shared_csr_duplicate_type<float, double, double>();
        shared_csr_duplicate_type<double, float, double>();
        shared_csr_duplicate_type<double, double, double>();
    }
    cudaStream_t stream{}; gpu(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    elementwise_case<float>(0, stream); elementwise_case<double>(0, stream);
    gpu(cudaStreamDestroy(stream));
    if (dump) { variance_pipeline(false); variance_pipeline(true); }
    if (!dump) std::cout << "FP64 prepared relation and elementwise device checks passed\n";
}
