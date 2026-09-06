#include <Cellerator/compute/operation/relation_update.hh>
#include "../../tests/relation_update_spine_v1/reference_math.hh"
#include <cuda_profiler_api.h>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <Cellerator/compute/architecture/providers/nvidia/sm70/contract/contract_portfolio_v1.cuh>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <string>
namespace ce=cellerator::compute::relation;
namespace ex=cellerator::execution;
namespace ref=ru1_reference;
namespace {
std::uint64_t checks=0;
void require(bool ok,const char* message) {
    ++checks;if(!ok){std::cerr<<message<<'\n';std::exit(1);}
}
void gpu(cudaError_t code) {
    if(code!=cudaSuccess){std::cerr<<cudaGetErrorString(code)<<'\n';std::exit(1);}
}
void core(ce::status code) { if(!code){std::cerr<<code.message<<'\n';std::exit(1);} }
struct stream_owner {
    cudaStream_t value{};
    stream_owner(){gpu(cudaStreamCreateWithFlags(&value,cudaStreamNonBlocking));}
    ~stream_owner(){cudaStreamDestroy(value);}
};
template<class T> struct device_buffer {
    T* data=nullptr;std::size_t count=0;
    explicit device_buffer(std::size_t n):count(n){if(n)gpu(cudaMalloc(&data,n*sizeof(T)));}
    ~device_buffer(){if(data)cudaFree(data);}
    device_buffer(const device_buffer&)=delete;
    void upload(const std::vector<T>& values,cudaStream_t stream){require(values.size()==count,"upload shape");if(count)gpu(cudaMemcpyAsync(data,values.data(),count*sizeof(T),cudaMemcpyHostToDevice,stream));}
    std::vector<T> download(cudaStream_t stream){std::vector<T> result(count);if(count)gpu(cudaMemcpyAsync(result.data(),data,count*sizeof(T),cudaMemcpyDeviceToHost,stream));gpu(cudaStreamSynchronize(stream));return result;}
};
struct pair_owner {
    ce::prepared_relation_pair* value=nullptr;
    ~pair_owner(){ce::destroy(value);}
};
ce::axis_descriptor axis(unsigned id,unsigned extent) {
    ce::axis_descriptor a{};a.extent=extent;
    a.identity.header={ex::biological_abi_version,ex::serialized_record_kind::persistent_axis_identity,sizeof(a.identity)};
    a.identity.domain={id,0xf000000000000001ull};a.identity.order={id,0xe000000000000001ull};
    a.identity.geometry={id,0xd000000000000001ull};a.identity.partition={id,0xc000000000000001ull};return a;
}
using clock_type = std::chrono::steady_clock;
double milliseconds(clock_type::time_point start) {
    return std::chrono::duration<double, std::milli>(clock_type::now() - start).count();
}
struct options {
    std::string fixture = "dense", route = "sparse", profile = "half", operand_reuse = "refresh";
    unsigned modules = 16, horizon = 16, repeats = 5, warmup = 1, width = 16;
    bool diagnostic = false, profile_one = false;
};
options parse(int argc, char** argv) {
    options o;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--diagnostic") { o.diagnostic = true; continue; }
        if (key == "--profile-one") { o.profile_one = true; continue; }
        require(i + 1 < argc, "option requires value"); const std::string value = argv[++i];
        if (key == "--fixture") o.fixture = value;
        else if (key == "--route") o.route = value;
        else if (key == "--profile") o.profile = value;
        else if (key == "--operand-reuse") o.operand_reuse = value;
        else {
            require(!value.empty() && value.find_first_not_of("0123456789") == std::string::npos, "invalid positive integer");
            const auto number = std::stoull(value); require(number <= 65536, "option too large");
            if (key == "--modules") o.modules = number;
            else if (key == "--horizon") o.horizon = number;
            else if (key == "--repeats") o.repeats = number;
            else if (key == "--warmup") o.warmup = number;
            else if (key == "--width") o.width = number;
            else require(false, "unknown option");
        }
    }
    require(!o.profile_one || o.repeats == 1, "profile-one requires repeats1 for isolated diagnostic capture");
    require(o.modules && o.modules <= 256 && o.horizon && o.horizon <= 1024 && o.repeats && o.repeats <= 100, "bounded harness size exceeded");
    require(o.fixture == "dense" || o.fixture == "irregular" || o.fixture == "mixed", "unknown fixture");
    require(o.route == "sparse" || o.route == "hybrid", "unknown route");
    require(o.profile == "half" || o.profile == "full_f32", "unknown arithmetic profile");
    require(o.width == 1 || o.width == 16, "only N1/N16");
    require(o.width != 1 || o.route == "sparse", "N1 is read-only baseline");
    require(o.operand_reuse == "refresh" || o.operand_reuse == "fixed", "unknown operand reuse");
    return o;
}
ref::support make_fixture(const options& o) {
    ref::support graph{16 * o.modules + 3, 16 * o.modules + 5, {}};
    for (unsigned d = 0; d < graph.destinations; ++d) {
        if (o.fixture != "irregular" && d < 16 * o.modules) {
            const unsigned first = d / 16 * 16;
            for (unsigned s = first + 16; s > first; --s) graph.edges.push_back({d, s - 1});
        } else if (o.fixture == "irregular" || o.fixture == "mixed") {
            if (d % 4) {
                graph.edges.push_back({d, (d * 7 + 5) % graph.sources});
                graph.edges.push_back({d, (d * 7) % graph.sources});
            }
        }
    }
    return graph;
}
std::string fixture_fingerprint(const ref::support& graph, const std::vector<float>& x, const std::vector<float>& cot) {
    // FNV1a of canonical integers and f32 bits is an identity, not authentication.
    std::uint64_t hash = 14695981039346656037ull;
    auto word = [&](std::uint32_t value) { for (unsigned b = 0; b < 4; ++b) { hash ^= (value >> (8 * b)) & 255; hash *= 1099511628211ull; } };
    word(graph.sources); word(graph.destinations);
    for (auto edge : graph.edges) { word(edge.destination); word(edge.source); }
    for (const auto* values : {&x, &cot}) for (float value : *values) { std::uint32_t bits; std::memcpy(&bits, &value, 4); word(bits); }
    std::ostringstream out; out << std::hex << hash; return out.str();
}
struct event_owner {
    cudaEvent_t value{};
    event_owner() { gpu(cudaEventCreate(&value)); }
    ~event_owner() { cudaEventDestroy(value); }
};
void run(const options& o) {
    const auto graph = make_fixture(o);
    std::vector<unsigned> offsets(graph.destinations + 1), sources;
    for (auto edge : graph.edges) { ++offsets[edge.destination + 1]; sources.push_back(edge.source); }
    std::partial_sum(offsets.begin(), offsets.end(), offsets.begin());
    std::vector<float> hx(graph.sources * o.width), hc(graph.destinations * o.width);
    for (unsigned i = 0; i < hx.size(); ++i) hx[i] = float(int((i * 13 + 7) % 61) - 30) / 37;
    for (unsigned i = 0; i < hc.size(); ++i) hc[i] = float(int((i * 17 + 11) % 53) - 26) / 41;
    std::vector<std::uint16_t> initial(graph.edges.size());
    for (unsigned i = 0; i < initial.size(); ++i) initial[i] = ref::half_bits(float(int(i % 31) - 15) / 64);
    const auto fingerprint = fixture_fingerprint(graph, hx, hc);
    const auto expected_gradient = ref::edge_gradient(graph, hx, hc, o.width, o.profile == "half");
    stream_owner stream, consumer;
    device_buffer<float> x(hx.size()), cot(hc.size()), y(hc.size()), dx(hx.size()), gradient(initial.size());
    device_buffer<std::uint16_t> weights(initial.size()), observed(initial.size());
    const auto transfer_start = clock_type::now();
    x.upload(hx, stream.value); cot.upload(hc, stream.value); weights.upload(initial, stream.value);
    gpu(cudaStreamSynchronize(stream.value)); const double h2d_ms = milliseconds(transfer_start);
    ce::relation_calculus_descriptor semantic{};
    auto& topology = semantic.forward.topology;
    topology = {{71, 0xf000000000000003ull}, {3}, axis(10, graph.sources), axis(20, graph.destinations), {31, 0xb000000000000001ull}, graph.edges.size()};
    semantic.forward.dense_width = o.width; semantic.transpose = semantic.forward; semantic.transpose.direction = ce::orientation::transpose;
    semantic.gradient = o.profile == "half" ? ce::gradient_arithmetic::round_operands_f16_rne : ce::gradient_arithmetic::full_f32;
    pair_owner pair;
    const auto prepare_start = clock_type::now();
    core(ce::prepare_relation_pair(semantic.forward, semantic.transpose, {offsets.data(), offsets.size(), sources.data(), sources.size()}, {0, 128ull << 20}, stream.value, &pair.value));
    gpu(cudaStreamSynchronize(stream.value)); const double topology_ms = milliseconds(prepare_start);
    const auto gradient_prepare_start = clock_type::now();
    if (o.width == 16) core(ce::prepare_relation_gradient(*pair.value, semantic, {o.route == "hybrid" ? ce::gradient_route::force_hybrid : ce::gradient_route::force_sparse, 128ull << 20}, stream.value));
    gpu(cudaStreamSynchronize(stream.value)); const double gradient_prepare_ms = milliseconds(gradient_prepare_start);
    ce::edge_layout_view layout{};
    if (o.width == 16) core(ce::inspect_edge_layout(*pair.value, &layout));
    const ce::device_state_view input{x.data, x.count, topology.source, 0}, cotangent{cot.data, cot.count, topology.destination, 0};
    const ce::device_result_view output{y.data, y.count, topology.destination, 0}, transpose_output{dx.data, dx.count, topology.source, 0};
    const ce::edge_plane_view edge_output{gradient.data, gradient.count, topology.identity, topology.epoch, layout.order, 0};
    event_owner start, stop;
    // Optional event-rich diagnostics are separately labelled: they perturb the
    // schedule, whereas ordinary timing inserts only two lifetime boundary events.
    std::vector<event_owner> phase_events(o.diagnostic ? 4 * o.horizon : 0);
    std::uint64_t generation = 0, version = 0;
    for (unsigned sample = 0; sample < o.warmup + o.repeats; ++sample) {
        const auto reset_start = clock_type::now();
        core(ce::publish_values(*pair.value, {weights.data, weights.count, topology.identity, topology.epoch, topology.logical_edge_order, {++generation}, 0}, stream.value));
        gpu(cudaStreamSynchronize(stream.value)); const double reset_ms = milliseconds(reset_start);
        ce::relation_update_report before{}, after{}; core(ce::inspect_updates(*pair.value, &before));
        const bool profiling = o.profile_one && sample == o.warmup;
        if (profiling) gpu(cudaProfilerStart());
        const auto wall_start = clock_type::now(); gpu(cudaEventRecord(start.value, stream.value));
        const auto fixed_version = ++version;
        for (unsigned step = 0; step < o.horizon; ++step) {
            if (o.diagnostic) gpu(cudaEventRecord(phase_events[step * 4].value, stream.value));
            core(ce::enqueue(*pair.value, semantic.forward, input, output, {generation}, stream.value));
            if (o.width == 16) core(ce::enqueue(*pair.value, semantic.transpose, cotangent, transpose_output, {generation}, stream.value));
            if (o.diagnostic) gpu(cudaEventRecord(phase_events[step * 4 + 1].value, stream.value));
            if (o.width == 16) {
                ce::gradient_stamp stamp{};
                const auto input_version = o.operand_reuse == "fixed" ? fixed_version : ++version;
                core(ce::enqueue_edge_gradient(*pair.value, semantic, input, cotangent, {101, input_version}, {202, input_version}, {generation}, edge_output, &stamp, stream.value));
                if (o.diagnostic) gpu(cudaEventRecord(phase_events[step * 4 + 2].value, stream.value));
                core(ce::enqueue_value_update(*pair.value, {ce::value_update_kind::gradient_step, edge_output, {generation}, {generation + 1}, 1.0f / 1024, stamp}, stream.value));
                ++generation;
            }
            if (o.diagnostic) gpu(cudaEventRecord(phase_events[step * 4 + 3].value, stream.value));
        }
        gpu(cudaEventRecord(stop.value, stream.value)); gpu(cudaEventSynchronize(stop.value));
        const double resident_wall_ms = milliseconds(wall_start);
        if (profiling) gpu(cudaProfilerStop());
        float resident_gpu_ms = 0; gpu(cudaEventElapsedTime(&resident_gpu_ms, start.value, stop.value));
        core(ce::inspect_updates(*pair.value, &after));
        double dense_ms = 0, gradient_ms = 0, update_ms = 0;
        if (o.diagnostic) for (unsigned step = 0; step < o.horizon; ++step) {
            float duration = 0;
            gpu(cudaEventElapsedTime(&duration, phase_events[step * 4].value, phase_events[step * 4 + 1].value)); dense_ms += duration;
            if (o.width == 16) {
                gpu(cudaEventElapsedTime(&duration, phase_events[step * 4 + 1].value, phase_events[step * 4 + 2].value)); gradient_ms += duration;
                gpu(cudaEventElapsedTime(&duration, phase_events[step * 4 + 2].value, phase_events[step * 4 + 3].value)); update_ms += duration;
            }
        }
        const auto observation_start = clock_type::now();
        std::vector<std::uint16_t> result;
        if (o.width == 16) {
            ce::value_read_lease lease{}; core(ce::begin_value_read(*pair.value, {generation}, consumer.value, &lease));
            gpu(cudaMemcpyAsync(observed.data, lease.physical_f16_values, observed.count * 2, cudaMemcpyDeviceToDevice, consumer.value));
            core(ce::end_value_read(*pair.value, lease, consumer.value)); result = observed.download(consumer.value);
        }
        const double observation_ms = milliseconds(observation_start);
        double worst_gradient = 0;
        if (o.width == 16) {
            const auto actual = gradient.download(stream.value);
            for (unsigned logical = 0; logical < initial.size(); ++logical) {
                const auto physical = layout.logical_to_physical[logical];
                worst_gradient = std::max(worst_gradient, std::abs(actual[physical] - expected_gradient[logical]));
                require(std::abs(actual[physical] - expected_gradient[logical]) <= 2e-5 * (1 + std::abs(expected_gradient[logical])), "independent gradient mismatch");
                auto bits = initial[logical];
                for (unsigned step = 0; step < o.horizon; ++step) bits = ref::gradient_step(bits, actual[physical], 1.0f / 1024);
                require(result[physical] == bits, "reset or resident physical update mismatch");
            }
            require(after.physical_updates - before.physical_updates == o.horizon, "update attribution missing");
            if (o.route == "hybrid") require(after.wmma_launches - before.wmma_launches == o.horizon, "forced WMMA missing");
            if (o.route == "hybrid" && o.fixture == "mixed") require(after.residual_launches > before.residual_launches, "mixed residual missing");
            if (o.route == "sparse") require(after.sparse_launches - before.sparse_launches == o.horizon, "forced sparse missing");
        } else {
            const auto actual = y.download(stream.value);
            std::vector<double> w(initial.size()); for (unsigned i = 0; i < w.size(); ++i) w[i] = ref::half_value(initial[i]);
            const auto expected = ref::forward(graph, w, hx, 1);
            for (unsigned i = 0; i < actual.size(); ++i) require(std::abs(actual[i] - expected[i]) < 2e-5 * (1 + std::abs(expected[i])), "N1 baseline mismatch");
        }
        require(after.relation.topology_preparations == 1, "resident re-preparation");
        if (sample < o.warmup) continue;
        std::cout << std::setprecision(10) << "{\"fixture\":\"" << o.fixture << "\",\"fixture_fnv1a64\":\"" << fingerprint
          << "\",\"seed\":\"formula-v1-13-17-61-53\",\"route\":\"" << o.route << "\",\"profile\":\"" << o.profile
          << "\",\"operand_reuse\":\"" << o.operand_reuse << "\",\"width\":" << o.width << ",\"modules\":" << o.modules
          << ",\"sources\":" << graph.sources << ",\"destinations\":" << graph.destinations << ",\"edges\":" << graph.edges.size()
          << ",\"horizon\":" << o.horizon << ",\"sample\":" << sample - o.warmup << ",\"warmup\":" << o.warmup
          << ",\"repeats\":" << o.repeats << ",\"diagnostic\":" << (o.diagnostic ? "true" : "false")
          << ",\"profiled\":" << (profiling ? "true" : "false") << ",\"reset\":\"logical_f16_upload_outside_timing\""
          << ",\"topology_prepare_ms\":" << topology_ms << ",\"gradient_prepare_ms\":" << gradient_prepare_ms << ",\"h2d_ms\":" << h2d_ms
          << ",\"reset_ms\":" << reset_ms << ",\"resident_gpu_ms\":" << resident_gpu_ms << ",\"resident_wall_ms\":" << resident_wall_ms
          << ",\"observation_d2d_d2h_ms\":" << observation_ms << ",\"persistent_bytes\":" << after.persistent_bytes << ",\"scratch_bytes\":" << after.scratch_bytes
          << ",\"forward_launches\":" << after.relation.accepted_forward_launches - before.relation.accepted_forward_launches
          << ",\"transpose_launches\":" << after.relation.accepted_transpose_launches - before.relation.accepted_transpose_launches
          << ",\"gradient_launches\":" << after.gradient_launches - before.gradient_launches << ",\"updates\":" << after.physical_updates - before.physical_updates
          << ",\"ready_records\":" << after.ready_records - before.ready_records << ",\"wmma_launches\":" << after.wmma_launches - before.wmma_launches
          << ",\"residual_launches\":" << after.residual_launches - before.residual_launches << ",\"sparse_launches\":" << after.sparse_launches - before.sparse_launches
          << ",\"operand_pack_refreshes\":" << after.operand_pack_refreshes - before.operand_pack_refreshes
          << ",\"gradient_max_abs_error\":" << worst_gradient << ",\"correct\":true";
        if (o.diagnostic) std::cout << ",\"dense_forward_transpose_ms\":" << dense_ms << ",\"gradient_including_pack_wmma_extract_residual_ms\":" << gradient_ms << ",\"update_including_publication_ms\":" << update_ms;
        std::cout << "}\n";
    }
    core(ce::close_relation_pair(&pair.value));
}
}
int main(int argc, char** argv) {
    try {
        const auto o = parse(argc, argv); gpu(cudaSetDevice(0)); cudaDeviceProp device{}; gpu(cudaGetDeviceProperties(&device, 0));
        require(device.major == 7 && device.minor == 0, "actual sm70 required; missing GPU is failure");
        run(o);
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
