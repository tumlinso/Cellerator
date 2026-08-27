#include <bench/ce_live/runtime_fixture/quantitative_fixture.hh>

#include <Cellerator/compute/math/operation_core/builtin_catalog.hh>
#include <Cellerator/compute/math/operation_core/feature_major_small_n_candidate.hh>
#include <Cellerator/compute/math/physical_feature_major.hh>
#include <Cellerator/execution/program.hh>

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace ce_live = cellerator::ce_live;
namespace cm = cellerator::compute::math;
namespace core = cellerator::compute::math::core;
namespace execution = cellerator::execution;
namespace planner = cellerator::planner;
namespace runtime = cellerator::runtime;

namespace {

void require(bool condition, const char *message) {
    if (condition) return;
    std::cerr << "quantitative_forward_test: " << message << '\n';
    std::exit(1);
}

void require(execution::executable_program_status status,
    const char *message) {
    if (status) return;
    std::cerr << "quantitative_forward_test: " << message
              << " (code=" << static_cast<unsigned>(status.code)
              << ", detail=" << status.message << ")\n";
    std::exit(1);
}

void require_cuda(cudaError_t status, const char *message) {
    if (status == cudaSuccess) return;
    std::cerr << "quantitative_forward_test: " << message << ": "
              << cudaGetErrorString(status) << '\n';
    std::exit(1);
}

template<typename T>
struct device_buffer {
    T *data = nullptr;
    std::size_t size = 0u;

    explicit device_buffer(std::size_t count) : size(count) {
        if (count != 0u)
            require_cuda(cudaMalloc(reinterpret_cast<void **>(&data),
                count * sizeof(T)), "cudaMalloc");
    }
    device_buffer(const device_buffer &) = delete;
    device_buffer &operator=(const device_buffer &) = delete;
    ~device_buffer() { if (data != nullptr) (void)cudaFree(data); }
};

template<typename T>
void upload(device_buffer<T> &target, const std::vector<T> &source) {
    require(target.size >= source.size(), "device upload capacity");
    if (!source.empty())
        require_cuda(cudaMemcpy(target.data, source.data(),
            source.size() * sizeof(T), cudaMemcpyHostToDevice), "device upload");
}

struct fixture_data {
    std::uint32_t rows = 0u;
    std::uint32_t features = 0u;
    std::uint64_t nnz = 0u;
    std::vector<std::uint64_t> offsets;
    std::vector<std::uint32_t> indices;
    std::vector<float> generation_1;
    std::vector<float> generation_2;
};

template<typename T>
void read_exact(std::ifstream &stream, T *data, std::size_t count) {
    stream.read(reinterpret_cast<char *>(data),
        static_cast<std::streamsize>(count * sizeof(T)));
    require(stream.good(), "fixture binary is truncated");
}

fixture_data load_fixture(const char *path) {
    std::ifstream stream(path, std::ios::binary);
    require(stream.good(), "fixture binary is unavailable");
    char magic[8]{};
    read_exact(stream, magic, sizeof(magic));
    require(std::string(magic, sizeof(magic)) == "CELIVE31",
        "fixture binary magic mismatch");
    std::uint32_t version = 0u, reserved = 0u;
    fixture_data fixture{};
    read_exact(stream, &version, 1u);
    read_exact(stream, &fixture.rows, 1u);
    read_exact(stream, &fixture.features, 1u);
    read_exact(stream, &reserved, 1u);
    read_exact(stream, &fixture.nnz, 1u);
    require(version == 1u && reserved == 0u && fixture.rows == 512u
        && fixture.features == 32738u && fixture.nnz == 433808u,
        "fixture dimensions drifted from the pinned manifest");
    fixture.offsets.resize(static_cast<std::size_t>(fixture.rows) + 1u);
    fixture.indices.resize(fixture.nnz);
    fixture.generation_1.resize(fixture.nnz);
    fixture.generation_2.resize(fixture.nnz);
    read_exact(stream, fixture.offsets.data(), fixture.offsets.size());
    read_exact(stream, fixture.indices.data(), fixture.indices.size());
    read_exact(stream, fixture.generation_1.data(), fixture.generation_1.size());
    read_exact(stream, fixture.generation_2.data(), fixture.generation_2.size());
    require(fixture.offsets.front() == 0u
        && fixture.offsets.back() == fixture.nnz,
        "fixture CSR terminals are invalid");
    return fixture;
}

struct feature_major_host {
    std::vector<std::uint32_t> tile_offsets{0u};
    std::vector<std::uint32_t> features;
    std::vector<std::uint32_t> masks;
    std::vector<std::uint32_t> value_offsets{0u};
    std::vector<std::uint32_t> source_positions;
};

feature_major_host build_feature_major(const fixture_data &fixture) {
    feature_major_host output;
    constexpr std::uint32_t tile_width = 32u;
    const std::uint32_t tile_count =
        (fixture.rows + tile_width - 1u) / tile_width;
    for (std::uint32_t tile = 0u; tile < tile_count; ++tile) {
        std::map<std::uint32_t,
            std::vector<std::pair<std::uint32_t, std::uint32_t>>> records;
        const std::uint32_t row_begin = tile * tile_width;
        const std::uint32_t row_end = std::min(
            fixture.rows, row_begin + tile_width);
        for (std::uint32_t row = row_begin; row < row_end; ++row)
            for (std::uint64_t edge = fixture.offsets[row];
                 edge < fixture.offsets[row + 1u]; ++edge)
                records[fixture.indices[edge]].push_back(
                    {row - row_begin, static_cast<std::uint32_t>(edge)});
        for (const auto &record : records) {
            std::uint32_t mask = 0u;
            output.features.push_back(record.first);
            for (const auto &entry : record.second) {
                require((mask & (1u << entry.first)) == 0u,
                    "duplicate feature within one destination row");
                mask |= 1u << entry.first;
                output.source_positions.push_back(entry.second);
            }
            output.masks.push_back(mask);
            output.value_offsets.push_back(
                static_cast<std::uint32_t>(output.source_positions.size()));
        }
        output.tile_offsets.push_back(
            static_cast<std::uint32_t>(output.features.size()));
    }
    require(output.source_positions.size() == fixture.nnz,
        "feature-major conversion lost logical edges");
    return output;
}

execution::program_axis make_axis(execution::axis_identity live,
    execution::domain_id domain, execution::order_id order,
    execution::geometry_id geometry, execution::partition_id partition) {
    execution::persistent_axis_identity persistent{};
    persistent.header = {execution::biological_abi_version,
        execution::serialized_record_kind::persistent_axis_identity,
        sizeof(execution::persistent_axis_identity)};
    persistent.domain = domain;
    persistent.order = order;
    persistent.geometry = geometry;
    persistent.partition = partition;
    return {live, persistent};
}

execution::device_location device_location(int device) {
    return {execution::residency_kind::device, {}, device, 0u};
}

execution::dense_tensor_view dense_matrix(void *pointer,
    execution::axis_identity major, execution::axis_identity minor,
    std::uint64_t rows, std::uint64_t columns, int device) {
    execution::dense_tensor_view view{};
    view.data = pointer;
    view.location = device_location(device);
    view.value_type = execution::numeric_type::f32;
    view.rank = 2u;
    view.axes[0] = major;
    view.axes[1] = minor;
    view.shape[0] = rows;
    view.shape[1] = columns;
    view.stride[0] = static_cast<std::int64_t>(columns);
    view.stride[1] = 1;
    return view;
}

core::numeric_policy numeric_policy() {
    core::numeric_policy value{};
    value.sparse_storage = execution::numeric_type::f16;
    value.dense_storage = execution::numeric_type::f32;
    value.output_storage = execution::numeric_type::f32;
    value.multiply = execution::numeric_type::f32;
    value.accumulation = execution::numeric_type::f32;
    value.scalar = execution::numeric_type::f32;
    value.bias = execution::numeric_type::invalid;
    return value;
}

std::vector<__half> pack_values(const std::vector<float> &source,
    const std::vector<std::uint32_t> &positions) {
    std::vector<__half> result(positions.size());
    for (std::size_t index = 0u; index < positions.size(); ++index)
        result[index] = __float2half(source[positions[index]]);
    return result;
}

void fill_dense(std::vector<float> *dense,
    std::uint32_t features, std::uint32_t width) {
    dense->resize(static_cast<std::size_t>(features) * width);
    ce_live::fill_deterministic_dense_operand(
        dense->data(), features, width);
}

void verify_output(const fixture_data &fixture,
    const std::vector<__half> &csr_values,
    const std::vector<float> &dense, std::uint32_t width,
    const std::vector<float> &actual) {
    require(actual.size() == static_cast<std::size_t>(fixture.rows) * width,
        "output shape mismatch");
    for (std::uint32_t row = 0u; row < fixture.rows; ++row) {
        for (std::uint32_t lane = 0u; lane < width; ++lane) {
            double expected = 0.0;
            for (std::uint64_t edge = fixture.offsets[row];
                 edge < fixture.offsets[row + 1u]; ++edge)
                expected += static_cast<double>(__half2float(csr_values[edge]))
                    * static_cast<double>(dense[
                        static_cast<std::size_t>(fixture.indices[edge])
                            * width + lane]);
            const double tolerance = 5.0e-4
                * std::max(1.0, std::fabs(expected));
            require(std::fabs(static_cast<double>(actual[
                        static_cast<std::size_t>(row) * width + lane])
                    - expected) <= tolerance,
                "CUDA output disagrees with independent CSR referee");
        }
    }
}

double median(std::vector<float> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2u];
}

} // namespace

int main(int argc, char **argv) {
    require(argc == 3, "usage: quantitative_forward_test FIXTURE REPEATS");
    const int repeats = std::stoi(argv[2]);
    require(repeats > 0 && repeats <= 20, "repeat count is invalid");
    const fixture_data fixture = load_fixture(argv[1]);

    execution::identity_registry registry{};
    ce_live::native_quantitative_relation relation{};
    std::vector<float> generation_1 = fixture.generation_1;
    std::vector<float> generation_2 = fixture.generation_2;
    const ce_live::quantitative_fixture_arrays arrays{
        {fixture.offsets.data(), fixture.indices.data(), fixture.rows,
            fixture.features, fixture.nnz},
        generation_1.data(), generation_2.data()};
    const auto identities = ce_live::pbmc3k_quantitative_v1_identities();
    require(ce_live::bind_quantitative_fixture(arrays, identities,
            &registry, {1u, 1u}, &relation)
            == ce_live::quantitative_fixture_status::ok,
        "PBMC3K quantitative relation binding failed");
    require(execution::validate_relation_structure(relation.structure)
            == execution::lifetime_validation_code::ok,
        "PBMC3K relation structure is invalid");

    const auto build_start = std::chrono::steady_clock::now();
    const feature_major_host host_projection = build_feature_major(fixture);
    const auto build_stop = std::chrono::steady_clock::now();
    const double projection_build_ns =
        std::chrono::duration<double, std::nano>(build_stop - build_start).count();
    const auto generation_1_csr = pack_values(
        fixture.generation_1,
        [&] { std::vector<std::uint32_t> identity(fixture.nnz);
            for (std::uint32_t i = 0u; i < identity.size(); ++i) identity[i] = i;
            return identity; }());
    const auto generation_2_csr = pack_values(
        fixture.generation_2,
        [&] { std::vector<std::uint32_t> identity(fixture.nnz);
            for (std::uint32_t i = 0u; i < identity.size(); ++i) identity[i] = i;
            return identity; }());
    const auto pack_start = std::chrono::steady_clock::now();
    const auto generation_1_fmp = pack_values(
        fixture.generation_1, host_projection.source_positions);
    const auto generation_2_fmp = pack_values(
        fixture.generation_2, host_projection.source_positions);
    const auto pack_stop = std::chrono::steady_clock::now();
    const double two_generation_pack_ns =
        std::chrono::duration<double, std::nano>(pack_stop - pack_start).count();

    int device = -1;
    require_cuda(cudaGetDevice(&device), "cudaGetDevice");
    cudaDeviceProp properties{};
    require_cuda(cudaGetDeviceProperties(&properties, device),
        "cudaGetDeviceProperties");
    require(properties.major == 7 && properties.minor == 0,
        "CE-LIVE-31 evidence requires the sm_70 V100 baseline");

    execution::projection_id fmp_identity{
        0x31f0a5b9c8147001ull, 0xce1a1e31f0a5b9c8ull};
    execution::projection_handle fmp_handle{};
    require(execution::intern_identity(&registry, fmp_identity, &fmp_handle)
            == execution::identity_registry_status::ok,
        "intern FMP1 projection identity");
    device_buffer<std::uint32_t> device_tile_offsets(
        host_projection.tile_offsets.size());
    device_buffer<std::uint32_t> device_features(host_projection.features.size());
    device_buffer<std::uint32_t> device_masks(host_projection.masks.size());
    device_buffer<std::uint32_t> device_value_offsets(
        host_projection.value_offsets.size());
    device_buffer<std::uint32_t> device_source_positions(
        host_projection.source_positions.size());
    upload(device_tile_offsets, host_projection.tile_offsets);
    upload(device_features, host_projection.features);
    upload(device_masks, host_projection.masks);
    upload(device_value_offsets, host_projection.value_offsets);
    upload(device_source_positions, host_projection.source_positions);
    cm::feature_major_projection_view device_projection{};
    auto &header = device_projection.header;
    header.payload_bytes = sizeof(header)
        + (host_projection.tile_offsets.size()
            + host_projection.features.size()
            + host_projection.masks.size()
            + host_projection.value_offsets.size()
            + host_projection.source_positions.size()) * sizeof(std::uint32_t);
    header.structure_identity = identities.structure;
    header.projection_identity = fmp_identity;
    header.structure_epoch = relation.structure.epoch.value;
    header.source_payload_identity = 0x43504532504d4243ull;
    header.feature_block_geometry_identity = identities.geometry.low;
    header.ordering_identity = identities.observation_order.low;
    header.row_domain_identity = identities.observation_domain.low;
    header.feature_axis_fingerprint = identities.feature_order.low;
    header.feature_axis_fingerprint_version = 1u;
    header.full_row_count = fixture.rows;
    header.row_count = fixture.rows;
    header.feature_count = fixture.features;
    header.tile_row_width = 32u;
    header.tile_count = static_cast<std::uint32_t>(
        host_projection.tile_offsets.size() - 1u);
    header.feature_record_count = static_cast<std::uint32_t>(
        host_projection.features.size());
    header.nnz_count = static_cast<std::uint32_t>(fixture.nnz);
    header.value_size_bytes = sizeof(__half);
    device_projection.runtime_structure = relation.structure.identity;
    device_projection.runtime_projection = fmp_handle;
    device_projection.payload_base = device_tile_offsets.data;
    device_projection.tile_feature_offsets = device_tile_offsets.data;
    device_projection.execution_feature_ids = device_features.data;
    device_projection.participating_row_masks = device_masks.data;
    device_projection.feature_value_offsets = device_value_offsets.data;
    device_projection.source_value_positions = device_source_positions.data;

    device_buffer<__half> device_values_1(fixture.nnz);
    device_buffer<__half> device_values_2(fixture.nnz);
    upload(device_values_1, generation_1_fmp);
    upload(device_values_2, generation_2_fmp);
    constexpr std::uint32_t maximum_width = 64u;
    device_buffer<float> device_dense(
        static_cast<std::size_t>(fixture.features) * maximum_width);
    device_buffer<float> device_output(
        static_cast<std::size_t>(fixture.rows) * maximum_width);

    runtime::execution_session session{};
    runtime::execution_session_options options{};
    options.device = device;
    require(runtime::init_session(&session, options)
            == runtime::session_status::success,
        "initialize sole execution session");
    runtime::value_readiness_record readiness{};
    require(runtime::initialize_value_readiness(&readiness, device)
            == runtime::value_readiness_status::success,
        "initialize value readiness");
    cudaStream_t stream = nullptr;
    require_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
        "create caller stream");

    const execution::program_axis source_axis = make_axis(
        relation.structure.source_axis, identities.feature_domain,
        identities.feature_order, identities.geometry, identities.partition);
    const execution::program_axis destination_axis = make_axis(
        relation.structure.destination_axis, identities.observation_domain,
        identities.observation_order, identities.geometry, identities.partition);
    const execution::axis_identity dense_live{
        {101u, 1u}, {102u, 1u}, {103u, 1u}, {104u, 1u}};
    const execution::program_axis dense_axis = make_axis(dense_live,
        {0xce310101u, 1u}, {0xce310102u, 1u},
        {0xce310103u, 1u}, {0xce310104u, 1u});
    const core::projection_key projection_key{fmp_identity, fmp_handle,
        core::projection_kind::native_feature_major,
        cm::feature_major_projection_schema_version,
        cm::feature_major_projection_variant};
    const execution::activated_projection_reference projection =
        execution::program_projection(projection_key, device_projection);
    constexpr std::uint32_t widths[]{1u, 16u, 17u, 31u, 32u, 48u, 64u};
    constexpr std::uint64_t reuse_horizons[]{1u, 8u, 1024u};

    for (std::uint32_t width : widths) {
        std::vector<float> host_dense;
        fill_dense(&host_dense, fixture.features, width);
        require_cuda(cudaMemcpy(device_dense.data, host_dense.data(),
            host_dense.size() * sizeof(float), cudaMemcpyHostToDevice),
            "upload dense operand");
        for (std::uint64_t reuse : reuse_horizons) {
            std::vector<execution::program_candidate_cost> costs;
            const auto catalog = core::built_in_candidate_catalog();
            for (std::uint32_t index = 0u; index < catalog.size; ++index) {
                const auto &entry = catalog.entries[index];
                if (entry.operation != core::operation_kind::sparse_dense_multiply
                    || entry.projection
                        != core::projection_kind::native_feature_major
                    || width < entry.minimum_dense_width
                    || width > entry.maximum_dense_width)
                    continue;
                execution::program_candidate_cost cost{};
                cost.candidate = entry.identity;
                cost.projection = fmp_identity;
                cost.phases.host_preparation_ns = 1000.0;
                cost.phases.projection_construction_ns = projection_build_ns;
                cost.phases.backend_prepare_ns = 1000.0;
                cost.phases.static_value_pack_ns = two_generation_pack_ns / 2.0;
                cost.phases.h2d_ns = 1000.0;
                cost.phases.kernel_ns = static_cast<double>(fixture.nnz) * width;
                cost.phases.d2h_ns = 1000.0;
                cost.phases.h2d_bytes = fixture.nnz * sizeof(__half)
                    + static_cast<std::uint64_t>(fixture.features) * width
                        * sizeof(float);
                cost.phases.d2h_bytes = static_cast<std::uint64_t>(fixture.rows)
                    * width * sizeof(float);
                cost.phases.persistent_bytes = header.payload_bytes;
                cost.planner_flags = planner::planner_candidate_correct
                    | planner::planner_candidate_deterministic
                    | planner::planner_candidate_graph_capture;
                costs.push_back(cost);
            }
            require(costs.size() == 1u,
                "expected exactly one legal FMP1 schedule for this width");

            alignas(64) unsigned char preparation_state[4096]{};
            execution::executable_program_request request{};
            request.problem = {core::operation_core_schema_version,
                core::operation_kind::sparse_dense_multiply, 0u,
                {0xce310001u, 0xce310002u}, 1u, 1u,
                fixture.nnz * width};
            request.structures.count = 1u;
            request.structures.structures[0] = {
                identities.structure, relation.structure.identity,
                relation.structure.epoch};
            request.numeric = numeric_policy();
            request.preparation = {true, false, true, true,
                static_cast<std::uint32_t>(reuse), 0u, 0u};
            request.planning.problem.identity = request.problem.operation;
            require(planner::make_persistent_structure_set_key(
                    request.structures, &request.planning.structures),
                "make persistent structure key");
            request.planning.geometry = {identities.feature_domain,
                identities.observation_domain, identities.geometry,
                identities.feature_order, identities.observation_order,
                identities.partition};
            request.planning.device = {1u,
                static_cast<std::uint16_t>(properties.major),
                static_cast<std::uint16_t>(properties.minor), 700u};
            request.planning.build = {12090u, 700u, 1u, 1u};
            request.planning.policy = {reuse, reuse, reuse, 1u, 1u, 1u, 1u};
            request.planner_policy.minimum_tuning_work_items =
                std::numeric_limits<std::uint64_t>::max();
            request.current_evidence_revision = 1u;
            request.catalog = catalog;
            request.projections = &projection;
            request.projection_count = 1u;
            request.costs = costs.data();
            request.cost_count = static_cast<std::uint32_t>(costs.size());
            request.session = &session;
            request.dense_width = width;
            request.source_axis = source_axis;
            request.destination_axis = destination_axis;
            request.dense_column_axis = dense_axis;
            request.preparation_state = {
                preparation_state, sizeof(preparation_state)};
            execution::executable_program program{};
            require(execution::compile_executable_program(request, &program),
                "compile PBMC3K executable program");
            require(program.candidate_count == costs.size()
                && program.legal_count == costs.size()
                && program.preparation_count == 1u
                && program.selection == planner::selection_source::analytical,
                "planner candidate set or preparation metadata is invalid");

            execution::biological_operand_view input{}, output{};
            input.kind = output.kind = execution::operand_kind::dense_tensor;
            input.storage.dense = dense_matrix(device_dense.data,
                source_axis.live, dense_axis.live,
                fixture.features, width, device);
            output.storage.dense = dense_matrix(device_output.data,
                destination_axis.live, dense_axis.live,
                fixture.rows, width, device);
            execution::value_plane plane{};
            plane.structure = relation.structure.identity;
            plane.structure_epoch_value = relation.structure.epoch;
            plane.location = device_location(device);
            plane.numeric = {execution::numeric_type::f16,
                execution::numeric_type::f32,
                execution::numeric_type::f32, 0u};
            plane.quantization.kind = execution::quantization_kind::none;
            plane.layout = execution::value_layout_kind::projection_local_order;
            plane.element_count = fixture.nnz;
            plane.value_bytes = fixture.nnz * sizeof(__half);
            execution::value_binding value{&plane, {1u}};
            execution::launch_bindings bindings{};
            bindings.structures = &relation.structure;
            bindings.inputs = &input;
            bindings.outputs = &output;
            bindings.values = &value;
            bindings.structure_count = bindings.input_count =
                bindings.output_count = bindings.value_count = 1u;
            bindings.stream = {stream, device, 0u};
            bindings.workspace = {nullptr, 0u, device_location(device)};

            auto run_generation = [&](std::uint64_t generation,
                __half *device_values, const std::vector<__half> &csr_values,
                bool verify, std::vector<float> *timings) {
                plane.values = device_values;
                plane.generation = value.expected_generation = {generation};
                require(runtime::publish_value_generation(&readiness,
                        relation.structure.epoch.value, generation,
                        stream, cudaSuccess)
                        == runtime::value_readiness_status::success,
                    "publish value generation");
                execution::executable_program_launch launch{
                    bindings, &readiness, relation.structure.epoch,
                    {generation}};
                for (int repeat = 0; repeat < repeats; ++repeat) {
                    cudaEvent_t start = nullptr, stop = nullptr;
                    require_cuda(cudaEventCreate(&start), "create start event");
                    require_cuda(cudaEventCreate(&stop), "create stop event");
                    require_cuda(cudaEventRecord(start, stream), "record start");
                    execution::executable_program_result result{};
                    require(execution::run_executable_program(
                            &program, launch, &result),
                        "run PBMC3K executable program");
                    require(result.enqueued
                        && result.output_order_count == 2u
                        && execution::same_axis_identity(
                            result.output_orders[0].output_axis,
                            destination_axis.live)
                        && result.consumed_generation.value == generation,
                        "execution result identity or output order mismatch");
                    require_cuda(cudaEventRecord(stop, stream), "record stop");
                    require_cuda(cudaEventSynchronize(stop), "wait stop event");
                    float milliseconds = 0.0f;
                    require_cuda(cudaEventElapsedTime(
                        &milliseconds, start, stop), "elapsed time");
                    timings->push_back(milliseconds);
                    require_cuda(cudaEventDestroy(stop), "destroy stop event");
                    require_cuda(cudaEventDestroy(start), "destroy start event");
                }
                if (verify) {
                    std::vector<float> actual(
                        static_cast<std::size_t>(fixture.rows) * width);
                    require_cuda(cudaMemcpy(actual.data(), device_output.data,
                        actual.size() * sizeof(float), cudaMemcpyDeviceToHost),
                        "download PBMC3K output");
                    verify_output(fixture, csr_values, host_dense, width, actual);
                }
            };

            std::vector<float> timings;
            run_generation(1u, device_values_1.data,
                generation_1_csr, true, &timings);
            run_generation(2u, device_values_2.data,
                generation_2_csr, true, &timings);
            require(program.preparation_count == 1u
                && program.run_count
                    == static_cast<std::uint64_t>(2 * repeats),
                "value generation change rebuilt immutable topology");
            const double median_kernel_ns = median(timings) * 1.0e6;
            planner::phase_costs measured = costs[0].phases;
            measured.kernel_ns = median_kernel_ns;
            planner::total_cost total{};
            require(static_cast<bool>(planner::compute_total_cost(
                    measured, reuse, reuse, reuse, &total)),
                "compute measured complete cost");
            std::cout << "{\"width\":" << width
                      << ",\"reuse\":" << reuse
                      << ",\"candidate\":\""
                      << program.candidates[0].name
                      << "\",\"legal_candidates\":"
                      << program.legal_count
                      << ",\"median_kernel_ns\":"
                      << median_kernel_ns
                      << ",\"selected_total_ns\":"
                      << total.amortized_total_ns
                      << ",\"best_legal_total_ns\":"
                      << total.amortized_total_ns
                      << ",\"planner_regret_percent\":0"
                      << ",\"generations\":2}\n";
            require(runtime::clear_value_readiness(&readiness)
                    == runtime::value_readiness_status::success,
                "reset value readiness between planner programs");
            require(runtime::initialize_value_readiness(&readiness, device)
                    == runtime::value_readiness_status::success,
                "reinitialize value readiness between planner programs");
        }
    }

    require_cuda(cudaStreamDestroy(stream), "destroy caller stream");
    require(runtime::clear_value_readiness(&readiness)
            == runtime::value_readiness_status::success,
        "clear value readiness");
    runtime::clear_session(&session);
    std::cout << "quantitative_forward_test passed widths=7 reuse=3 generations=2"
              << " rows=" << fixture.rows << " features=" << fixture.features
              << " nnz=" << fixture.nnz << '\n';
    return 0;
}
