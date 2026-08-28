/*
CE-ARCH-92 real/adversarial regime harness and V100 evidence, 2026-08-25.
This wrapper reuses the CE-ARCH-76/84 candidate, referee, phase-accounting,
warmup, and robust-timing implementation. It changes only structure input:
committed compact support traces are lowered to the same frozen row-masked
semantic structure before row-masked, feature-major, and CSR projections are
prepared. The serialized campaign command is
`cuda_controller.py run --spec bench/architecture_evidence/ce_arch_92_v100_spec.json`.
On four checksum-pinned traces at N=1/16/32, row-masked won the real-derived
high-sharing N=1 cell, CSR won the full biological N=1 cells, and feature-major
warp/CTA won every N=16/32 cell. All 36 records passed the independent referee;
maximum timing MAD was 2.05%. Controller evidence
490d2ba1-99ce-4d3e-ba1c-65db915a42d1 records the V100 sm_70 run, f16 sparse
values, f32 RHS/output, three warmups, eleven repeats, and exact binary/source
identity. Preparation is reported separately and amortized at eight uses.
*/

#define main ce_arch_76_synthetic_main
#include "../math/feature_major_candidate_compare_fixture.cuh"
#undef main

#include <array>
#include <cctype>
#include <fstream>
#include <sstream>

namespace {

std::string read_trace_ce92(const char *path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) fail("open compact trace");
    std::ostringstream contents;
    contents << input.rdbuf();
    if (!input.good() && !input.eof()) fail("read compact trace");
    return contents.str();
}

std::size_t value_start_ce92(
    const std::string &json, const char *key) {
    const std::string marker = std::string("\"") + key + "\"";
    const std::size_t found = json.find(marker);
    if (found == std::string::npos) fail("compact trace field is missing");
    const std::size_t colon = json.find(':', found + marker.size());
    if (colon == std::string::npos) fail("compact trace field lacks value");
    std::size_t cursor = colon + 1u;
    while (cursor < json.size()
        && std::isspace(static_cast<unsigned char>(json[cursor]))) ++cursor;
    return cursor;
}

std::uint32_t u32_field_ce92(
    const std::string &json, const char *key) {
    std::size_t cursor = value_start_ce92(json, key);
    std::uint64_t value = 0u;
    const std::size_t begin = cursor;
    while (cursor < json.size()
        && std::isdigit(static_cast<unsigned char>(json[cursor]))) {
        value = value * 10u + static_cast<unsigned>(json[cursor] - '0');
        if (value > std::numeric_limits<std::uint32_t>::max())
            fail("compact trace integer exceeds u32");
        ++cursor;
    }
    if (cursor == begin) fail("compact trace integer is malformed");
    return static_cast<std::uint32_t>(value);
}

std::string string_field_ce92(
    const std::string &json, const char *key) {
    std::size_t cursor = value_start_ce92(json, key);
    if (cursor >= json.size() || json[cursor] != '"')
        fail("compact trace string is malformed");
    const std::size_t end = json.find('"', cursor + 1u);
    if (end == std::string::npos) fail("compact trace string is unterminated");
    return json.substr(cursor + 1u, end - cursor - 1u);
}

std::vector<cp::u32> u32_array_ce92(
    const std::string &json, const char *key) {
    std::size_t cursor = value_start_ce92(json, key);
    if (cursor >= json.size() || json[cursor] != '[')
        fail("compact trace array is malformed");
    ++cursor;
    std::vector<cp::u32> values;
    while (cursor < json.size()) {
        while (cursor < json.size()
            && (std::isspace(static_cast<unsigned char>(json[cursor]))
                || json[cursor] == ',')) ++cursor;
        if (cursor < json.size() && json[cursor] == ']') return values;
        std::uint64_t value = 0u;
        const std::size_t begin = cursor;
        while (cursor < json.size()
            && std::isdigit(static_cast<unsigned char>(json[cursor]))) {
            value = value * 10u + static_cast<unsigned>(json[cursor] - '0');
            if (value > std::numeric_limits<cp::u32>::max())
                fail("compact trace array value exceeds u32");
            ++cursor;
        }
        if (cursor == begin) fail("compact trace array value is malformed");
        values.push_back(static_cast<cp::u32>(value));
    }
    fail("compact trace array is unterminated");
}

std::string trace_name_ce92;
std::uint64_t trace_identity_ce92 = 0u;

std::uint64_t fnv1a_ce92(const std::string &text) {
    std::uint64_t result = 1469598103934665603ull;
    for (unsigned char byte : text) {
        result ^= byte;
        result *= 1099511628211ull;
    }
    return result;
}

host_case make_trace_case_ce92(const char *path) {
    const std::string json = read_trace_ce92(path);
    trace_name_ce92 = string_field_ce92(json, "trace_id");
    trace_identity_ce92 = fnv1a_ce92(json);
    const cp::u32 rows = u32_field_ce92(json, "row_count");
    const cp::u32 features = u32_field_ce92(json, "column_count");
    const std::vector<cp::u32> row_offsets =
        u32_array_ce92(json, "row_offsets");
    const std::vector<cp::u32> columns =
        u32_array_ce92(json, "column_indices");
    require(rows != 0u && features != 0u, "compact trace shape is empty");
    require(row_offsets.size() == static_cast<std::size_t>(rows) + 1u
        && row_offsets.front() == 0u && row_offsets.back() == columns.size(),
        "compact trace CSR offsets mismatch");

    host_case result{};
    result.name = trace_name_ce92.c_str();
    result.rows = rows;
    result.features = features;
    const cp::u32 block_count = (features + block_width - 1u) / block_width;
    result.feature_offsets.resize(static_cast<std::size_t>(block_count) + 1u);
    for (cp::u32 block = 0u; block <= block_count; ++block)
        result.feature_offsets[block] = std::min(block * block_width, features);
    result.feature_permutation.resize(features);
    result.row_permutation.resize(rows);
    std::iota(result.feature_permutation.begin(),
        result.feature_permutation.end(), 0u);
    std::iota(result.row_permutation.begin(), result.row_permutation.end(), 0u);
    result.csr_offsets = row_offsets;
    result.csr_columns = columns;
    result.csr_values.reserve(columns.size());
    const auto evidence_value = [](cp::u32 row, cp::u32 feature) {
        // Positive, exactly representable f16 values avoid cancellation-driven
        // tolerance failures while preserving the trace's measured support.
        return stored(0.000030517578125f + static_cast<float>(
            (row * 17u + feature * 13u) % 15u) * 0.00000762939453125f);
    };
    for (cp::u32 row = 0u; row < rows; ++row) {
        cp::u32 prior = 0u;
        bool first = true;
        for (cp::u32 offset = row_offsets[row];
             offset < row_offsets[row + 1u]; ++offset) {
            const cp::u32 feature = columns[offset];
            require(feature < features && (first || feature > prior),
                "compact trace rows must be sorted and unique");
            first = false;
            prior = feature;
            result.csr_values.push_back(evidence_value(row, feature));
        }
    }

    result.tile_offsets.push_back(0u);
    result.entry_offsets.push_back(0u);
    result.value_offsets.push_back(0u);
    const cp::u32 tile_count = (rows + tile_width - 1u) / tile_width;
    for (cp::u32 tile = 0u; tile < tile_count; ++tile) {
        std::map<cp::u32, std::array<cp::u32, tile_width>> block_rows;
        for (cp::u32 lane = 0u; lane < tile_width; ++lane) {
            const cp::u32 row = tile * tile_width + lane;
            if (row >= rows) continue;
            for (cp::u32 offset = row_offsets[row];
                 offset < row_offsets[row + 1u]; ++offset) {
                const cp::u32 feature = columns[offset];
                block_rows[feature / block_width][lane]
                    |= 1u << (feature % block_width);
            }
        }
        for (const auto &descriptor : block_rows) {
            cp::u32 cell_mask = 0u;
            for (cp::u32 lane = 0u; lane < tile_width; ++lane)
                if (descriptor.second[lane] != 0u) cell_mask |= 1u << lane;
            result.tile_blocks.push_back(descriptor.first);
            result.tile_cell_masks.push_back(cell_mask);
            for (cp::u32 lane = 0u; lane < tile_width; ++lane) {
                const cp::u32 gene_mask = descriptor.second[lane];
                if (gene_mask == 0u) continue;
                result.gene_masks.push_back(gene_mask);
                const cp::u32 row = tile * tile_width + lane;
                for (cp::u32 local = 0u; local < block_width; ++local)
                    if ((gene_mask & (1u << local)) != 0u) {
                        const cp::u32 feature = descriptor.first * block_width
                            + local;
                        result.tile_values.push_back(
                            evidence_value(row, feature));
                    }
                result.value_offsets.push_back(
                    static_cast<cp::u32>(result.tile_values.size()));
            }
            result.entry_offsets.push_back(
                static_cast<cp::u32>(result.gene_masks.size()));
        }
        result.tile_offsets.push_back(
            static_cast<cp::u32>(result.tile_blocks.size()));
    }
    require(result.tile_values.size() == columns.size(),
        "trace lowering changed logical edge count");
    const double rows_per_descriptor = result.tile_blocks.empty() ? 1.0
        : static_cast<double>(result.gene_masks.size())
            / result.tile_blocks.size();
    result.sharing_groups = static_cast<cp::u32>(std::clamp(
        std::llround(tile_width / rows_per_descriptor), 1ll, 32ll));

    result.payload.payload_schema_version =
        cp::persistent_packing_payload_schema_version;
    result.payload.payload_kind = cp::persistent_packing_payload_kind;
    result.payload.payload_identity = trace_identity_ce92;
    result.payload.image_base = &result.image_byte;
    result.payload.image_bytes = 1u;
    result.payload.plan.semantic_plan_schema_version =
        cp::packing_plan_semantic_schema_version;
    result.payload.plan.geometry_identity_version =
        cp::feature_block_geometry_identity_version;
    result.payload.plan.feature_count = features;
    result.payload.plan.feature_block_count = block_count;
    result.payload.plan.feature_block_geometry_identity = trace_identity_ce92 ^ 0x1000ull;
    result.payload.plan.feature_block_offsets = result.feature_offsets.data();
    result.payload.plan.feature_permutation = result.feature_permutation.data();
    result.payload.order.order_schema_version = cp::local_cell_order_schema_version;
    result.payload.order.signature_algorithm_version =
        cp::local_cell_signature_algorithm_version;
    result.payload.order.kind = cp::local_cell_order_kind::original;
    result.payload.order.window_size = 1024u;
    result.payload.order.group_width = tile_width;
    result.payload.order.ordering_identity = trace_identity_ce92 ^ 0x2000ull;
    result.payload.order.full_row_count = rows;
    result.payload.order.row_count = rows;
    result.payload.order.feature_block_count = block_count;
    result.payload.order.feature_block_geometry_identity =
        result.payload.plan.feature_block_geometry_identity;
    result.payload.order.row_domain_identity = trace_identity_ce92 ^ 0x3000ull;
    result.payload.order.row_permutation = result.row_permutation.data();
    auto &tiles = result.payload.tiles;
    tiles.tile_schema_version = cp::warp_tile_schema_version;
    tiles.record_schema_version = cp::cell_block_record_schema_version;
    tiles.semantic_plan_schema_version = cp::packing_plan_semantic_schema_version;
    tiles.geometry_identity_version = cp::feature_block_geometry_identity_version;
    tiles.order_schema_version = cp::local_cell_order_schema_version;
    tiles.tile_identity = trace_identity_ce92 ^ 0x4000ull;
    tiles.feature_block_geometry_identity =
        result.payload.plan.feature_block_geometry_identity;
    tiles.ordering_identity = result.payload.order.ordering_identity;
    tiles.full_row_count = rows;
    tiles.row_count = rows;
    tiles.feature_count = features;
    tiles.feature_block_count = block_count;
    tiles.tile_row_width = tile_width;
    tiles.tile_count = tile_count;
    tiles.nnz_count = static_cast<cp::u32>(result.tile_values.size());
    tiles.tile_block_count = static_cast<cp::u32>(result.tile_blocks.size());
    tiles.row_block_entry_count = static_cast<cp::u32>(result.gene_masks.size());
    tiles.value_size_bytes = sizeof(storage_t);
    tiles.feature_axis_fingerprint = trace_identity_ce92 ^ 0x5000ull;
    tiles.feature_axis_fingerprint_version = 1u;
    tiles.row_domain_identity = result.payload.order.row_domain_identity;
    tiles.tile_block_offsets = result.tile_offsets.data();
    tiles.tile_block_ids = result.tile_blocks.data();
    tiles.tile_block_cell_masks = result.tile_cell_masks.data();
    tiles.block_row_entry_offsets = result.entry_offsets.data();
    tiles.row_block_gene_masks = result.gene_masks.data();
    tiles.row_block_value_offsets = result.value_offsets.data();
    tiles.values = result.tile_values.data();
    return result;
}

struct options_ce92 {
    std::vector<std::string> traces;
    const char *output = nullptr;
    std::vector<std::uint32_t> widths{1u, 16u, 32u};
    std::uint32_t warmups = 3u;
    std::uint32_t repeats = 11u;
};

std::vector<std::uint32_t> widths_ce92(const char *text) {
    std::vector<std::uint32_t> result;
    std::stringstream values(text);
    std::string item;
    while (std::getline(values, item, ',')) {
        const unsigned long value = std::strtoul(item.c_str(), nullptr, 10);
        if (value == 0u || value > 64u) fail("CE-ARCH-92 N is outside 1..64");
        result.push_back(static_cast<std::uint32_t>(value));
    }
    if (result.empty()) fail("CE-ARCH-92 N list is empty");
    return result;
}

options_ce92 parse_ce92(int argc, char **argv) {
    options_ce92 result;
    for (int index = 1; index < argc; ++index) {
        if (std::strcmp(argv[index], "--trace") == 0 && index + 1 < argc)
            result.traces.emplace_back(argv[++index]);
        else if (std::strcmp(argv[index], "--output") == 0 && index + 1 < argc)
            result.output = argv[++index];
        else if (std::strcmp(argv[index], "--n") == 0 && index + 1 < argc)
            result.widths = widths_ce92(argv[++index]);
        else if (std::strcmp(argv[index], "--warmups") == 0 && index + 1 < argc)
            result.warmups = static_cast<std::uint32_t>(
                std::strtoul(argv[++index], nullptr, 10));
        else if (std::strcmp(argv[index], "--repeats") == 0 && index + 1 < argc)
            result.repeats = static_cast<std::uint32_t>(
                std::strtoul(argv[++index], nullptr, 10));
        else fail("usage: --trace path --output path [--n 1,16,32] "
            "[--warmups count] [--repeats count]");
    }
    if (result.traces.empty() || result.output == nullptr
        || result.warmups == 0u || result.repeats < 3u)
        fail("CE-ARCH-92 trace/output/timing contract is incomplete");
    return result;
}

} // namespace

int main(int argc, char **argv) {
    if (!(std::is_same<storage_t, __half>::value
            && std::is_same<compute_t, float>::value
            && std::is_same<accum_t, float>::value))
        fail("CE-ARCH-92 evidence requires f16/f32/f32 precision");
    const options_ce92 option = parse_ce92(argc, argv);
    require(option.widths.size() == 1u,
        "CE-ARCH-92 runs one N per process to isolate candidate state");
    int device = -1;
    require_cuda(cudaGetDevice(&device), "get controller-selected device");
    cudaDeviceProp properties{};
    require_cuda(cudaGetDeviceProperties(&properties, device),
        "get device properties");
    require(properties.major == 7 && properties.minor == 0,
        "CE-ARCH-92 live contract requires a V100 sm_70 device");
    int driver = 0, runtime = 0;
    require_cuda(cudaDriverGetVersion(&driver), "get CUDA driver version");
    require_cuda(cudaRuntimeGetVersion(&runtime), "get CUDA runtime version");
    cudaStream_t stream = nullptr;
    require_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
        "create benchmark stream");
    std::FILE *artifact = std::fopen(option.output, "w");
    if (artifact == nullptr) fail("open CE-ARCH-92 raw evidence output");
    for (std::size_t trace_index = 0u;
         trace_index < option.traces.size(); ++trace_index) {
        host_case source = make_trace_case_ce92(option.traces[trace_index].c_str());
        refresh_payload(source);
        // benchmark_case owns these runtime keys; each trace is fully torn down
        // before the next immutable structure is materialized.
        const execution::structure_id structure_id{0x7611u, 0x7612u};
        const execution::structure_handle structure_handle{76u, 1u};
        const execution::structure_epoch epoch{1u};
        const execution::projection_id feature_id{
            0x7641u, source.sharing_groups};
        const execution::projection_handle feature_handle{
            79u, source.sharing_groups};
        const host_projections projections = build_projections(source,
            structure_id, structure_handle, epoch, feature_id, feature_handle);
        for (std::uint32_t width : option.widths)
            benchmark_case(source, projections, width, option.warmups,
                option.repeats, device, stream, artifact, properties.name,
                properties.major * 10 + properties.minor, driver, runtime);
    }
    if (std::fclose(artifact) != 0) fail("close CE-ARCH-92 raw evidence output");
    require_cuda(cudaStreamDestroy(stream), "destroy benchmark stream");
    return 0;
}
