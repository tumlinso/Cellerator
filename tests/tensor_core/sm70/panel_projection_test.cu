#include <Cellerator/layout/projection/panel_projection.hh>
#include <Cellerator/memory/tiled/PanelStagingSpan.hh>
#include <Cellerator/ops/matrix/indexed_tiled_matmul_cuda.cuh>
#include <Cellerator/ops/matrix/panel_projection_cuda.cuh>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {

namespace layout = cellerator::layout::projection;
namespace matrix = cellerator::ops::matrix;
namespace memory = cellerator::memory;
namespace gpu = memory::cuda;
using Status = layout::PanelPackingStatus;

constexpr size_t source_rows = 19, features = 97, selected_rows = 17, output_tiles = 3;
constexpr float sentinel = -1234.0f;

void require(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}

uint16_t half_bits(float value) { return __half_as_ushort(__float2half_rn(value)); }
float as_float(uint16_t bits) { return __half2float(__ushort_as_half(bits)); }

int find_volta() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) return -1;
    for (int device = 0; device < count; ++device) {
        cudaDeviceProp properties{};
        gpu::check(cudaGetDeviceProperties(&properties, device), "read device properties");
        if (properties.major == 7 && properties.minor == 0) return device;
    }
    return -1;
}

struct Fixture {
    uint32_t offsets[source_rows + 1]{}, rows[selected_rows]{};
    uint16_t block_features[7*16]{};
    const uint16_t dense_blocks[5]{4, 0, 1, 2, 3};
    uint32_t feature_positions[features]{};
    uint16_t packed_columns[features]{};
    std::vector<uint32_t> columns;
    std::vector<uint16_t> values, weights;
    std::vector<uint16_t> dense_features, residual_features, dense, residual, packed_weights;
    std::vector<uint32_t> residual_offsets;
    std::vector<uint16_t> routes;
    layout::PanelPackingSource source;
    layout::PanelPackingLayout panel_layout;
    layout::PanelPackingPlan plan;

    void prepare() {
        for (size_t row = 0; row < source_rows; ++row) {
            for (size_t feature = row%3; feature < features; feature += 3) {
                columns.push_back(uint32_t(feature));
                values.push_back(half_bits(float(int((row*7 + feature*3)%19) - 9)*0.125f));
            }
            offsets[row + 1] = uint32_t(columns.size());
        }
        for (size_t row = 0; row < selected_rows; ++row) rows[row] = uint32_t(source_rows - 1 - row);
        uint32_t feature_order[features];
        for (size_t feature = 0; feature < features; ++feature) feature_order[feature] = uint32_t(feature);
        const uint32_t block_offsets[]{0, 16, 32, 48, 64, 80, 96, features};
        require(layout::pack_panel_block_features(feature_order, features, block_offsets, 7,
                    block_features, 7*16) == Status::ok, "convert prepared feature blocks");
        require(block_features[96] == 96 && block_features[97] == layout::panel_padding_feature,
                "short feature block padding");
        const cellerator::tensor::base::CompressedStructureView<uint32_t> csr{
            source_rows, features, columns.size(), cellerator::tensor::base::CompressedAxis::row,
            offsets, columns.data()};
        require(layout::prepare_panel_packing_source(csr, block_features, 7, feature_positions,
                    features, &source) == Status::ok, "prepare source CSR and feature topology");
        dense_features.resize(2*layout::panel_columns);
        const layout::PanelPackingLayoutStorage layout_storage{dense_features.data(),
            dense_features.size(), packed_columns, nullptr, features};
        auto short_layout_storage = layout_storage;
        --short_layout_storage.feature_capacity;
        require(layout::prepare_panel_packing_layout(source, dense_blocks, 5, short_layout_storage,
                    &panel_layout) == Status::insufficient_capacity, "layout accepted short column storage");
        const uint16_t duplicate_blocks[]{4, 4};
        require(layout::prepare_panel_packing_layout(source, duplicate_blocks, 2, layout_storage,
                    &panel_layout) == Status::duplicate_feature, "layout accepted a repeated logical block");
        require(layout::prepare_panel_packing_layout(source, dense_blocks, 5, layout_storage,
                    &panel_layout) == Status::ok, "prepare recurring panel layout after rejection");
        weights.resize(output_tiles*16*features);
        for (size_t out = 0; out < output_tiles*16; ++out)
            for (size_t feature = 0; feature < features; ++feature)
                weights[out*features + feature] = half_bits(float(int((out*11 + feature*13)%17) - 8)*0.0625f);
        packed_weights.resize(output_tiles*panel_layout.panels*layout::panel_elements);
        require(layout::pack_panel_weights(panel_layout, weights.data(), output_tiles, features,
                    packed_weights.data(), packed_weights.size()) == Status::ok, "pack panel weights");

        uint8_t row_marks[(source_rows + 7)/8]{};
        const layout::PanelPackingWorkspace workspace{nullptr, 0, row_marks, sizeof(row_marks)};
        layout::PanelPackingSelection selection;
        require(layout::query_panel_packing(source, panel_layout, rows, selected_rows, {}, &selection) ==
                    Status::insufficient_capacity, "query accepted insufficient row workspace");
        require(layout::query_panel_packing(source, panel_layout, rows, selected_rows, workspace,
                    &selection) == Status::ok, "query panel packing with prepared order");
        const auto &req = selection.shape;
        require(req.groups == 2 && req.panels == 2 && req.residual_offsets == 33 && req.residual_nnz,
                "fixture must exercise panel and cell tails and residuals");
        require(req.routes == columns.size() - offsets[2] && !req.wide_routes,
                "fixture must use one compact route per selected entry");
        residual_features.resize(req.residual_nnz);
        residual_offsets.resize(req.residual_offsets);
        routes.resize(req.routes);
        const layout::PanelPackingMetadata metadata{nullptr, 0, residual_offsets.data(),
            residual_offsets.size(), residual_features.data(),
            residual_features.size()};
        const layout::PanelPackingStorage storage{metadata, routes.data(), nullptr, routes.size()};
        auto short_storage = storage;
        --short_storage.route_capacity;
        require(layout::prepare_panel_packing(selection, {}, short_storage, &plan) ==
                    Status::insufficient_capacity, "prepare accepted insufficient route capacity");
        require(layout::prepare_panel_packing(selection, {}, storage, &plan) == Status::ok,
                "prepare panel packing");
        require(plan.dense_features == panel_layout.dense_features,
                "cohort plan did not borrow the shared dense feature table");
        dense.assign(req.dense_values, uint16_t(0xffff));
        residual.resize(req.residual_nnz);
        require(layout::pack_panel_values(plan, values.data(), values.size(), dense.data(), dense.size(),
                    residual.data(), residual.size()) == Status::ok, "pack panel values");
        std::vector<uint16_t> direct_dense(req.dense_values, 0xffff), direct_residual(req.residual_nnz);
        std::vector<uint16_t> direct_residual_features(req.residual_nnz);
        std::vector<uint32_t> direct_offsets(req.residual_offsets);
        const layout::PanelPackingMetadata direct_metadata{nullptr, 0, direct_offsets.data(), direct_offsets.size(),
            direct_residual_features.data(), direct_residual_features.size()};
        require(layout::pack_panel_values_once(selection, {}, direct_metadata,
                    values.data(), values.size(), direct_dense.data(), direct_dense.size(),
                    direct_residual.data(), direct_residual.size()) == Status::ok, "one-shot panel pack");
        require(direct_dense == dense && direct_residual == residual && direct_offsets == residual_offsets &&
                    direct_residual_features == residual_features,
                "one-shot and reusable plans disagree");
        uint32_t duplicate_rows[selected_rows];
        std::copy_n(rows, selected_rows, duplicate_rows);
        duplicate_rows[selected_rows - 1] = duplicate_rows[0];
        layout::PanelPackingSelection duplicate_selection;
        require(layout::query_panel_packing(source, panel_layout, duplicate_rows, selected_rows, workspace,
                    &duplicate_selection) == Status::duplicate_row, "query accepted repeated source row");
        require(layout::query_panel_packing(source, panel_layout, rows, selected_rows, workspace,
                    &duplicate_selection) == Status::ok, "row scratch could not be reused after rejection");
        require(dense_features[0] == 64 && dense_features[16] == 0
                    && dense_features[80] == layout::panel_padding_feature,
                "selected block order or panel padding");
        require(dense[2*layout::panel_elements + 64] == 0 && dense[layout::panel_elements + 16] == 0,
                "cell or feature tail remains dirty");
        for (size_t row = selected_rows; row < req.groups*16; ++row)
            require(residual_offsets[row] == residual_offsets[row + 1], "nonempty sparse tail row");
    }

    void verify_packing() const {
        std::vector<uint16_t> expected_dense(dense.size(), 0);
        std::vector<uint16_t> expected_residual, expected_features;
        std::vector<uint32_t> expected_offsets;
        for (size_t lane = 0; lane < plan.shape.groups*16; ++lane) {
            expected_offsets.push_back(uint32_t(expected_residual.size()));
            if (lane >= selected_rows) continue;
            for (size_t k = offsets[rows[lane]]; k < offsets[rows[lane] + 1]; ++k) {
                const uint32_t feature = columns[k];
                if (feature >= 80) {
                    expected_residual.push_back(values[k]);
                    expected_features.push_back(uint16_t(feature));
                } else {
                    // Block 4 comes first, followed by blocks 0 through 3.
                    const size_t column = feature >= 64 ? feature - 64 : feature + 16;
                    const size_t panel = 2*(lane/16) + column/64;
                    expected_dense[panel*1024 + (lane%16)*64 + column%64] = values[k];
                }
            }
        }
        expected_offsets.push_back(uint32_t(expected_residual.size()));
        require(dense == expected_dense, "dense pack differs from CSR coordinates");
        require(residual == expected_residual && residual_features == expected_features &&
                    residual_offsets == expected_offsets, "residual pack differs from CSR coordinates");
    }

    void verify_shared_layout() const {
        const size_t last_row = rows[selected_rows - 1];
        const size_t last_begin = offsets[last_row], last_nnz = offsets[last_row + 1] - last_begin;
        const uint32_t tail_offsets[]{0, uint32_t(last_nnz)};
        const cellerator::tensor::base::CompressedStructureView<uint32_t> tail_csr{
            1, features, last_nnz, cellerator::tensor::base::CompressedAxis::row,
            tail_offsets, columns.data() + last_begin};
        uint32_t tail_positions[features];
        layout::PanelPackingSource tail_source;
        require(layout::prepare_panel_packing_source(tail_csr, block_features, 7, tail_positions,
                    features, &tail_source) == Status::ok, "prepare another CSR with the same feature catalog");
        for (size_t group = 0; group < plan.shape.groups; ++group) {
            const size_t row_start = group*layout::panel_rows;
            const size_t row_count = std::min(layout::panel_rows, selected_rows - row_start);
            const size_t residual_begin = residual_offsets[row_start];
            const size_t residual_end = residual_offsets[row_start + layout::panel_rows];
            const bool tail = row_count == 1;
            layout::PanelPackingSelection selection;
            require(layout::query_panel_packing(tail ? tail_source : source, panel_layout,
                        tail ? nullptr : rows + row_start, row_count,
                        {}, &selection) == Status::ok, "query a distinct cohort with the shared order");
            const auto &req = selection.shape;
            require(req.groups == 1 && req.panels == panel_layout.panels &&
                        req.dense_values == dense.size()/plan.shape.groups &&
                        req.residual_offsets == layout::panel_rows + 1 &&
                        req.residual_nnz == residual_end - residual_begin,
                    "cohort extents differ from its slice of the shared layout");
            std::vector<uint16_t> cohort_routes(req.routes), cohort_dense(req.dense_values, 0xffff);
            std::vector<uint16_t> cohort_residual(req.residual_nnz), cohort_features(req.residual_nnz);
            uint32_t cohort_offsets[layout::panel_rows + 1]{};
            const layout::PanelPackingMetadata metadata{nullptr, 0, cohort_offsets,
                layout::panel_rows + 1, cohort_features.data(), cohort_features.size()};
            const layout::PanelPackingStorage storage{metadata, cohort_routes.data(), nullptr,
                cohort_routes.size()};
            layout::PanelPackingPlan cohort;
            require(layout::prepare_panel_packing(selection, {}, storage, &cohort) == Status::ok &&
                        cohort.dense_features == plan.dense_features,
                    "prepare rows without rebuilding the shared feature table");
            require(layout::pack_panel_values(cohort, values.data() + (tail ? last_begin : 0),
                        tail ? last_nnz : values.size(),
                        cohort_dense.data(), cohort_dense.size(), cohort_residual.data(),
                        cohort_residual.size()) == Status::ok, "pack independent cohort");
            require(std::equal(cohort_dense.begin(), cohort_dense.end(), dense.data() + group*req.dense_values)
                        && std::equal(cohort_residual.begin(), cohort_residual.end(),
                                      residual.data() + residual_begin)
                        && std::equal(cohort_features.begin(), cohort_features.end(),
                                      residual_features.data() + residual_begin),
                    "independent cohort differs from its CSR-verified batch slice");
            for (size_t row = 0; row < req.residual_offsets; ++row)
                require(cohort_offsets[row] == residual_offsets[row_start + row] - residual_begin,
                        "cohort residual offsets were not rebased onto its rows");
        }
        uint16_t copied_catalog[7*16];
        std::copy_n(block_features, 7*16, copied_catalog);
        require(layout::prepare_panel_packing_source(tail_csr, copied_catalog, 7, tail_positions,
                    features, &tail_source) == Status::ok, "prepare CSR with a distinct catalog");
        layout::PanelPackingSelection selection;
        require(layout::query_panel_packing(tail_source, panel_layout, nullptr, 1, {}, &selection) ==
                    Status::invalid_argument, "layout accepted another catalog identity");
    }

    void verify_scatter() const {
        std::vector<float> dense_grad(dense.size()), residual_grad(residual.size());
        for (size_t i = 0; i < dense.size(); ++i) dense_grad[i] = as_float(dense[i]);
        for (size_t i = 0; i < residual.size(); ++i) residual_grad[i] = as_float(residual[i]);
        std::vector<float> canonical(values.size(), sentinel);
        require(layout::scatter_panel_value_gradients(plan, dense_grad.data(), dense_grad.size(),
                    residual_grad.data(), residual_grad.size(), canonical.data(), canonical.size()) == Status::ok,
                "scatter panel gradients");
        for (size_t i = 0; i < canonical.size(); ++i)
            require(canonical[i] == (i < offsets[2] ? sentinel : as_float(values[i])),
                    "reverse scatter did not preserve CSR position or row subset");

        std::vector<float> weight_gradient(packed_weights.size(), 1.0f);
        std::vector<float> canonical_weight_gradient(weights.size(), 0.0f);
        for (int pass = 0; pass < 2; ++pass)
            require(layout::accumulate_panel_weight_gradients(plan, weight_gradient.data(),
                    weight_gradient.size(), output_tiles, canonical_weight_gradient.data(), features) ==
                    Status::ok, "accumulate packed weight gradients");
        for (size_t i = 0; i < weights.size(); ++i)
            require(canonical_weight_gradient[i] == (i%features < 80 ? 2.0f : 0.0f),
                    "packed weight gradients missed or duplicated a canonical coordinate");
    }

    void verify_refresh() const {
        std::vector<uint16_t> changed(values.size()), refreshed_dense(dense.size(), 0xffff);
        std::vector<uint16_t> refreshed_residual(residual.size(), 0xffff);
        for (size_t k = 0; k < changed.size(); ++k) changed[k] = values[k] ? values[k] ^ 0x8000 : 0;
        require(layout::pack_panel_values(plan, changed.data(), changed.size(),
                    refreshed_dense.data(), refreshed_dense.size(),
                    refreshed_residual.data(), refreshed_residual.size()) == Status::ok,
                "refresh cohort from prepared routes");
        for (size_t k = 0; k < dense.size(); ++k)
            require(refreshed_dense[k] == (dense[k] ? (dense[k] ^ 0x8000) : 0),
                    "dense value refresh differs or left a stale hole");
        for (size_t k = 0; k < residual.size(); ++k)
            require(refreshed_residual[k] == (residual[k] ? (residual[k] ^ 0x8000) : 0),
                    "residual value refresh differs");

        const size_t source_entry = offsets[rows[0]], packed_entry = 16;
        require(columns[source_entry] == 0 && dense[packed_entry] != 0,
                "fixture must overwrite a nonzero value in the second panel slice");
        changed[source_entry] = 0;
        refreshed_dense[packed_entry] = 0;
        auto seeded_dense = dense, seeded_residual = residual;
        require(layout::refresh_panel_values(plan, changed.data(), changed.size(), seeded_dense.data(),
                    seeded_dense.size(), seeded_residual.data(), seeded_residual.size()) == Status::ok,
                "overwrite existing cohort layout");
        require(seeded_dense == refreshed_dense && seeded_residual == refreshed_residual,
                "refresh of the existing layout differs or preserves a stale value");
    }

    float reference(size_t out, size_t row) const {
        if (row >= selected_rows) return 0.0f;
        float sum = 0.0f;
        for (size_t k = offsets[rows[row]]; k < offsets[rows[row] + 1]; ++k)
            sum += as_float(values[k])*as_float(weights[out*features + columns[k]]);
        return sum;
    }
};

void verify_wide_routes() {
    constexpr size_t block_count = 4096, column_count = block_count*16;
    constexpr uint32_t canonical_feature = uint32_t(column_count - 2);
    const uint32_t offsets[]{0, 2}, features[]{0, canonical_feature};
    const uint16_t values[]{half_bits(1), half_bits(2)};
    const cellerator::tensor::base::CompressedStructureView<uint32_t> csr{
        1, column_count - 1, 2, cellerator::tensor::base::CompressedAxis::row, offsets, features};
    std::vector<uint16_t> block_features(column_count, layout::panel_padding_feature);
    for (size_t k = 2; k < column_count; ++k) block_features[k] = uint16_t(k - 1);
    std::vector<uint32_t> feature_positions(csr.cols);
    layout::PanelPackingSource source;
    require(layout::prepare_panel_packing_source(csr, block_features.data(), block_count,
                feature_positions.data(), feature_positions.size(), &source) == Status::ok,
            "prepare wide column topology");
    std::vector<uint16_t> blocks(block_count), dense_features(column_count);
    std::vector<uint32_t> packed_columns(csr.cols);
    for (size_t k = 0; k < block_count; ++k) blocks[k] = uint16_t(k);
    const layout::PanelPackingLayoutStorage layout_storage{dense_features.data(),
        dense_features.size(), nullptr, packed_columns.data(), packed_columns.size()};
    layout::PanelPackingLayout panel_layout;
    require(layout::prepare_panel_packing_layout(source, blocks.data(), blocks.size(), layout_storage,
                &panel_layout) == Status::ok, "prepare shared wide panel layout");
    layout::PanelPackingSelection selection;
    require(layout::query_panel_packing(source, panel_layout, nullptr, 1, {}, &selection) == Status::ok,
            "query wide panel packing");
    const auto &req = selection.shape;
    require(req.wide_routes && req.routes == 2 && req.residual_nnz == 1,
            "wide local column and residual collided");
    std::vector<uint16_t> dense(req.dense_values);
    std::vector<uint32_t> residual_offsets(req.residual_offsets);
    uint16_t residual_features[1]{}, residual[1]{};
    uint32_t routes[2]{};
    const layout::PanelPackingMetadata metadata{nullptr, 0,
        residual_offsets.data(), residual_offsets.size(), residual_features, 1};
    layout::PanelPackingPlan plan;
    require(layout::prepare_panel_packing(selection, {}, {metadata, nullptr, routes, 2}, &plan) ==
                Status::ok, "prepare wide panel packing");
    require(routes[0] == UINT32_MAX && routes[1] == UINT16_MAX,
            "wide route did not distinguish last column from residual");
    require(layout::pack_panel_values(plan, values, 2, dense.data(), dense.size(), residual, 1) ==
                Status::ok, "pack wide panel values");
    const size_t wide_position = (req.panels - 1)*1024 + 63;
    require(dense[wide_position] == values[1] && residual[0] == values[0] &&
                residual_features[0] == 0, "wide route packed the wrong coordinates");
    std::vector<float> dense_gradient(dense.size(), 0);
    dense_gradient[wide_position] = 3.0f;
    const float residual_gradient[]{-4.0f};
    float source_gradient[2]{};
    require(layout::scatter_panel_value_gradients(plan, dense_gradient.data(),
                dense_gradient.size(), residual_gradient, 1, source_gradient, 2) == Status::ok &&
                source_gradient[0] == -4.0f && source_gradient[1] == 3.0f,
            "wide route restored the wrong gradients");
}

void verify_overlapping_blocks() {
    const uint32_t offsets[]{0, 3}, features[]{0, 1, 2};
    const cellerator::tensor::base::CompressedStructureView<uint32_t> csr{
        1, 3, 3, cellerator::tensor::base::CompressedAxis::row, offsets, features};
    uint16_t block_features[3*16];
    std::fill_n(block_features, 3*16, layout::panel_padding_feature);
    block_features[0] = 0;
    block_features[1] = 1;
    block_features[16 + 4] = 1;
    block_features[16 + 6] = 2;
    uint32_t feature_positions[3]{};
    layout::PanelPackingSource source;
    require(layout::prepare_panel_packing_source(csr, block_features, 3, feature_positions,
                3, &source) == Status::ok && source.ambiguous_features == 1,
            "prepare a catalog containing a shared feature");
    uint16_t slots[3]{};
    uint32_t ambiguous_columns[1]{};
    const layout::PanelPackingWorkspace workspace{slots, 3, nullptr, 0, ambiguous_columns, 1};
    const uint16_t blocks[]{0, 1, 2};
    layout::PanelPackingSelection selection;
    require(layout::query_panel_packing(source, {nullptr, 1, blocks, 2}, workspace, &selection) ==
                Status::duplicate_feature, "query accepted selected blocks sharing a feature");
    require(layout::query_panel_packing(source, {nullptr, 1, blocks + 2, 1}, workspace, &selection) !=
                Status::ok, "query accepted a padding-only block");
    uint16_t dense_features[64]{}, residual_features[1]{}, routes[3]{};
    uint32_t residual_offsets[17]{};
    const layout::PanelPackingMetadata metadata{dense_features, 64,
        residual_offsets, 17, residual_features, 1};
    const layout::PanelPackingStorage storage{metadata, routes, nullptr, 3};
    uint16_t packed_columns[3]{};
    const layout::PanelPackingLayoutStorage layout_storage{dense_features, 64, packed_columns, nullptr, 3};
    layout::PanelPackingLayout panel_layout;
    require(layout::prepare_panel_packing_layout(source, blocks, 2, layout_storage, &panel_layout) ==
                Status::duplicate_feature, "shared layout accepted two occurrences of one feature");
    require(layout::prepare_panel_packing_layout(source, blocks + 2, 1, layout_storage, &panel_layout) !=
                Status::ok, "shared layout accepted a padding-only block");
    const layout::PanelPackingMetadata borrowed_metadata{nullptr, 0, residual_offsets, 17, residual_features, 1};
    const layout::PanelPackingStorage borrowed_storage{borrowed_metadata, routes, nullptr, 3};
    layout::PanelPackingPlan plan;
    for (size_t block = 0; block < 2; ++block) {
        require(layout::query_panel_packing(source, {nullptr, 1, blocks + block, 1},
                    workspace, &selection) == Status::ok &&
                    layout::prepare_panel_packing(selection, workspace, storage, &plan) == Status::ok,
                "could not rebind a cohort to a disjoint selection");
        require(routes[1] == (block ? 4 : 1) && routes[block ? 0 : 2] == UINT16_MAX &&
                    residual_features[0] == (block ? 0 : 2),
                "shared feature selected the wrong route");
        require(layout::prepare_panel_packing_layout(source, blocks + block, 1, layout_storage,
                    &panel_layout) == Status::ok &&
                    layout::query_panel_packing(source, panel_layout, nullptr, 1, {}, &selection) == Status::ok &&
                    layout::prepare_panel_packing(selection, {}, borrowed_storage, &plan) == Status::ok,
                "could not prepare a disjoint layout after rejecting an overlap");
        require(routes[1] == (block ? 4 : 1) && routes[block ? 0 : 2] == UINT16_MAX &&
                    residual_features[0] == (block ? 0 : 2),
                "prepared layout retained stale columns for an overlapping catalog");
    }
}

struct PanelWorklists {
    static constexpr size_t logical_blocks = (features + 15)/16;
    static constexpr unsigned groups = 2, panels_per_group = 2, panel_count = 4, packed_panel_count = 3;
    matrix::TiledMatmulShape shape{output_tiles, groups, logical_blocks};
    uint32_t packed_offsets[groups + 1]{0, 5, 7};
    uint32_t panel_bases[groups]{0, 2};
    uint16_t packed_ids[7]{4, 0, 1, 2, 3, 4, 0};
    uint32_t pair_offsets[groups + 1]{0, 5, 10};
    matrix::IndexedTilePair pairs[10]{
        {3, 4}, {1, 2}, {0, 1}, {4, 0}, {2, 3},
        {4, 8}, {2, 11}, {1, 10}, {3, 12}, {0, 9}};

    void validate() const {
        const matrix::PackedPanelWorklist packed{
            {packed_offsets, groups + 1, {}}, {panel_bases, groups, {}}, {packed_ids, 7, {}}};
        const matrix::IndexedTileWorklist indexed{{pair_offsets, groups + 1, {}}, {pairs, 10, {}}};
        require(matrix::validate_indexed_panel_worklist(shape, packed, packed_panel_count) == cudaSuccess,
                "valid packed panel worklist rejected");
        require(matrix::validate_indexed_panel_worklist(shape, indexed, panel_count) == cudaSuccess,
                "valid arbitrary-slice worklist rejected");
        require(matrix::validate_indexed_panel_worklist(shape, packed, 2) == cudaErrorInvalidValue,
                "packed worklist accepted a cohort outside the resident span");
        require(matrix::validate_indexed_panel_worklist(shape, indexed, 3) == cudaErrorInvalidValue,
                "arbitrary-slice worklist accepted a slice outside the resident span");
    }
};

template<class T>
memory::const_array_view<T> upload_view(gpu::DeviceBuffer &buffer, const T *host, size_t count,
                                      const gpu::Stream &stream) {
    buffer.upload_async(host, count*sizeof(T), stream);
    return {static_cast<const T *>(buffer.data()), count,
            {memory::domain::device, int16_t(buffer.device())}};
}

void project(const Fixture &f, const PanelWorklists &work, int device) {
    const gpu::DeviceGuard guard(device);
    const gpu::Stream copy(device), compute(device);
    gpu::Event ready(device);
    const memory::placement where{memory::domain::device, int16_t(device)};
    const size_t panels = f.plan.shape.groups*f.plan.shape.panels;
    const size_t cells = f.plan.shape.groups*16, stride = cells + 4;
    require(panels == work.panel_count && f.plan.shape.groups == work.groups
                && f.plan.shape.panels == work.panels_per_group, "unexpected panel fixture shape");
    gpu::PinnedBuffer pinned(f.dense.size()*sizeof(uint16_t));
    memory::TiledBuffer<__half> feature_storage(1 + panels*memory::panel_tiles, device);
    const memory::PanelStagingSpan<__half> stage(feature_storage, 1, panels, pinned);
    std::vector<uint16_t> staged_residual(f.residual.size());
    require(layout::pack_panel_values(f.plan, f.values.data(), f.values.size(),
        static_cast<uint16_t *>(stage.host_data()), f.dense.size(), staged_residual.data(),
        staged_residual.size()) == Status::ok, "pack values directly into pinned panel span");
    gpu::DeviceBuffer w(f.weights.size()*sizeof(uint16_t), device);
    gpu::DeviceBuffer packed(f.packed_weights.size()*sizeof(uint16_t), device);
    gpu::DeviceBuffer offsets(f.residual_offsets.size()*sizeof(uint32_t), device);
    gpu::DeviceBuffer features_gpu(f.residual_features.size()*sizeof(uint16_t), device);
    gpu::DeviceBuffer residual_gpu(f.residual.size()*sizeof(uint16_t), device);
    std::vector<float> y(output_tiles*16*stride, sentinel);
    gpu::DeviceBuffer output(y.size()*sizeof(float), device);
    w.upload_async(f.weights.data(), w.bytes(), copy);
    packed.upload_async(f.packed_weights.data(), packed.bytes(), copy);
    offsets.upload_async(f.residual_offsets.data(), offsets.bytes(), copy);
    features_gpu.upload_async(f.residual_features.data(), features_gpu.bytes(), copy);
    residual_gpu.upload_async(staged_residual.data(), residual_gpu.bytes(), copy);
    stage.upload_async(panels, copy, ready);
    compute.wait(ready);

    const matrix::PanelProjectionResidualView residual{
        {static_cast<const uint32_t *>(offsets.data()), f.residual_offsets.size(), where},
        {static_cast<const uint16_t *>(features_gpu.data()), f.residual_features.size(), where},
        {static_cast<const __half *>(residual_gpu.data()), f.residual.size(), where}};
    const auto x = stage.view(panels);
    const auto reset_output = [&] {
        std::fill(y.begin(), y.end(), sentinel);
        output.upload_async(y.data(), output.bytes(), compute);
    };
    const auto verify_output = [&](size_t outputs, const char *message) {
        output.download_async(y.data(), output.bytes(), compute);
        compute.synchronize();
        for (size_t out = 0; out < outputs; ++out) {
            for (size_t row = 0; row < cells; ++row)
                require(y[out*stride + row] == f.reference(out, row), message);
            for (size_t row = cells; row < stride; ++row)
                require(y[out*stride + row] == sentinel, "projection touched stride padding");
        }
    };
    const auto weight_view = [&](size_t outputs) -> memory::const_matrix_view<__half> {
        return {static_cast<const __half *>(w.data()), outputs, features, features, where};
    };
    const auto output_view = [&](size_t outputs) -> memory::matrix_view<float> {
        return {static_cast<float *>(output.data()), outputs, cells, stride, where};
    };
    for (const size_t out_tiles : {size_t(1), output_tiles}) {
        reset_output();
        matrix::PanelMatmulPlan plan;
        gpu::check(matrix::prepare_panel_matmul(device, {f.plan.shape.groups, f.plan.shape.panels, out_tiles},
            plan), "prepare CUDA projection");
        const size_t outputs = out_tiles*16;
        gpu::check(matrix::enqueue_panel_projection(plan,
            {static_cast<const __half *>(packed.data()), f.packed_weights.size(), where}, x,
            residual, weight_view(outputs), output_view(outputs), compute.get()), "enqueue CUDA projection");
        verify_output(outputs, "panel and residual projection differs from CSR");
    }

    using Tile = memory::Tile<__half>;
    const size_t outputs = output_tiles*16;
    constexpr size_t logical_tiles = output_tiles*PanelWorklists::logical_blocks;
    constexpr size_t logical_values = logical_tiles*Tile::rows*Tile::columns;
    alignas(Tile) uint16_t logical_weights[logical_values];
    require(layout::pack_panel_source_weights(f.source, f.weights.data(), output_tiles, features,
                logical_weights, logical_values) == Status::ok, "pack logical weight bank");
    memory::TiledBuffer<__half> logical(logical_tiles, device);
    logical.upload_slots_async(logical_weights, logical_tiles, compute);
    const auto logical_view = memory::as_const(logical.view());
    matrix::IndexedTiledMatmulPlan indexed_plan;
    gpu::check(matrix::prepare_indexed_tiled_matmul(device, work.shape, indexed_plan),
               "prepare indexed panel projection");

    gpu::DeviceBuffer pair_offsets(sizeof(work.pair_offsets), device), pairs(sizeof(work.pairs), device);
    const matrix::IndexedTileWorklist arbitrary{
        upload_view(pair_offsets, work.pair_offsets, work.groups + 1, compute),
        upload_view(pairs, work.pairs, 10, compute)};
    reset_output();
    gpu::check(matrix::enqueue_indexed_panel_projection(indexed_plan, logical_view, x, arbitrary, residual,
        weight_view(outputs), output_view(outputs), compute.get()), "enqueue arbitrary-slice panel projection");
    verify_output(outputs, "arbitrary-slice panel projection differs from CSR");

    // The previous verification drained compute. Reuse the same pinned and device
    // spans for independent cohorts with five and two active blocks.
    uint32_t ragged_residual_offsets[PanelWorklists::groups*16 + 1]{};
    uint32_t ragged_panel_bases[PanelWorklists::groups]{};
    std::vector<uint16_t> ragged_residual_features;
    std::vector<uint16_t> ragged_residual_values;
    uint16_t block_slots[PanelWorklists::logical_blocks]{};
    const layout::PanelPackingWorkspace workspace{block_slots, work.logical_blocks};
    auto *pinned_values = static_cast<uint16_t *>(stage.host_data());
    size_t ragged_panels = 0;
    for (size_t group = 0; group < work.groups; ++group) {
        const size_t row_start = group*16, block_start = work.packed_offsets[group];
        const layout::PanelPackingInput input{f.rows + row_start, std::min(size_t(16), selected_rows - row_start),
            work.packed_ids + block_start, work.packed_offsets[group + 1] - block_start};
        layout::PanelPackingSelection selection;
        require(layout::query_panel_packing(f.source, input, workspace, &selection) == Status::ok,
                "query ragged panel cohort");
        const auto &req = selection.shape;
        require(req.groups == 1 && req.panels <= panels - ragged_panels, "ragged cohort exceeds staging span");

        uint32_t residual_offsets[17];
        const size_t residual_base = ragged_residual_values.size();
        ragged_residual_features.resize(residual_base + req.residual_nnz);
        ragged_residual_values.resize(residual_base + req.residual_nnz);
        const layout::PanelPackingMetadata metadata{nullptr, 0, residual_offsets, 17,
            ragged_residual_features.data() + residual_base, req.residual_nnz};
        require(layout::pack_panel_values_once(selection, workspace, metadata, f.values.data(), f.values.size(),
            pinned_values + ragged_panels*memory::panel_elements, req.dense_values,
            ragged_residual_values.data() + residual_base, req.residual_nnz) == Status::ok,
            "pack ragged cohort directly into pinned span");
        for (size_t row = 0; row < req.residual_offsets; ++row)
            ragged_residual_offsets[row_start + row] = uint32_t(residual_base + residual_offsets[row]);
        ragged_panel_bases[group] = uint32_t(ragged_panels);
        ragged_panels += req.panels;
    }
    require(ragged_panels == work.packed_panel_count, "ragged panel span contains unused panels");
    stage.upload_async(ragged_panels, copy, ready);
    compute.wait(ready);

    gpu::DeviceBuffer ragged_offsets(sizeof(ragged_residual_offsets), device);
    gpu::DeviceBuffer ragged_features(ragged_residual_features.size()*sizeof(uint16_t), device);
    gpu::DeviceBuffer ragged_values(ragged_residual_values.size()*sizeof(__half), device);
    ragged_values.upload_async(ragged_residual_values.data(), ragged_values.bytes(), compute);
    const matrix::PanelProjectionResidualView ragged_residual{
        upload_view(ragged_offsets, ragged_residual_offsets, cells + 1, compute),
        upload_view(ragged_features, ragged_residual_features.data(), ragged_residual_features.size(), compute),
        {static_cast<const __half *>(ragged_values.data()), ragged_residual_values.size(), where}};
    gpu::DeviceBuffer packed_offsets(sizeof(work.packed_offsets), device), panel_bases(sizeof(work.panel_bases), device);
    gpu::DeviceBuffer packed_ids(sizeof(work.packed_ids), device);
    const matrix::PackedPanelWorklist packed_order{
        upload_view(packed_offsets, work.packed_offsets, work.groups + 1, compute),
        upload_view(panel_bases, ragged_panel_bases, work.groups, compute),
        upload_view(packed_ids, work.packed_ids, 7, compute)};
    reset_output();
    gpu::check(matrix::enqueue_indexed_panel_projection(indexed_plan, logical_view, stage.view(ragged_panels),
        packed_order, ragged_residual, weight_view(outputs), output_view(outputs), compute.get()),
        "enqueue packed-order panel projection");
    verify_output(outputs, "packed-order panel projection differs from CSR");
}

void project_indexed(int device) {
    using Tile = memory::Tile<__half>;
    const gpu::DeviceGuard guard(device);
    const gpu::Stream stream(device);
    const memory::placement where{memory::domain::device, int16_t(device)};
    const matrix::TiledMatmulShape shape{1, 2, 2};
    const uint32_t host_offsets[]{0, 2, 2};
    const matrix::IndexedTilePair host_pairs[]{{1, 4}, {0, 1}};
    require(matrix::validate_indexed_tile_worklist(shape,
        {{host_offsets, 3, {}}, {host_pairs, 2, {}}}, 5) == cudaSuccess, "indexed worklist");
    Tile weight[2]{}, activation[2]{};
    for (unsigned i = 0; i < 16; ++i) {
        weight[0].data()[17*i] = __float2half(1.0f);
        weight[1].data()[17*i] = __float2half(2.0f);
    }
    for (unsigned i = 0; i < 256; ++i) {
        activation[0].data()[i] = __float2half(float(i));
        activation[1].data()[i] = __float2half(2.0f);
    }
    gpu::DeviceBuffer offsets(sizeof(host_offsets), device), pairs(sizeof(host_pairs), device);
    memory::TiledBuffer<__half> w(2, device), x(5, device);
    float values[16*32];
    gpu::DeviceBuffer y(sizeof(values), device);
    offsets.upload_async(host_offsets, sizeof(host_offsets), stream);
    pairs.upload_async(host_pairs, sizeof(host_pairs), stream);
    w.upload_async(weight, 2, stream);
    x.upload_async(activation, 1, stream, 1);
    x.upload_async(activation + 1, 1, stream, 4);
    matrix::IndexedTiledMatmulPlan plan;
    gpu::check(matrix::prepare_indexed_tiled_matmul(device, shape, plan), "prepare indexed projection");
    gpu::check(matrix::enqueue_indexed_tiled_matmul(plan, memory::as_const(w.view()),
        memory::as_const(x.view()), {{static_cast<const uint32_t *>(offsets.data()), 3, where},
        {static_cast<const matrix::IndexedTilePair *>(pairs.data()), 2, where}},
        {static_cast<float *>(y.data()), 16, 32, 32, where}, stream.get()), "indexed projection");
    y.download_async(values, sizeof(values), stream);
    stream.synchronize();
    for (unsigned out = 0; out < 16; ++out)
        for (unsigned cell = 0; cell < 32; ++cell)
            require(values[out*32 + cell] == (cell < 16 ? float(16*out + cell + 4) : 0.0f),
                    "indexed projection chose the wrong block or slot");
}

} // namespace

int main(int argc, char **argv) try {
    Fixture fixture;
    fixture.prepare();
    fixture.verify_packing();
    fixture.verify_shared_layout();
    fixture.verify_refresh();
    fixture.verify_scatter();
    verify_wide_routes();
    verify_overlapping_blocks();
    PanelWorklists work;
    work.validate();
    if (argc == 2 && std::strcmp(argv[1], "--host-only") == 0) return 0;
    const int device = find_volta();
    if (device < 0) return 77;
    project(fixture, work, device);
    project_indexed(device);
} catch (const std::exception &error) {
    std::fprintf(stderr, "panel projection test: %s\n", error.what());
    return 1;
}
