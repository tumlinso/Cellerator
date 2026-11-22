#include <Cellerator/layout/projection/feature_major_tiles.hh>

#include <bit>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace project = cellerator::layout::projection;
namespace tensor = cellerator::tensor;
using Status = project::FeatureMajorTilesStatus;
using Grouping = project::FeatureMajorTileGrouping;

static void check(bool condition, int line) {
    if (condition) return;
    std::fprintf(stderr, "feature-major tile host check failed at line %d\n", line);
    std::exit(1);
}
#define CHECK(condition) check((condition), __LINE__)

static void check_layout(project::FeatureMajorTilesCsr csr, Grouping grouping,
                         const uint32_t *expected_rows = nullptr) {
    size_t count = 0;
    CHECK(project::feature_major_tiles_workspace_count(csr, grouping, &count) == Status::ok);
    std::vector<uint32_t> scratch(count);
    project::FeatureMajorTilesRequirements required{};
    CHECK(project::query_feature_major_tiles(
        csr, grouping, scratch.data(), scratch.size(), &required) == Status::ok);

    std::vector<uint32_t> indices(required.index_count, UINT32_MAX);
    std::vector<project::FeatureMajorTileRange> chunks(required.chunk_count);
    CHECK(project::prepare_feature_major_tiles(csr, required, scratch.data(), scratch.size(),
        {indices.data(), indices.size() - 1, chunks.data(), chunks.size()})
        == Status::insufficient_capacity);
    for (uint32_t word : indices) CHECK(word == UINT32_MAX);
    CHECK(project::prepare_feature_major_tiles(csr, required, scratch.data(), scratch.size(),
        {indices.data(), indices.size(), chunks.data(), chunks.size()}) == Status::ok);

    tensor::FeatureMajorTileStructureView structure{};
    project::FeatureMajorTilesIndexView index{};
    CHECK(project::bind_feature_major_tiles(required, indices.data(), indices.size(),
        chunks.data(), chunks.size(), &structure, &index) == Status::ok);
    CHECK(structure.rows == csr.rows && structure.features == csr.cols);
    CHECK(structure.tile_count == (csr.rows + 31) / 32 && structure.nnz == csr.nnz);
    CHECK((index.logical_rows != nullptr) == (grouping == Grouping::local_feature_hash));

    std::vector<bool> rows(csr.rows), positions(csr.nnz), reversed(structure.record_count);
    bool changed_order = false;
    for (uint32_t physical = 0; physical < structure.rows; ++physical) {
        uint32_t logical = index.logical_rows ? index.logical_rows[physical] : physical;
        CHECK(logical < csr.rows && !rows[logical]);
        CHECK(logical / 128 == physical / 128);
        if (expected_rows) CHECK(logical == expected_rows[physical]);
        rows[logical] = true;
        if (logical != physical) changed_order = true;
    }
    if (csr.rows > 32 && index.logical_rows) CHECK(changed_order);

    uint32_t value = 0;
    for (uint32_t tile = 0; tile < structure.tile_count; ++tile) {
        uint32_t begin = structure.tile_offsets[tile], end = structure.tile_offsets[tile + 1];
        CHECK(begin <= end && end <= structure.record_count);
        for (uint32_t p = begin; p < end; ++p) {
            uint32_t feature = structure.feature_ids[p], mask = structure.row_masks[p];
            CHECK(feature < csr.cols && (p == begin || structure.feature_ids[p - 1] < feature));
            CHECK(mask && structure.value_offsets[p] == value);
            for (; mask; mask &= mask - 1) {
                uint32_t physical = 32 * tile + std::countr_zero(mask);
                CHECK(physical < csr.rows && value < structure.nnz);
                uint32_t logical = index.logical_rows ? index.logical_rows[physical] : physical;
                uint32_t position = index.source_value_positions[value++];
                CHECK(position >= csr.major_ptr[logical] && position < csr.major_ptr[logical + 1]);
                CHECK(csr.minor_idx[position] == feature && !positions[position]);
                positions[position] = true;
            }
        }
    }
    CHECK(structure.tile_offsets[structure.tile_count] == structure.record_count);
    CHECK(value == csr.nnz);

    for (uint32_t feature = 0; feature < structure.features; ++feature) {
        uint32_t start = index.feature_offsets[feature], stop = index.feature_offsets[feature + 1];
        CHECK(start <= stop && stop <= structure.record_count);
        uint32_t edges = 0;
        for (uint32_t p = start; p < stop; ++p) {
            uint32_t record = index.record_ids[p], tile = index.tile_ids[p];
            CHECK(tile < structure.tile_count && (p == start || tile > index.tile_ids[p - 1]));
            CHECK(record >= structure.tile_offsets[tile] && record < structure.tile_offsets[tile + 1]);
            CHECK(structure.feature_ids[record] == feature && !reversed[record]);
            reversed[record] = true;
            edges += std::popcount(structure.row_masks[record]);
        }
        uint32_t first = index.feature_chunk_offsets ? index.feature_chunk_offsets[feature] : 0;
        uint32_t end = index.feature_chunk_offsets ? index.feature_chunk_offsets[feature + 1] : 0;
        CHECK((edges > 256) == (first < end) && end <= index.chunk_count);
        for (uint32_t p = first; p < end; ++p) {
            auto chunk = index.chunks[p];
            CHECK(chunk.feature == feature && chunk.begin == start && chunk.end <= stop);
            CHECK(chunk.end > chunk.begin);
            uint32_t size = 0;
            for (uint32_t i = chunk.begin; i < chunk.end; ++i)
                size += std::popcount(structure.row_masks[index.record_ids[i]]);
            CHECK(size > 0 && size <= 256);
            start = chunk.end;
        }
        if (first < end) CHECK(start == stop);
    }
    for (bool visited : reversed) CHECK(visited);
}

static void check_stable_groups() {
    constexpr uint32_t rows = 135;
    std::vector<uint32_t> offsets{0}, columns, expected;
    for (uint32_t row = 0; row < rows; ++row) {
        if (row % 3 != 1) columns.push_back(uint32_t(row % 3 == 0));
        offsets.push_back(uint32_t(columns.size()));
    }
    // Feature 0 hashes below feature 1. Empty rows come last in each window.
    const uint32_t classes[] = {2, 0, 1};
    for (uint32_t begin = 0; begin < rows; begin += 128) {
        for (uint32_t kind : classes) {
            for (uint32_t row = begin; row < begin + 128 && row < rows; ++row)
                if (row % 3 == kind) expected.push_back(row);
        }
    }
    CHECK(expected.size() == rows);
    check_layout({rows, 2, columns.size(), tensor::base::CompressedAxis::row,
                  offsets.data(), columns.data()}, Grouping::local_feature_hash, expected.data());
}

int main() {
    using tensor::base::CompressedAxis;
    std::vector<uint32_t> offsets{0}, columns;
    constexpr uint32_t rows = 547;
    for (uint32_t row = 0; row < rows; ++row) {
        if (row % 11 != 2) {
            columns.push_back(row % 13 + 2);
            if (row % 3 != 1) columns.push_back(1);
        }
        offsets.push_back(uint32_t(columns.size()));
    }
    project::FeatureMajorTilesCsr csr{rows, 16, columns.size(), CompressedAxis::row,
                                      offsets.data(), columns.data()};
    check_layout(csr, Grouping::contiguous);
    check_layout(csr, Grouping::local_feature_hash);
    CHECK(offsets[0] == 0 && offsets[1] == 2);

    std::vector<uint32_t> scratch;
    size_t count = 0;
    project::FeatureMajorTilesRequirements required{};
    CHECK(project::feature_major_tiles_workspace_count(csr, Grouping::contiguous, &count) == Status::ok);
    scratch.resize(count);
    columns[0] = columns[1];
    CHECK(project::query_feature_major_tiles(csr, Grouping::contiguous, scratch.data(),
        scratch.size(), &required) == Status::duplicate_coordinate);
    columns[0] = 16;
    CHECK(project::query_feature_major_tiles(csr, Grouping::contiguous, scratch.data(),
        scratch.size(), &required) == Status::invalid_csr);
    // A wide feature axis with few occupied features exercises sparse emission.
    for (size_t p = 0; p < columns.size(); ++p) columns[p] = uint32_t((131 * p) % 8191);
    csr.cols = 8192;
    check_layout(csr, Grouping::local_feature_hash);
    uint32_t empty_offsets[2]{};
    check_layout({0, 0, 0, CompressedAxis::row, empty_offsets, nullptr}, Grouping::contiguous);
    check_layout({1, 3, 0, CompressedAxis::row, empty_offsets, nullptr}, Grouping::local_feature_hash);
    check_stable_groups();
}
