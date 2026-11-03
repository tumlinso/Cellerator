#include "Cellerator/tensor/block_masked/structure.hh"
#include "Cellerator/tensor/block_masked/view.hh"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using cellerator::tensor::block_masked::LayoutView;
using cellerator::tensor::block_masked::LoadStatus;
using cellerator::tensor::block_masked::Structure;
using cellerator::tensor::block_masked::View;
using cellerator::tensor::block_masked::try_load;
using cellerator::tensor::block_masked::validate;
using cellerator::tensor::block_masked::value_offset;

void require(bool condition, const char *message) {
    if (!condition) {
        std::cerr << "block_masked_test: " << message << '\n';
        std::exit(1);
    }
}

template<class Function>
void require_throws(Function &&function, const char *message) {
    bool threw = false;
    try {
        function();
    } catch (const std::exception &) {
        threw = true;
    }
    require(threw, message);
}

void test_lookup_and_fixed_support() {
    const std::vector<std::vector<std::uint32_t>> support{
        {12, 5, 4, 0}, {5}, {}, {6, 2, 1}, {12, 11}};
    Structure structure = Structure::from_rows(
        13,
        support,
        5,
        3,
        {100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112},
        {900, 904, 906, 910, 911});
    validate(structure.view());

    require(structure.rows() == 5, "row count");
    require(structure.features() == 13, "feature count");
    require(structure.tiles() == 2, "tile count");
    require(structure.feature_blocks() == 3, "feature block count");
    require(structure.tile_blocks() == 6, "tile block count");
    require(structure.row_block_entries() == 7, "row block entry count");
    require(structure.stored_values() == 10, "allocated support count");
    require(structure.physical_feature(112) == 12, "canonical feature binding");
    require(structure.canonical_rows()[3] == 910, "canonical row binding");
    require_throws([&] { (void)structure.physical_feature(999); }, "unknown canonical feature throws");

    std::vector<float> values(structure.stored_values());
    for (std::size_t i = 0; i < values.size(); ++i) values[i] = static_cast<float>(i) + 0.5f;
    values[4] = 0.0f; // A numerical zero remains an allocated, addressable slot.
    const View<const float> view{structure.view(), values.data()};

    std::uint64_t offset = 999;
    require(value_offset(view.layout, 0, 12, &offset) && offset == 4,
        "lookup in the final feature block");
    require(structure.value_offset(1, 5) == 3, "row rank inside a block descriptor");
    require(structure.value_offset(3, 1) == 5, "lookup in the next row tile");
    require(structure.value_offset(4, 12) == 9, "lookup in a partial feature block");
    const auto stored_zero = try_load(view, 0, 12);
    require(stored_zero.status == LoadStatus::stored && stored_zero.has_value() &&
        stored_zero.value == 0.0f,
        "stored numerical zero is distinct from omitted support");

    std::vector<bool> seen(values.size(), false);
    for (std::uint64_t row = 0; row < structure.rows(); ++row) {
        for (std::uint64_t feature = 0; feature < structure.features(); ++feature) {
            const bool expected = std::find(
                support[row].begin(), support[row].end(), feature) != support[row].end();
            std::uint64_t found_offset = 0;
            const bool found = value_offset(view.layout, row, feature, &found_offset);
            require(found == expected, "every coordinate lookup matches declared support");
            if (found) {
                require(found_offset < seen.size(), "lookup offset remains in the value allocation");
                require(!seen[found_offset], "each allocated coordinate has one unique offset");
                seen[found_offset] = true;
            }
        }
    }
    require(std::all_of(seen.begin(), seen.end(), [](bool value) { return value; }),
        "support maps densely onto every stored value slot");

    offset = 999;
    require(!value_offset(view.layout, 0, 6, &offset) && offset == 0,
        "an unallocated coordinate is absent");
    require(try_load(view, 0, 6).status == LoadStatus::absent,
        "a valid unallocated coordinate is reported absent");
    require(!value_offset(view.layout, 5, 0, &offset), "out-of-range row is absent");
    require(!value_offset(view.layout, 0, 13, &offset), "out-of-range feature is absent");
    require(try_load(view, 5, 0).status == LoadStatus::out_of_bounds,
        "out-of-range reads have an explicit status");
    require(!value_offset(view.layout, 0, 0, nullptr), "null output is rejected");
    require_throws([&] { (void)structure.value_offset(0, 6); },
        "checked host lookup rejects unallocated coordinates");

    const View<const float> missing_values{structure.view(), nullptr};
    require(try_load(missing_values, 0, 12).status == LoadStatus::invalid_storage,
        "a present slot with no value allocation is a storage error");
    require(try_load(missing_values, 0, 6).status == LoadStatus::absent,
        "absent support remains distinguishable from missing value storage");
    require(try_load(missing_values, 5, 0).status == LoadStatus::out_of_bounds,
        "out-of-range remains distinct when value storage is missing");

    auto malformed_layout = structure.view();
    malformed_layout.tile_row_width = 0;
    const View<const float> malformed_view{malformed_layout, values.data()};
    require(try_load(malformed_view, 0, 12).status == LoadStatus::invalid_layout,
        "malformed metadata is not reported as absent support");
}

void test_empty_extents_and_zero_support() {
    const Structure empty = Structure::from_rows(0, {});
    validate(empty.view());
    require(empty.rows() == 0 && empty.features() == 0 && empty.stored_values() == 0,
        "empty structure is valid");

    const Structure empty_rows = Structure::from_rows(9, {});
    validate(empty_rows.view());
    require(empty_rows.rows() == 0 && empty_rows.features() == 9,
        "zero rows retain the declared feature extent");
    require(empty_rows.physical_feature(8) == 8,
        "identity canonical maps use direct coordinate lookup");
    require_throws([&] { (void)empty_rows.physical_feature(9); },
        "identity lookup rejects coordinates beyond the feature extent");

    const Structure zero_features = Structure::from_rows(0, {{}, {}});
    validate(zero_features.view());
    require(zero_features.rows() == 2 && zero_features.features() == 0 &&
        zero_features.tiles() == 1 && zero_features.tile_blocks() == 0,
        "zero features retain valid row tiles");

    const Structure empty_support = Structure::from_rows(7, {{}, {}, {}}, 4, 2);
    validate(empty_support.view());
    require(empty_support.rows() == 3 && empty_support.tiles() == 2 &&
        empty_support.tile_blocks() == 0 && empty_support.row_block_entries() == 0 &&
        empty_support.stored_values() == 0,
        "rows with no allocated support retain valid empty tiles");
    const View<const float> view{empty_support.view(), nullptr};
    require(try_load(view, 2, 6).status == LoadStatus::absent,
        "empty support is reported absent without requiring value storage");

    LayoutView<std::uint8_t> narrow_empty;
    const std::uint8_t terminal[] = {0};
    narrow_empty.tile_block_offsets = terminal;
    narrow_empty.block_row_entry_offsets = terminal;
    narrow_empty.row_block_value_offsets = terminal;
    narrow_empty.feature_block_offsets = terminal;
    validate(narrow_empty);
}

void test_rejects_invalid_support_and_maps() {
    require_throws(
        [] { (void)Structure::from_rows(3, {{1, 1}}); },
        "duplicate row feature is rejected");
    require_throws(
        [] { (void)Structure::from_rows(3, {{3}}); },
        "out-of-range row feature is rejected");
    require_throws(
        [] { (void)Structure::from_rows(3, {{}}, 0); },
        "zero block width is rejected");
    require_throws(
        [] { (void)Structure::from_rows(3, {{}}, 32, 33); },
        "tile widths beyond the 32-bit grammar are rejected");
    require_throws(
        [] { (void)Structure::from_rows(std::numeric_limits<std::uint64_t>::max(), {}); },
        "feature extent outside default offset range is rejected");
    require_throws(
        [] { (void)Structure::from_rows(2, {{}}, 32, 32, {9, 9}); },
        "duplicate canonical feature coordinates are rejected");
    require_throws(
        [] { (void)Structure::from_rows(2, {{}, {}}, 32, 32, {}, {4, 4}); },
        "duplicate canonical row coordinates are rejected");
}

void test_layout_validation_catches_corruption() {
    const Structure structure = Structure::from_rows(8, {{0, 4}, {4}, {7}}, 4, 3);
    const auto good = structure.view();
    validate(good);

    auto row_masks = std::vector<std::uint32_t>(
        good.participating_row_masks,
        good.participating_row_masks + good.tile_blocks);
    row_masks[0] |= std::uint32_t{1} << 31;
    auto bad_row_mask = good;
    bad_row_mask.participating_row_masks = row_masks.data();
    require_throws([&] { validate(bad_row_mask); }, "invalid row mask is rejected");

    auto block_ids = std::vector<std::uint32_t>(good.block_ids, good.block_ids + good.tile_blocks);
    block_ids[0] = 1;
    auto bad_order = good;
    bad_order.block_ids = block_ids.data();
    require_throws([&] { validate(bad_order); }, "unsorted descriptor ids are rejected");

    auto feature_masks = std::vector<std::uint32_t>(
        good.feature_masks,
        good.feature_masks + good.row_block_entries);
    feature_masks[0] = 0;
    auto bad_feature_mask = good;
    bad_feature_mask.feature_masks = feature_masks.data();
    require_throws([&] { validate(bad_feature_mask); }, "empty feature mask is rejected");

    auto values = std::vector<std::uint32_t>(
        good.row_block_value_offsets,
        good.row_block_value_offsets + good.row_block_entries + 1);
    values.back() -= 1;
    auto bad_terminal = good;
    bad_terminal.row_block_value_offsets = values.data();
    require_throws([&] { validate(bad_terminal); }, "incorrect terminal value offset is rejected");
}

void test_large_permuted_canonical_map() {
    constexpr std::uint64_t feature_count = 20000;
    std::vector<std::uint64_t> canonical(feature_count);
    for (std::uint64_t physical = 0; physical < feature_count; ++physical) {
        // Reverse physical order and leave gaps between IDs to exercise a
        // sparse canonical namespace without relying on a random fixture.
        canonical[physical] = 1000000007ull + (feature_count - physical - 1) * 17ull;
    }

    const auto prepare_begin = std::chrono::steady_clock::now();
    const Structure structure = Structure::from_rows(
        feature_count, {}, 32, 32, std::move(canonical));
    const auto prepare_end = std::chrono::steady_clock::now();

    std::uint64_t checksum = 0;
    const auto lookup_begin = std::chrono::steady_clock::now();
    for (std::uint64_t physical = 0; physical < feature_count; ++physical) {
        checksum += structure.physical_feature(structure.canonical_features()[physical]);
    }
    const auto lookup_end = std::chrono::steady_clock::now();
    const std::uint64_t expected_checksum =
        feature_count * (feature_count - 1) / 2;
    require(checksum == expected_checksum,
        "large permuted sparse canonical IDs map back to every physical coordinate");
    require_throws([&] { (void)structure.physical_feature(7); },
        "unknown sparse canonical ID is rejected");

    const auto prepare_ms = std::chrono::duration<double, std::milli>(
        prepare_end - prepare_begin).count();
    const auto lookup_ms = std::chrono::duration<double, std::milli>(
        lookup_end - lookup_begin).count();
    std::cout << "block_masked_test: canonical map diagnostic (N=" << feature_count
              << "): prepare_ms=" << prepare_ms
              << ", lookup_20k_ms=" << lookup_ms << '\n';
}

} // namespace

int main() {
    static_assert(std::is_trivially_copyable<LayoutView<>>::value, "layout view ABI");
    static_assert(std::is_trivially_copyable<View<const float>>::value, "value view ABI");
    static_assert(!std::is_convertible<
        cellerator::tensor::block_masked::LoadResult<float>, float>::value,
        "load outcomes must not implicitly become values");
    test_lookup_and_fixed_support();
    test_empty_extents_and_zero_support();
    test_rejects_invalid_support_and_maps();
    test_layout_validation_catches_corruption();
    test_large_permuted_canonical_map();
    std::cout << "block_masked_test: PASS\n";
}
