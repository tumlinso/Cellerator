#include <Cellerator/layout/sliced_ell_pack.hh>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>

namespace {

using namespace cellerator;
using namespace cellerator::layout;
using memory::status;
using tensor::base::CompressedAxis;
using tensor::base::CompressedView;
using tensor::base::device::SlicedEllView;

void check(bool ok, const char *where) {
    if (ok) return;
    std::cerr << "SELL pack check failed: " << where << '\n';
    std::abort();
}

void pack_cells() {
    constexpr int32_t offsets[] = {0, 1, 4, 5, 7, 7, 9, 11};
    constexpr int64_t indices[] = {0, 4, 1, 4, 1, 2, 4, 0, 1, 2, 3};
    const double val[] = {-0., 1.00048828125, 0., 1.00146484375, 2., 3., 4., -1., 5., 6., 7.};
    constexpr uint16_t expected_bits[] = {0x8000, 0x3c00, 0, 0x3c02, 0x4000, 0x4200,
                                          0x4400, 0xbc00, 0x4500, 0x4600, 0x4700};
    constexpr SlicedEllRow logical[] = {{0,9}, {1,4}, {2,9}, {3,18}, {4,4}, {5,33}, {6,13}};
    constexpr unsigned expected_order[] = {1, 3, 0, 2, 5, 6, 4};
    constexpr unsigned expected_widths[] = {3, 1, 2, 0};
    constexpr unsigned expected_starts[] = {0, 6, 8, 12};
    unsigned lengths[7], order[7];
    SlicedEllSlice slices[4];
    SlicedEllRow map[7];
    SlicedEllPackPlan plan;
    check(sliced_ell_row_lengths(offsets, 7, 11, lengths, 7) == status::success, "derived lengths");
    check(plan_sliced_ell(lengths, 7, 6, {2, 4, true, true}, order, 7, slices, 4, &plan) == status::success,
          "plan bounded sorting");
    check(plan.shape.nnz == 11 && plan.shape.slots == 12 && plan.shape.slice_count == 4, "exact layout size");
    for (size_t i = 0; i < 7; ++i) check(order[i] == expected_order[i], "stable bounded order");

    alignas(32) unsigned char csr_bytes[256], csc_bytes[256];
    std::memset(csr_bytes, 0xa5, sizeof(csr_bytes));
    std::memset(csc_bytes, 0x5a, sizeof(csc_bytes));
    const CompressedView<int64_t, const double, int32_t> csr{{7, 6, 11, CompressedAxis::row, offsets, indices}, val};
    const CompressedView<int64_t, const double, int32_t> csc{{6, 7, 11, CompressedAxis::column, offsets, indices}, val};
    check(pack_sliced_ell(plan, csr, logical, csr_bytes, sizeof(csr_bytes), map, 7) == status::success, "pack CSR");
    check(pack_sliced_ell(plan, csc, logical, csc_bytes, sizeof(csc_bytes), map, 7) == status::success, "pack CSC by cell");
    check(std::memcmp(csr_bytes, csc_bytes, plan.layout.bytes()) == 0, "same byte packet for both orientations");

    SlicedEllView view{};
    check(plan.layout.bind(csr_bytes, sizeof(csr_bytes), &view) == status::success, "bind packet");
    check(view.slice_row_offsets[4] == 7 && view.nnz == 11, "partial terminal slice and explicit zero count");
    for (unsigned i = 0; i < 4; ++i) {
        check(view.slice_widths[i] == expected_widths[i], "slice width");
        check(view.slice_slot_offsets[i] == expected_starts[i], "slice offset");
        check(view.slice_row_offsets[i] == 2*i, "slice row start");
        for (unsigned p = view.slice_row_offsets[i]; p < view.slice_row_offsets[i+1]; ++p) {
            const unsigned src = expected_order[p];
            check(map[p].logical_position == logical[src].logical_position && map[p].source_cell == logical[src].source_cell,
                  "logical occurrence and source map");
            const unsigned row_start = view.slice_slot_offsets[i] + (p-view.slice_row_offsets[i])*view.slice_widths[i];
            const unsigned length = unsigned(offsets[src+1]-offsets[src]);
            for (unsigned j = 0; j < view.slice_widths[i]; ++j) {
                const unsigned k = row_start+j;
                const unsigned ref = unsigned(offsets[src])+j;
                const unsigned expected_col = j < length ? unsigned(indices[ref]) : unsigned(tensor::base::sliced_ell_invalid_col);
                const uint16_t expected_val = j < length ? expected_bits[ref] : 0;
                check(view.col_idx[k] == expected_col, "entry order, duplicates and padding columns");
                check(__builtin_bit_cast(uint16_t, view.val[k]) == expected_val, "entry bits, ties, and positive padding zero");
            }
        }
    }
    check(csr_bytes[plan.layout.bytes()] == 0xa5, "adjacent packet untouched");

    check(pack_sliced_ell(plan, csr, logical, csr_bytes, plan.layout.bytes()-1, map, 7) == status::capacity_exceeded,
          "short packet");
    check(pack_sliced_ell(plan, csr, logical, csr_bytes, sizeof(csr_bytes), map, 6) == status::capacity_exceeded, "short map");
    check(pack_sliced_ell(plan, csr, logical, csr_bytes+1, sizeof(csr_bytes)-1, map, 7) == status::invalid_alignment,
          "unaligned packet");
    const auto old_bytes = plan.layout.bytes();
    check(plan_sliced_ell(lengths, 7, 6, {2, 4, true, true}, order, 6, slices, 4, &plan) == status::capacity_exceeded,
          "short order");
    check(plan.layout.bytes() == old_bytes, "failed plan leaves descriptor unchanged");
}

void precision_and_empty() {
    constexpr unsigned lengths[] = {1, 0};
    constexpr unsigned indices[] = {5};
    constexpr int32_t offsets[] = {0, 1, 1};
    constexpr SlicedEllRow rows[] = {{17, 22}, {18, 23}};
    alignas(32) unsigned char packet[128];
    unsigned order[2];
    SlicedEllSlice slices[1];
    SlicedEllRow map[2];
    SlicedEllPackPlan plan;
    check(plan_sliced_ell(lengths, 2, 8, {}, order, 2, slices, 1, &plan) == status::success, "default plan");
    check(plan.shape.slice_rows == 16 && order[0] == 0 && order[1] == 1, "defaults keep order");
    double value[] = {65520.};
    const CompressedView<unsigned, double, int32_t> wide{{2, 8, 1, CompressedAxis::row, offsets, indices}, value};
    check(pack_sliced_ell(plan, wide, rows, packet, sizeof(packet), map, 2) == status::invalid_argument, "FP16 opt-in");
    check(plan_sliced_ell(lengths, 2, 8, {.allow_fp16_conversion = true}, order, 2, slices, 1, &plan) == status::success,
          "consented plan");
    check(pack_sliced_ell(plan, wide, rows, packet, sizeof(packet), map, 2) == status::arithmetic_overflow,
          "finite limit ties to infinity");
    // One binary64 ulp above a binary16 halfway value still rounds to the
    // midpoint in binary32. The packer must round from binary64 directly.
    value[0] = 1.00048828125 + 0x1p-52;
    check(pack_sliced_ell(plan, wide, rows, packet, sizeof(packet), map, 2) == status::success, "refill with changed values");
    SlicedEllView view{};
    check(plan.layout.bind(packet, sizeof(packet), &view) == status::success &&
          __builtin_bit_cast(uint16_t, view.val[0]) == 0x3c01, "binary64 rounded without binary32 staging");

    const real::f16_t half = __builtin_bit_cast(real::f16_t, uint16_t(0xfe0a));
    const CompressedView<unsigned, const real::f16_t, int32_t> same{{2, 8, 1, CompressedAxis::row, offsets, indices}, &half};
    check(plan_sliced_ell(lengths, 2, 8, {}, order, 2, slices, 1, &plan) == status::success, "native FP16 plan");
    check(pack_sliced_ell(plan, same, rows, packet, sizeof(packet), map, 2) == status::success, "native FP16 no conversion");
    check(plan.layout.bind(packet, sizeof(packet), &view) == status::success, "bind native FP16");
    check(__builtin_bit_cast(uint16_t, view.val[0]) == 0xfe0a && __builtin_bit_cast(uint16_t, view.val[1]) == 0,
          "native FP16 NaN bits retained, pad is positive zero");

    constexpr unsigned zero[] = {0, 0};
    constexpr int32_t no_entries[] = {0, 0, 0};
    const CompressedView<unsigned, real::f16_t, int32_t> empty_rows{{2, 8, 0, CompressedAxis::row, no_entries, nullptr}, nullptr};
    check(plan_sliced_ell(zero, 2, 8, {}, order, 2, slices, 1, &plan) == status::success, "empty rows plan");
    check(pack_sliced_ell(plan, empty_rows, rows, packet, sizeof(packet), map, 2) == status::success, "empty rows packing");
    check(plan.layout.bind(packet, sizeof(packet), &view) == status::success && !view.val && !view.col_idx,
          "empty rows have no entry planes");
    check(view.slice_widths[0] == 0 && view.slice_row_offsets[1] == 2, "empty rows metadata");
    const CompressedView<unsigned, const real::f16_t, int32_t> empty{{0, 8, 0, CompressedAxis::row, nullptr, nullptr}, nullptr};
    check(plan_sliced_ell(nullptr, 0, 8, {}, nullptr, 0, nullptr, 0, &plan) == status::success, "empty batch plan");
    check(pack_sliced_ell(plan, empty, nullptr, packet, sizeof(packet), nullptr, 0) == status::success, "empty batch packing");
    check(plan.layout.bind(packet, sizeof(packet), &view) == status::success && view.slice_count == 0 && view.slice_row_offsets[0] == 0,
          "empty batch terminal offset");

    const int32_t invalid[] = {0, -1, 1};
    check(sliced_ell_row_lengths(invalid, 2, 1, order, 2) == status::invalid_argument, "negative offsets");
    const CompressedView<int, const real::f16_t, int32_t> negative_index{{2, 8, 1, CompressedAxis::row, offsets, invalid+1}, &half};
    check(plan_sliced_ell(lengths, 2, 8, {}, order, 2, slices, 1, &plan) == status::success, "index test plan");
    check(pack_sliced_ell(plan, negative_index, rows, packet, sizeof(packet), map, 2) == status::invalid_argument,
          "negative index");
}

void sorting_and_width_limits() {
    constexpr unsigned n = 517, window = 256;
    unsigned lengths[n], order[n], reference[n];
    SlicedEllSlice slices[33];
    SlicedEllPackPlan plan;
    for (unsigned i = 0; i < n; ++i) {
        lengths[i] = (271*i+13)%73;
        reference[i] = i;
    }
    for (unsigned i = 0; i < n; i += window) {
        std::stable_sort(reference+i, reference + std::min(i+window,n), [&](unsigned a, unsigned b) {
            return lengths[a] > lengths[b];
        });
    }
    check(plan_sliced_ell(lengths, n, 100, {.reorder_rows = true}, order, n, slices, 33, &plan) == status::success,
          "default sorting window");
    check(std::equal(order, order+n, reference), "bounded sorting compared with stable reference");

    constexpr unsigned oversized_nnz[] = {UINT32_MAX, 1};
    check(plan_sliced_ell(oversized_nnz, 2, 2, {}, order, n, slices, 33, &plan) == status::arithmetic_overflow,
          "stored-entry sum overflow");
    constexpr unsigned oversized_base[] = {UINT32_MAX, 0, 0};
    check(plan_sliced_ell(oversized_base, 3, 2, {.slice_rows = 2}, order, n, slices, 33, &plan) == status::arithmetic_overflow,
          "padded slice base overflow");
}

} // namespace

int main() {
    pack_cells();
    precision_and_empty();
    sorting_and_width_limits();
}
