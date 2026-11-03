#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <type_traits>
#include <vector>

#include <Cellerator/interop/formats/affine_blocked_ell_view.hh>
#include <Cellerator/ops/quantized/api.cuh>

namespace tq = ::cellerator::tensor::quantized;
namespace legacy = ::cellerator::interop::formats;

namespace {

template<int Bits, typename Metadata>
bool test_csr_round_trip(Metadata metadata, const std::vector<float>& expected_values) {
    const int rows = 3;
    const int cols = 7;
    const int nnz = 7;
    const int row_ptr[] = {0, 3, 3, 7};
    const int col_idx[] = {0, 2, 6, 1, 3, 4, 6};
    int packed_row_ptr[rows + 1] = {};
    int block_row_ptr[3] = {};
    const int block_count = tq::block_count_for_rows(rows, 2);

    if (tq::build_packed_row_ptr<Bits>(row_ptr, rows, packed_row_ptr) < 0) return false;
    if (tq::build_uniform_block_row_ptr(rows, 2, block_row_ptr) != block_count) return false;

    std::vector<unsigned char> packed(static_cast<std::size_t>(packed_row_ptr[rows]), 0u);
    std::vector<unsigned char> expected_packed(packed.size(), 0u);
    std::vector<float> recovered(static_cast<std::size_t>(nnz), 0.0f);
    auto view = tq::make_csr_view<Bits>(
        rows, cols, nnz, block_count, row_ptr, packed_row_ptr, col_idx, block_row_ptr,
        packed.data(), metadata);

    const unsigned int code_mask = (1u << Bits) - 1u;
    for (int row = 0; row < rows; ++row) {
        for (int index = row_ptr[row]; index < row_ptr[row + 1]; ++index) {
            const unsigned int code = static_cast<unsigned int>((index * 3 + row + 1) & code_mask);
            const int local = index - row_ptr[row];
            const int packed_at = packed_row_ptr[row] + local / tq::format_traits<Bits>::codes_per_byte;
            const int shift = (local % tq::format_traits<Bits>::codes_per_byte) * Bits;
            expected_packed[static_cast<std::size_t>(packed_at)] |= static_cast<unsigned char>(code << shift);
        }
    }

    if (tq::pack_nnz_values(&view, expected_values.data()) != 0) return false;
    if (packed != expected_packed) return false;
    if (tq::unpack_nnz_values(&view, recovered.data()) != 0) return false;
    for (int index = 0; index < nnz; ++index) {
        if (std::fabs(recovered[static_cast<std::size_t>(index)] - expected_values[static_cast<std::size_t>(index)]) > 1.0e-6f) {
            return false;
        }
    }
    return true;
}

template<int Bits>
bool test_legacy_blocked_ell_payload(bool per_gene_affine) {
    constexpr std::uint32_t rows = 3;
    constexpr std::uint32_t cols = 6;
    constexpr std::uint32_t block_size = 1;
    constexpr std::uint32_t ell_cols = 6;
    constexpr std::uint32_t ell_width = ell_cols / block_size;
    const std::uint32_t policy = per_gene_affine
        ? legacy::quantized_blocked_ell_decode_policy_per_gene_affine
        : legacy::quantized_blocked_ell_decode_policy_column_scale_row_offset;

    legacy::AffineBlockedEllCodec archive{};
    legacy::init(&archive, rows, cols, rows * cols, block_size, ell_cols, Bits, policy);
    if (!legacy::allocate(&archive)) return false;

    const auto runtime_view = legacy::make_runtime_view(archive);
    if (runtime_view.rows != rows || runtime_view.cols != cols || runtime_view.nnz != rows * cols
        || runtime_view.bits != Bits || runtime_view.decode_policy != policy
        || runtime_view.block_col_idx != archive.blockColIdx
        || runtime_view.packed_values != archive.packed_values
        || runtime_view.column_scales != archive.column_scales
        || runtime_view.column_offsets != archive.column_offsets
        || runtime_view.row_offsets != archive.row_offsets) {
        legacy::clear(&archive);
        return false;
    }

    for (std::uint32_t row = 0; row < rows; ++row) {
        for (std::uint32_t block = 0; block < ell_width; ++block) {
            archive.blockColIdx[row * ell_width + block] = block;
        }
        archive.row_offsets[row] = static_cast<float>(row % 2u) * 0.25f;
    }
    for (std::uint32_t col = 0; col < cols; ++col) {
        archive.column_scales[col] = (col % 2u == 0u) ? 0.25f : 0.5f;
        archive.column_offsets[col] = static_cast<float>(col % 3u) * 0.25f;
    }

    const int active_row_bytes = tq::format_traits<Bits>::row_bytes(static_cast<int>(ell_cols));
    const int expected_stride = tq::format_traits<Bits>::aligned_row_bytes(static_cast<int>(ell_cols));
    if (archive.row_stride_bytes != expected_stride || archive.row_stride_bytes < active_row_bytes) {
        legacy::clear(&archive);
        return false;
    }
    if (legacy::quantized_blocked_ell_row_bytes(Bits, ell_cols) != static_cast<std::uint32_t>(active_row_bytes)
        || legacy::quantized_blocked_ell_aligned_row_bytes(Bits, ell_cols) != archive.row_stride_bytes) {
        legacy::clear(&archive);
        return false;
    }

    std::vector<float> slot_values(static_cast<std::size_t>(rows) * ell_cols);
    std::vector<unsigned char> expected(legacy::packed_value_bytes(&archive), 0u);
    const unsigned int code_mask = (1u << Bits) - 1u;
    const int codes_per_byte = 8 / Bits;
    for (std::uint32_t row = 0; row < rows; ++row) {
        for (std::uint32_t slot = 0; slot < ell_cols; ++slot) {
            const unsigned int code = (row * 5u + slot * 3u + 1u) & code_mask;
            const float scale = archive.column_scales[slot];
            const float offset = per_gene_affine ? archive.column_offsets[slot] : archive.row_offsets[row];
            slot_values[static_cast<std::size_t>(row) * ell_cols + slot] = offset + scale * static_cast<float>(code);
            const std::size_t byte = static_cast<std::size_t>(row) * archive.row_stride_bytes + slot / codes_per_byte;
            expected[byte] |= static_cast<unsigned char>(code << ((slot % codes_per_byte) * Bits));
        }
    }

    if (per_gene_affine) {
        auto view = tq::make_blocked_ell_view<Bits>(
            static_cast<int>(archive.rows), static_cast<int>(archive.cols), static_cast<int>(archive.nnz),
            static_cast<int>(archive.block_size), static_cast<int>(archive.ell_cols),
            static_cast<int>(archive.row_stride_bytes), archive.blockColIdx, archive.packed_values,
            tq::make_per_gene_affine(archive.column_scales, archive.column_offsets));
        if (tq::blocked_ell::pack_row_major_values(&view, archive.packed_values, slot_values.data()) != 0) {
            legacy::clear(&archive);
            return false;
        }
        std::vector<float> recovered(slot_values.size(), 0.0f);
        if (tq::blocked_ell::unpack_row_major_values(&view, recovered.data()) != 0) {
            legacy::clear(&archive);
            return false;
        }
        for (std::size_t i = 0; i < slot_values.size(); ++i) {
            if (std::fabs(recovered[i] - slot_values[i]) > 1.0e-6f) {
                legacy::clear(&archive);
                return false;
            }
        }
    } else {
        auto view = tq::make_blocked_ell_view<Bits>(
            static_cast<int>(archive.rows), static_cast<int>(archive.cols), static_cast<int>(archive.nnz),
            static_cast<int>(archive.block_size), static_cast<int>(archive.ell_cols),
            static_cast<int>(archive.row_stride_bytes), archive.blockColIdx, archive.packed_values,
            tq::make_column_scale_row_offset(archive.column_scales, archive.row_offsets));
        if (tq::blocked_ell::pack_row_major_values(&view, archive.packed_values, slot_values.data()) != 0) {
            legacy::clear(&archive);
            return false;
        }
        std::vector<float> recovered(slot_values.size(), 0.0f);
        if (tq::blocked_ell::unpack_row_major_values(&view, recovered.data()) != 0) {
            legacy::clear(&archive);
            return false;
        }
        for (std::size_t i = 0; i < slot_values.size(); ++i) {
            if (std::fabs(recovered[i] - slot_values[i]) > 1.0e-6f) {
                legacy::clear(&archive);
                return false;
            }
        }
    }

    const bool same_bytes = std::equal(
        expected.begin(), expected.end(), archive.packed_values);
    legacy::clear(&archive);
    return same_bytes;
}

template<int Bits>
bool test_bit_width() {
    std::vector<float> csr_values(7u);
    for (int i = 0; i < 7; ++i) {
        const int row = i < 3 ? 0 : 2;
        csr_values[static_cast<std::size_t>(i)] = static_cast<float>((i * 3 + row + 1) & ((1 << Bits) - 1));
    }
    const auto per_gene = tq::make_per_gene_affine(static_cast<const float*>(nullptr));
    return test_csr_round_trip<Bits>(per_gene, csr_values)
        && test_legacy_blocked_ell_payload<Bits>(true)
        && test_legacy_blocked_ell_payload<Bits>(false);
}

} // namespace

int main() {
    static_assert(std::is_trivially_copyable_v<tq::CsrView<4, float, tq::per_gene_affine<float>>>);
    static_assert(std::is_trivially_copyable_v<tq::BlockedEllView<4, float, tq::per_gene_affine<float>>>);
    if (!test_bit_width<1>() || !test_bit_width<2>() || !test_bit_width<4>() || !test_bit_width<8>()) {
        std::fprintf(stderr, "quantized format contract failed\n");
        return 1;
    }
    return 0;
}
