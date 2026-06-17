#include <CellPack/permutation.hh>
#include <CellPack/validate.hh>

#include <stdexcept>

namespace {

void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

} // namespace

int main() {
    const cellpack::u32 row_permutation[] = { 2u, 0u, 3u, 1u };
    cellpack::u32 inverse[4] = {};
    require(cellpack::build_inverse_permutation(row_permutation, 4u, inverse), "inverse permutation build failed");
    require(inverse[0] == 1u && inverse[1] == 3u && inverse[2] == 0u && inverse[3] == 2u, "inverse permutation values mismatch");
    require(cellpack::validate_inverse_permutation(row_permutation, inverse, 4u), "inverse permutation failed validation");

    const cellpack::u32 invalid_permutation[] = { 0u, 2u, 2u, 3u };
    require(!cellpack::validate_permutation(invalid_permutation, 4u), "duplicate permutation was accepted");

    cellpack::packed_region_desc primary{};
    primary.region_id = 0u;
    primary.parent_id = cellpack::invalid_id;
    primary.layout = cellpack::to_u32(cellpack::layout_kind::blocked_ell);
    primary.role = cellpack::to_u32(cellpack::region_role::primary);
    primary.module_id = 7u;
    primary.signature_id = 0u;
    primary.row_begin = 0u;
    primary.row_count = 2u;
    primary.feature_begin = 0u;
    primary.feature_count = 3u;
    primary.index_offset = cellpack::invalid_id;
    primary.value_offset = cellpack::invalid_id;
    primary.aux_offset = cellpack::invalid_id;
    primary.weight_offset = cellpack::invalid_id;
    primary.output_offset = cellpack::invalid_id;
    require(static_cast<bool>(cellpack::validate_region_desc(primary, 4u, 8u)), "valid primary region rejected");

    cellpack::packed_region_desc residual = primary;
    residual.region_id = 1u;
    residual.flags = cellpack::region_flag_residual;
    residual.layout = cellpack::to_u32(cellpack::layout_kind::residual_csr);
    residual.role = cellpack::to_u32(cellpack::region_role::residual);
    residual.row_begin = 0u;
    residual.row_count = 4u;
    residual.feature_begin = 6u;
    residual.feature_count = 2u;
    require(static_cast<bool>(cellpack::validate_region_desc(residual, 4u, 8u)), "valid residual region rejected");

    cellpack::packed_region_desc invalid = residual;
    invalid.layout = cellpack::to_u32(cellpack::layout_kind::blocked_ell);
    require(!static_cast<bool>(cellpack::validate_region_desc(invalid, 4u, 8u)), "invalid residual layout accepted");

    cellpack::packed_region_desc overlapping[2] = { primary, primary };
    overlapping[1].region_id = 2u;
    overlapping[1].feature_begin = 2u;
    require(!static_cast<bool>(cellpack::validate_region_sequence(overlapping, 2u, 4u, 8u)), "overlapping region sequence accepted");

    return 0;
}
