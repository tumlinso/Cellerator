#include <CellPack/matrix_view.hh>

#include <stdexcept>

namespace {

void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

} // namespace

int main() {
    const cellpack::u32 csr_offsets[] = { 0u, 2u, 2u, 4u };
    const cellpack::u32 csr_features[] = { 0u, 3u, 1u, 4u };
    const float csr_values[] = { 1.0f, 2.0f, 3.0f, 4.0f };
    cellpack::csr_view csr;
    csr.row_count = 3u;
    csr.feature_count = 5u;
    csr.nnz_count = 4u;
    csr.row_offsets = csr_offsets;
    csr.feature_ids = csr_features;
    csr.values = csr_values;
    require(static_cast<bool>(cellpack::validate_csr_view(csr)), "valid CSR rejected");

    const cellpack::u32 bad_csr_offsets[] = { 0u, 3u, 2u, 4u };
    csr.row_offsets = bad_csr_offsets;
    require(cellpack::validate_csr_view(csr).code == cellpack::validation_code::invalid_matrix_view, "non-monotonic CSR offsets accepted");

    const cellpack::u32 unsorted_csr_features[] = { 3u, 0u, 1u, 4u };
    csr.row_offsets = csr_offsets;
    csr.feature_ids = unsorted_csr_features;
    require(cellpack::validate_csr_view(csr).code == cellpack::validation_code::invalid_matrix_view, "unsorted CSR features accepted");

    const cellpack::u32 out_of_range_csr_features[] = { 0u, 5u, 1u, 4u };
    csr.feature_ids = out_of_range_csr_features;
    require(cellpack::validate_csr_view(csr).code == cellpack::validation_code::invalid_matrix_view, "out-of-range CSR feature accepted");

    const cellpack::u32 bad_final_offsets[] = { 0u, 2u, 2u, 3u };
    csr.row_offsets = bad_final_offsets;
    csr.feature_ids = csr_features;
    require(cellpack::validate_csr_view(csr).code == cellpack::validation_code::invalid_matrix_view, "CSR nnz mismatch accepted");

    const cellpack::u32 coo_rows[] = { 0u, 0u, 2u, 2u };
    const cellpack::u32 coo_features[] = { 0u, 3u, 1u, 4u };
    const float coo_values[] = { 1.0f, 2.0f, 3.0f, 4.0f };
    cellpack::coo_view coo;
    coo.row_count = 3u;
    coo.feature_count = 5u;
    coo.nnz_count = 4u;
    coo.row_ids = coo_rows;
    coo.feature_ids = coo_features;
    coo.values = coo_values;
    require(static_cast<bool>(cellpack::validate_coo_view(coo)), "valid COO rejected");

    const cellpack::u32 bad_coo_rows[] = { 0u, 2u, 1u, 2u };
    coo.row_ids = bad_coo_rows;
    require(cellpack::validate_coo_view(coo).code == cellpack::validation_code::invalid_matrix_view, "unsorted COO rows accepted");

    const cellpack::u32 out_of_range_coo_rows[] = { 0u, 0u, 3u, 3u };
    coo.row_ids = out_of_range_coo_rows;
    require(cellpack::validate_coo_view(coo).code == cellpack::validation_code::invalid_matrix_view, "out-of-range COO rows accepted");

    const cellpack::u32 unsorted_coo_features[] = { 3u, 0u, 1u, 4u };
    coo.row_ids = coo_rows;
    coo.feature_ids = unsorted_coo_features;
    require(cellpack::validate_coo_view(coo).code == cellpack::validation_code::invalid_matrix_view, "unsorted COO features accepted");

    return 0;
}
