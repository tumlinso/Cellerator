#include <Cellerator/core/interop/cellshard_access.cuh>

#include <cassert>

int main() {
    namespace ccm = cellerator::core::matrix;
    namespace cci = cellerator::core::interop;
    namespace csa = cellshard::access;

    ccm::blocked_ell blocked{};
    ccm::init(&blocked, 4u, 8u, 12u, 2u, 4u);
    auto blocked_binding = cci::make_cellshard_matrix_binding(blocked, 3u);
    auto blocked_view = csa::make_adapter_view(blocked_binding);
    static_assert(csa::is_archive_adapter<decltype(blocked_binding)>::value,
                  "CelleratorCore Blocked-ELL binding must satisfy the CellShard archive adapter contract");
    static_assert(csa::is_pack_adapter<decltype(blocked_binding)>::value,
                  "CelleratorCore Blocked-ELL binding must satisfy the CellShard pack adapter contract");

    const cellshard::access::archive_descriptor blocked_archive = csa::describe_archive(blocked_view);
    assert(blocked_archive.cell_count == 4u);
    assert(blocked_archive.feature_count == 8u);
    assert(blocked_archive.nnz == 12u);
    assert(blocked_archive.archive_format == cellshard::disk_format_blocked_ell);
    assert(blocked_archive.execution_format == cellshard::dataset_execution_format_blocked_ell);

    ccm::sliced_ell sliced{};
    ccm::init(&sliced, 5u, 9u, 11u);
    auto sliced_binding = cci::make_cellshard_matrix_binding(sliced, 4u);
    auto sliced_view = csa::make_adapter_view(sliced_binding);
    const cellshard::access::archive_descriptor sliced_archive = csa::describe_archive(sliced_view);
    assert(sliced_archive.archive_format == cellshard::disk_format_sliced_ell);
    assert(sliced_archive.execution_format == cellshard::dataset_execution_format_sliced_ell);

    ccm::quantized_blocked_ell quantized{};
    ccm::init(&quantized, 6u, 10u, 14u, 2u, 4u, 4u, ccm::quantized_blocked_ell_decode_policy_per_gene_affine);
    auto quantized_binding = cci::make_cellshard_matrix_binding(quantized, 5u);
    auto quantized_view = csa::make_adapter_view(quantized_binding);
    const cellshard::access::pack_descriptor quantized_pack = csa::describe_pack(quantized_view);
    assert(quantized_pack.execution_format == cellshard::dataset_execution_format_quantized_blocked_ell);
    assert(quantized_pack.nnz == 14u);

    return 0;
}
