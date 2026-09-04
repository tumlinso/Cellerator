#include <Cellerator/compiler/backend/nvcc/freeze_the_nvcc_backend_v1.hh>

#include <cassert>
#include <type_traits>

int main() {
    using namespace cellerator::compiler::backend::nvcc::v1;

    static_assert(std::is_trivially_copyable_v<nvcc_backend_receipt>);
    nvcc_backend_receipt receipt;
    receipt.receipt_mask = all_nvcc_backend_receipts;
    receipt.source_revision_low = 0x41430c16U;
    receipt.object_hash_low = 0x1U;
    receipt.compute_capability = 70;
    receipt.exact_output = true;
    receipt.conventional_fallback_retained = true;

    freeze_nvcc_backend_status status{};
    const auto frozen = freeze_nvcc_backend(receipt, &status);
    assert(frozen);
    assert(status == freeze_nvcc_backend_status::frozen);
    assert(frozen->backend_abi_version == 1);
    assert(frozen->realization_ir_version == 1);

    receipt.receipt_mask &= ~static_cast<std::uint32_t>(
        nvcc_backend_receipt_kind::diagnostics);
    assert(!freeze_nvcc_backend(receipt, &status));
    assert(status == freeze_nvcc_backend_status::incomplete_receipts);
}
