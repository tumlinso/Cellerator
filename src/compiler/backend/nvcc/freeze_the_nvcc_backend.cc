#include <Cellerator/compiler/backend/nvcc/freeze_the_nvcc_backend_v1.hh>

namespace cellerator::compiler::backend::nvcc::v1 {

std::optional<nvcc_backend_receipt> freeze_nvcc_backend(
    const nvcc_backend_receipt& receipt,
    freeze_nvcc_backend_status* status) noexcept {
    const auto set_status = [status](freeze_nvcc_backend_status value) {
        if (status != nullptr) {
            *status = value;
        }
    };
    if (receipt.contract_version != nvcc_backend_contract_version ||
        receipt.backend_abi_version != required_backend_abi_version ||
        receipt.realization_ir_version != required_realization_ir_version) {
        set_status(freeze_nvcc_backend_status::incompatible_interface);
        return std::nullopt;
    }
    if (receipt.receipt_mask != all_nvcc_backend_receipts) {
        set_status(freeze_nvcc_backend_status::incomplete_receipts);
        return std::nullopt;
    }
    if ((receipt.source_revision_high == 0 && receipt.source_revision_low == 0) ||
        (receipt.object_hash_high == 0 && receipt.object_hash_low == 0)) {
        set_status(freeze_nvcc_backend_status::missing_source_identity);
        return std::nullopt;
    }
    if (receipt.compute_capability != 70) {
        set_status(freeze_nvcc_backend_status::unsupported_architecture);
        return std::nullopt;
    }
    if (!receipt.exact_output) {
        set_status(freeze_nvcc_backend_status::unvalidated_output);
        return std::nullopt;
    }
    if (!receipt.conventional_fallback_retained) {
        set_status(freeze_nvcc_backend_status::missing_fallback);
        return std::nullopt;
    }
    set_status(freeze_nvcc_backend_status::frozen);
    return receipt;
}

} // namespace cellerator::compiler::backend::nvcc::v1
