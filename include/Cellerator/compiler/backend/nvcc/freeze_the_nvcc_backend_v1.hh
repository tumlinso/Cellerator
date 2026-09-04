#pragma once

#include <Cellerator/compiler/backend/nvcc/nvcc_backend_v1.hh>

#include <optional>

namespace cellerator::compiler::backend::nvcc::v1 {

enum class freeze_nvcc_backend_status : std::uint8_t {
    frozen = 0,
    incompatible_interface,
    incomplete_receipts,
    missing_source_identity,
    unsupported_architecture,
    unvalidated_output,
    missing_fallback,
};

[[nodiscard]] std::optional<nvcc_backend_receipt> freeze_nvcc_backend(
    const nvcc_backend_receipt& receipt,
    freeze_nvcc_backend_status* status = nullptr) noexcept;

} // namespace cellerator::compiler::backend::nvcc::v1
