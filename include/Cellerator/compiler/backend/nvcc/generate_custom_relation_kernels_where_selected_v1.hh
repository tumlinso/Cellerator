#pragma once
#include <Cellerator/compiler/backend/nvcc/implement_cuda_source_emission_v1.hh>
#include <cstdint>
#include <optional>
#include <string>
namespace cellerator::compiler::backend::nvcc::v1 {
struct custom_relation_kernel_request {
    std::string name;
    std::uint64_t exact_members=0, exact_edges=0, structure_epoch=0, value_generation=0;
    std::uint32_t width=0;
    bool persistent_order=false, additive_partial=false, affine_epilogue=false, prelinked_provider_selected=false;
};
enum class custom_kernel_status:std::uint8_t{ok=0,provider_already_selected,invalid_coverage,invalid_generation,invalid_width};
[[nodiscard]] std::optional<realized_cuda_entity> generate_custom_relation_kernel(const custom_relation_kernel_request&,custom_kernel_status* = nullptr) noexcept;
}
