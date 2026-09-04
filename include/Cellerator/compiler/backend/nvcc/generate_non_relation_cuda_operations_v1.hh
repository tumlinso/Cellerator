#pragma once
#include <Cellerator/compiler/backend/nvcc/implement_cuda_source_emission_v1.hh>
#include <cstdint>
#include <optional>
#include <string>
namespace cellerator::compiler::backend::nvcc::v1 {
enum class non_relation_operation:std::uint8_t{transpose=1,contraction,segment,normalization,gate,sparse_update,bundle,chain,moments,exchange,publish};
struct non_relation_request{non_relation_operation operation=non_relation_operation::transpose;std::string name;std::uint64_t input_generation=0,output_generation=0;bool deterministic=true;};
enum class non_relation_status:std::uint8_t{ok=0,invalid_operation,invalid_name,invalid_generation};
[[nodiscard]] std::optional<realized_cuda_entity> generate_non_relation_operation(const non_relation_request&,non_relation_status* = nullptr) noexcept;
}
