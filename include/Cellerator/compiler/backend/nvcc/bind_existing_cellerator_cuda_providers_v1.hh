#pragma once
#include <cstdint>
#include <optional>
#include <string>
#include <vector>
namespace cellerator::compiler::backend::nvcc::v1 {
struct provider_binding_request { std::uint64_t candidate=0,provider=0,prepared_contract=0; std::uint32_t architecture=0; };
struct source_linked_provider_binding { std::uint64_t candidate=0,provider=0,prepared_contract=0; std::string target; std::string entrypoint; bool generated_kernel=false; };
enum class provider_binding_status:std::uint8_t{ok=0,invalid_identity,unsupported_architecture,unknown_provider,contract_mismatch};
[[nodiscard]] std::optional<source_linked_provider_binding> bind_existing_provider(const provider_binding_request&, provider_binding_status* = nullptr) noexcept;
} // namespace cellerator::compiler::backend::nvcc::v1
