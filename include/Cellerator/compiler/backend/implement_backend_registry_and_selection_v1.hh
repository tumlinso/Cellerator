#pragma once

#include <Cellerator/compiler/backend/freeze_the_backend_provider_abi_v1.hh>

#include <cstdint>
#include <string_view>
#include <vector>

namespace cellerator::compiler::backend::v1 {

struct backend_registry_entry_v1 {
    backend_string_view_v1 name{};
    backend_string_view_v1 source_fragment{};
    const backend_provider_v1* provider = nullptr;
    std::uint32_t priority = 0;
};

enum class backend_selection_policy_v1 : std::uint8_t {
    highest_priority = 0,
    force_named,
};

struct backend_selection_request_v1 {
    backend_target_v1 target{};
    std::uint64_t required_capabilities = backend_capability_ordinary_object_v1;
    backend_selection_policy_v1 policy = backend_selection_policy_v1::highest_priority;
    backend_string_view_v1 forced_name{};
    bool allow_conventional_fallback = true;
};

enum class backend_selection_status_v1 : std::uint8_t {
    selected = 0,
    invalid_entry,
    unavailable,
    ambiguous,
    forced_backend_unavailable,
};

struct backend_selection_result_v1 {
    backend_selection_status_v1 status = backend_selection_status_v1::unavailable;
    const backend_registry_entry_v1* entry = nullptr;
    bool used_conventional_fallback = false;
};

class backend_registry_v1 {
public:
    [[nodiscard]] backend_selection_status_v1 register_backend(
        backend_registry_entry_v1 entry) noexcept;

    [[nodiscard]] backend_selection_result_v1 select(
        const backend_selection_request_v1& request,
        backend_diagnostic_sink_v1 diagnostics = {}) const noexcept;

    [[nodiscard]] std::size_t size() const noexcept { return entries_.size(); }

private:
    std::vector<backend_registry_entry_v1> entries_;
};

}  // namespace cellerator::compiler::backend::v1
