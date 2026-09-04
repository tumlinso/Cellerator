#pragma once

#include <Cellerator/compiler/frontend/cxx/integrate_overload_resolution_and_cellerator_semantic_ca_v1.hh>

#include <cstdint>
#include <string>
#include <vector>

namespace Cellerator::compiler::frontend::cxx {

inline constexpr std::uint32_t opaque_native_call_schema_version_v1 = 1;

enum native_effect_v1 : std::uint32_t {
    native_effect_read_v1 = 1u << 0,
    native_effect_write_v1 = 1u << 1,
    native_effect_escape_v1 = 1u << 2,
    native_effect_synchronize_v1 = 1u << 3,
};

enum class opaque_native_call_status_v1 : std::uint8_t {
    success = 0,
    schema_mismatch,
    invalid_candidate,
};

struct native_call_contract_v1 {
    std::string qualified_name;
    std::uint32_t effects = 0;
};

struct opaque_native_call_v1 {
    const void* selected_declaration = nullptr;
    std::string qualified_name;
    std::uint32_t effects = 0;
    bool semantic_barrier = false;
    bool contract_applied = false;
    std::string diagnostic;
};

opaque_native_call_status_v1 model_opaque_native_calls_v1(
    std::uint32_t schema_version,
    const std::vector<overload_semantic_candidate_v1>& calls,
    const std::vector<native_call_contract_v1>& contracts,
    std::vector<opaque_native_call_v1>* models) noexcept;

bool may_reorder_across_native_call_v1(
    const opaque_native_call_v1& call,
    std::uint32_t moving_effects) noexcept;

}  // namespace Cellerator::compiler::frontend::cxx
