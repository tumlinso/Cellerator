#pragma once
#include <Cellerator/compiler/diagnostics/freeze_validation_mode_semantics_v1.hh>
#include <cstdint>
#include <string_view>
namespace cellerator::compiler::diagnostics::v1 {
enum class target_native_issue:std::uint8_t{unsupported_instruction=0,capability_range,clobber,alignment,address_space,collective,abi,graph_capture,fallback_unavailable};
struct target_native_request{target_native_issue issue=target_native_issue::unsupported_instruction;validation_mode mode=validation_mode::checked;bool representable=false,fallback_available=false,unsafe_acknowledged=false;};
struct target_native_diagnostic{bool continue_compilation=false;bool unsafe_continuation=false;std::string_view explanation;};
[[nodiscard]] target_native_diagnostic diagnose_target_native(const target_native_request&) noexcept;
}
