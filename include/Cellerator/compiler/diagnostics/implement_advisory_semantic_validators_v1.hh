#pragma once
#include <Cellerator/compiler/diagnostics/freeze_validation_mode_semantics_v1.hh>
#include <cstdint>
namespace cellerator::compiler::diagnostics::v1 {
enum class semantic_inconsistency : std::uint32_t { domain=1U, order=2U, generation=4U, effect=8U, numerical=16U, identity=32U };
enum class advisory_disposition : std::uint8_t { clean=0, warning, error, suppressed, trusted_continuation, unsafe_continuation };
struct advisory_request { semantic_inconsistency issue=semantic_inconsistency::domain; validation_mode mode=validation_mode::checked; bool suppress_warning=false; bool escalate_warning=false; bool force_continuation=false; bool representable=true; };
[[nodiscard]] advisory_disposition validate_semantic_advisory(const advisory_request&) noexcept;
}
