#pragma once
#include <cstdint>
#include <string_view>
namespace cellerator::compiler::diagnostics::v1 {
enum class coverage_failure : std::uint8_t { none=0, omission, duplicate, wrong_role, incompatible_partial_algebra, halo_as_contributor, canonical_recovery_failure };
struct coverage_evidence { coverage_failure failure=coverage_failure::none; std::uint64_t member=0, expected_owner=0, observed_owner=0; };
struct coverage_diagnostic { bool valid=false; std::uint64_t member=0; std::string_view explanation; };
[[nodiscard]] coverage_diagnostic diagnose_exact_coverage(const coverage_evidence&) noexcept;
}
