#pragma once
#include <cstdint>
#include <optional>
#include <string>
#include <vector>
namespace Cellerator::compiler::composition {
struct basis_member_v1{std::string atom,production,membership;double redundancy=0;};
struct basis_manifest_v1{std::string id,evidence_fingerprint;std::uint64_t budget_bytes=0,evidence_generation=0;bool valid=false;std::vector<double> objective_vector;std::vector<basis_member_v1> members;};
[[nodiscard]] std::string print_basis_manifest_v1(const basis_manifest_v1&);
[[nodiscard]] std::optional<basis_manifest_v1> parse_basis_manifest_v1(const std::string&);
} // namespace Cellerator::compiler::composition
