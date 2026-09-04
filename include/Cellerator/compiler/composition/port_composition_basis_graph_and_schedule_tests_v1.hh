#pragma once
#include <cstddef>
#include <string>
#include <vector>
namespace Cellerator::compiler::composition {
enum class ported_test_kind_v1{derivation,no_basis,exact_coverage,performance_baseline};
struct ported_test_inventory_v1{std::string source_fixture,ported_fixture,provenance_hash;ported_test_kind_v1 kind=ported_test_kind_v1::derivation;std::size_t source_cases=0,ported_cases=0;};
[[nodiscard]] bool reconcile_ported_test_inventory_v1(const std::vector<ported_test_inventory_v1>&,std::string*error=nullptr);
} // namespace Cellerator::compiler::composition
