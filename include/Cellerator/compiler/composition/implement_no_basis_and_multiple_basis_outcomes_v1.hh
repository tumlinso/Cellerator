#pragma once
#include <optional>
#include <string>
#include <string_view>
#include <vector>
namespace Cellerator::compiler::composition {
struct basis_outcome_v1{std::string id,profile;double total_cost=0;bool valid=false,external=false;};
struct basis_selection_v1{bool use_basis=false;std::optional<basis_outcome_v1> selected;std::string reason;};
[[nodiscard]] basis_selection_v1 select_basis_outcome_v1(const std::vector<basis_outcome_v1>&,std::string_view profile,double no_basis_cost);
} // namespace Cellerator::compiler::composition
