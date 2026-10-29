#pragma once
#include <string>
#include <string_view>
namespace Cellerator::compiler::composition {
struct superatom_candidate_v1{std::string name,profile,exact_derivation,deconstruction,evidence;double total_cost=0,baseline_cost=0;bool exact=false,experimental=true;};
struct superatom_promotion_v1{bool promoted=false;std::string reason;};
[[nodiscard]] superatom_promotion_v1 evaluate_superatom_promotion_v1(const superatom_candidate_v1&,std::string_view active_profile);
} // namespace Cellerator::compiler::composition
