#pragma once
#include <cstddef>
#include <string>
#include <vector>
namespace Cellerator::compiler::composition {
struct induced_production_candidate_v1{std::string name,evidence,verifier;double confidence=0,total_cost=0;bool exact=false;};
struct induced_grammar_search_v1{std::vector<induced_production_candidate_v1> evaluated,promoted;bool no_promotion=false;};
[[nodiscard]] induced_grammar_search_v1 search_induced_grammar_v1(std::vector<induced_production_candidate_v1>,std::size_t bound,double baseline_cost,double minimum_confidence);
} // namespace Cellerator::compiler::composition
