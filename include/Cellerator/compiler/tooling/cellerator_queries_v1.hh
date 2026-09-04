#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace cellerator::compiler::tooling::v1 {

enum class semantic_query_kind : unsigned char {
    completion,
    hover,
    profile_state,
    semantic_ir,
    planning_ir,
    candidate_cost,
    mutation_staleness,
    decomposition,
    native_navigation,
};

struct semantic_query_request {
    semantic_query_kind kind = semantic_query_kind::completion;
    std::string source;
    std::size_t cursor = 0;
    bool prefer_cached = true;
};

struct semantic_query_result {
    semantic_query_kind kind = semantic_query_kind::completion;
    std::string payload;
    std::string source_location;
    bool stale = false;
};

struct celleratord_semantic_acceptance_v1 {
    std::vector<semantic_query_kind> supported_queries;
    std::vector<std::string> installed_profiles;
    bool lsp_integration = false;
    bool snapshots_stable = false;
};

[[nodiscard]] celleratord_semantic_acceptance_v1 query_celleratord_semantics_v1();

}  // namespace cellerator::compiler::tooling::v1
