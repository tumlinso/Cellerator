#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace Cellerator::compiler::backend::nvptx {

enum class mixed_backend_route_v1 : std::uint8_t {
    prelinked_provider = 1u,
    generated_nvcc,
    direct_ptx,
};

struct mixed_backend_candidate_v1 {
    std::uint64_t candidate_identity = 0u;
    mixed_backend_route_v1 route = mixed_backend_route_v1::prelinked_provider;
    bool available = false;
    bool exact = false;
    std::uint64_t abi_identity = 0u;
    std::uint64_t structure_identity = 0u;
    std::uint64_t input_order_identity = 0u;
    std::uint64_t output_order_identity = 0u;
    std::uint64_t input_generation = 0u;
    std::uint64_t output_generation = 0u;
};

struct mixed_backend_stage_v1 {
    std::uint64_t stage_identity = 0u;
    std::vector<mixed_backend_candidate_v1> prioritized_candidates;
};

struct mixed_backend_graph_v1 {
    std::uint64_t abi_identity = 0u;
    std::uint64_t structure_identity = 0u;
    std::uint64_t initial_order_identity = 0u;
    std::uint64_t initial_generation = 0u;
    std::vector<mixed_backend_stage_v1> stages;
};

enum class mixed_backend_graph_status_v1 : std::uint8_t {
    success = 0u,
    invalid_graph,
    no_available_exact_route,
};

struct mixed_backend_selection_v1 {
    std::uint64_t stage_identity = 0u;
    std::uint64_t candidate_identity = 0u;
    mixed_backend_route_v1 route = mixed_backend_route_v1::prelinked_provider;
    bool fallback_used = false;
};

struct mixed_backend_graph_result_v1 {
    mixed_backend_graph_status_v1 status = mixed_backend_graph_status_v1::invalid_graph;
    std::vector<mixed_backend_selection_v1> selections;
    std::vector<std::string> diagnostics;
    std::uint64_t final_order_identity = 0u;
    std::uint64_t final_generation = 0u;

    explicit operator bool() const noexcept {
        return status == mixed_backend_graph_status_v1::success;
    }
};

[[nodiscard]] mixed_backend_graph_result_v1 validate_and_select_mixed_backend_graph_v1(
    const mixed_backend_graph_v1& graph);

}  // namespace Cellerator::compiler::backend::nvptx
