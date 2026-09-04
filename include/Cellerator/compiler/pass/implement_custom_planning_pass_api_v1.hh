#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace cellerator::compiler::pass::v1 {

struct planning_atom_v1 { std::uint64_t id = 0; };
struct planning_evidence_v1 { std::uint64_t id = 0; double confidence = 0; };
struct planning_decomposition_v1 {
    std::uint64_t id = 0;
    std::vector<std::uint64_t> covered_atoms;
};
struct planning_candidate_v1 {
    std::uint64_t id = 0;
    std::uint64_t decomposition = 0;
    double total_cost = 0;
    std::string provider;
};

enum class planning_pass_mode_v1 : std::uint8_t { augment = 1, replace = 2 };

struct planning_pass_context_v1 {
    std::vector<planning_atom_v1>* atoms = nullptr;
    std::vector<planning_evidence_v1>* evidence = nullptr;
    std::vector<planning_decomposition_v1>* decompositions = nullptr;
    std::vector<planning_candidate_v1>* candidates = nullptr;
    std::uint64_t* selected_candidate = nullptr;
    std::vector<std::string>* diagnostics = nullptr;
    planning_pass_mode_v1 mode = planning_pass_mode_v1::augment;
};

using planning_pass_run_v1 = bool (*)(planning_pass_context_v1&) noexcept;

enum class planning_pass_status_v1 : std::uint8_t {
    success = 0, invalid_context, pass_failed, invalid_result,
};

[[nodiscard]] planning_pass_status_v1 run_custom_planning_pass_v1(
    planning_pass_context_v1& context, planning_pass_run_v1 pass) noexcept;

}  // namespace cellerator::compiler::pass::v1
