#pragma once

#include <cstdint>
#include <vector>

namespace Cellerator::compiler::planning {

enum class candidate_edit_authority_v1 : std::uint8_t {
    automatic = 0u,
    source = 1u,
    pipeline = 2u,
    user = 3u,
};

enum class candidate_edit_mode_v1 : std::uint8_t {
    offer = 1u,
    exclude,
    force,
    unsafe_force,
};

struct candidate_choice_v1 {
    std::uint64_t candidate_identity = 0u;
    std::uint64_t predicted_nanoseconds = 0u;
    bool admissible = true;
    bool dominated = false;
};

struct candidate_edit_v1 {
    std::uint64_t candidate_identity = 0u;
    candidate_edit_authority_v1 authority = candidate_edit_authority_v1::source;
    candidate_edit_mode_v1 mode = candidate_edit_mode_v1::offer;
};

enum class candidate_edit_diagnostic_v1 : std::uint8_t {
    none = 0u,
    unknown_candidate,
    impossible_choice,
    dominated_choice,
    lower_authority_ignored,
    no_candidate_available,
};

struct candidate_edit_receipt_v1 {
    std::uint64_t candidate_identity = 0u;
    candidate_edit_authority_v1 authority = candidate_edit_authority_v1::automatic;
    candidate_edit_mode_v1 mode = candidate_edit_mode_v1::offer;
    candidate_edit_diagnostic_v1 diagnostic = candidate_edit_diagnostic_v1::none;
    bool applied = false;
};

struct candidate_selection_v1 {
    std::uint64_t selected_candidate_identity = 0u;
    candidate_edit_mode_v1 selection_mode = candidate_edit_mode_v1::offer;
    bool unsafe = false;
    std::vector<std::uint64_t> offered_candidates;
    std::vector<candidate_edit_receipt_v1> receipts;
};

[[nodiscard]] candidate_selection_v1 apply_candidate_edits_v1(
    const std::vector<candidate_choice_v1>& candidates,
    const std::vector<candidate_edit_v1>& edits);

}  // namespace Cellerator::compiler::planning
