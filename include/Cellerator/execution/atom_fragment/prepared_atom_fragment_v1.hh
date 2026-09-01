#pragma once

#include <Cellerator/execution/atom_fragment/atom_bound_candidate_v1.hh>
#include <Cellerator/execution/program/program_v2.h>

namespace cellerator::execution::atom_fragment {

struct prepared_atom_fragment_v1 {
    atom_bound_candidate_v1 candidate{};
    const program::prepared_program_v2 *program = nullptr;
    order_id input_order{};
    order_id output_order{};
    std::uint64_t binding_count = 0u;
    std::uint64_t maximum_binding_workspace_bytes = 0u;
};

enum class prepared_atom_fragment_status_code_v1 : std::uint8_t {
    success = 0u,
    null_output,
    invalid_candidate,
    invalid_program,
    empty_program,
    foreign_candidate_stage,
    invalid_order,
};

struct prepared_atom_fragment_status_v1 {
    prepared_atom_fragment_status_code_v1 code =
        prepared_atom_fragment_status_code_v1::success;
    std::uint64_t index = 0u;

    constexpr explicit operator bool() const noexcept {
        return code == prepared_atom_fragment_status_code_v1::success;
    }
};

prepared_atom_fragment_status_v1 prepare_atom_fragment_v1(
    const atom_bound_candidate_v1 &candidate,
    const program::prepared_program_v2 &program,
    order_id input_order,
    order_id output_order,
    prepared_atom_fragment_v1 *output) noexcept;

} // namespace cellerator::execution::atom_fragment
