#include "Cellerator/execution/program/program_v1_adapter.h"

namespace cellerator::execution::program {

program_status adapt_legacy_program_v1(
        const legacy_program_v1& legacy,
        prepared_stage_v2* stage_storage,
        std::uint64_t stage_capacity,
        std::uint64_t* dependency_storage,
        std::uint64_t dependency_capacity,
        prepared_program_v2* program) noexcept {
    if (program == nullptr ||
        (legacy.entry_count != 0 &&
         (legacy.entries == nullptr || stage_storage == nullptr)) ||
        legacy.entry_count > 5 || stage_capacity < legacy.entry_count ||
        (legacy.entry_count > 1 &&
         (dependency_storage == nullptr ||
          dependency_capacity < legacy.entry_count - 1U))) {
        return program_status::invalid_argument;
    }
    for (std::uint32_t i = 0; i < legacy.entry_count; ++i) {
        const auto& entry = legacy.entries[i];
        stage_storage[i] = {entry.stable_stage_id, entry.candidate_id,
                            entry.prepared_state, entry.launch,
                            i == 0 ? 0U : i - 1U, i == 0 ? 0U : 1U,
                            entry.binding_index,
                            entry.required_workspace_bytes};
        if (i != 0) dependency_storage[i - 1U] = i - 1U;
    }
    *program = {2, 0, stage_storage, legacy.entry_count,
                dependency_storage, legacy.entry_count == 0
                        ? 0U : legacy.entry_count - 1U};
    return validate_prepared_program_v2(*program);
}

}  // namespace cellerator::execution::program
