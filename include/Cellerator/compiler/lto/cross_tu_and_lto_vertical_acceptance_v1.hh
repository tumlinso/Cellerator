#pragma once

#include <Cellerator/compiler/lto/lto_v1.hh>

#include <cstdint>

namespace cellerator::compiler::lto::v1 {

struct cross_tu_lto_acceptance_v1 {
    std::uint32_t separately_compiled_cell_translation_units = 0;
    bool ceir_embedded_and_extracted = false;
    bool authorized_chain_jointly_planned = false;
    bool ordinary_objects_emitted_and_linked = false;
    bool plain_cpp_coexists = false;
    bool profile_environment_merged = false;
    bool deterministic_output = false;
    bool incremental_rebuild_reused_unaffected_artifacts = false;
};

[[nodiscard]] bool validate_cross_tu_lto_acceptance_v1(
    const cross_tu_lto_acceptance_v1&) noexcept;

}  // namespace cellerator::compiler::lto::v1
