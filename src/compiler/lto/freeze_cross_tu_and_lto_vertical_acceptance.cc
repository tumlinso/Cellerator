#include <Cellerator/compiler/lto/freeze_cross_tu_and_lto_vertical_acceptance_v1.hh>

namespace cellerator::compiler::lto::v1 {

bool validate_cross_tu_lto_acceptance_v1(
    const cross_tu_lto_acceptance_v1& receipt) noexcept {
    return receipt.separately_compiled_cell_translation_units >= 2 &&
           receipt.ceir_embedded_and_extracted &&
           receipt.authorized_chain_jointly_planned &&
           receipt.ordinary_objects_emitted_and_linked &&
           receipt.plain_cpp_coexists && receipt.profile_environment_merged &&
           receipt.deterministic_output &&
           receipt.incremental_rebuild_reused_unaffected_artifacts;
}

}  // namespace cellerator::compiler::lto::v1
