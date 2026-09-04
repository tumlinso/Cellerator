#include <Cellerator/compiler/pass/freeze_the_open_compiler_extension_surface_v1.hh>

namespace cellerator::compiler::pass::v1 {

open_compiler_extension_surface_v1
open_compiler_extension_surface_descriptor_v1() noexcept {
    open_compiler_extension_surface_v1 descriptor;
    descriptor.capabilities = pass_management_v1 | pipeline_configuration_v1
        | semantic_rewrite_v1 | planning_rewrite_v1 | realization_rewrite_v1
        | stage_replacement_v1 | extension_registration_v1 | opaque_forwarding_v1
        | capability_negotiation_v1 | same_compilation_staging_v1
        | compiled_transform_cache_v1 | explicit_trust_policy_v1 | cold_provenance_v1;
    descriptor.pipeline_phase_count = static_cast<std::uint32_t>(pipeline_phase_count_v1);
    descriptor.pipeline_stage_count = static_cast<std::uint32_t>(pipeline_stage_count_v1);
    return descriptor;
}

}  // namespace cellerator::compiler::pass::v1
