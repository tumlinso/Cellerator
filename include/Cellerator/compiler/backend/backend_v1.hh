#pragma once

#include <Cellerator/compiler/backend/compile_generated_c_into_ordinary_objects_v1.hh>
#include <Cellerator/compiler/backend/implement_backend_code_generation_plans_v1.hh>
#include <Cellerator/compiler/backend/freeze_the_backend_provider_abi_v1.hh>
#include <Cellerator/compiler/backend/implement_backend_registry_and_selection_v1.hh>
#include <Cellerator/compiler/backend/implement_generated_c_representation_v1.hh>
#include <Cellerator/compiler/backend/map_backend_diagnostics_to_source_and_ceir_v1.hh>

namespace cellerator::compiler::backend::v1 {
inline constexpr std::uint32_t backend_thin_waist_version_v1 = 1;
}
