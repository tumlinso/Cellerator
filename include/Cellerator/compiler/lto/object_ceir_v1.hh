#pragma once

#include <Cellerator/compiler/lto/freeze_the_ceir_companion_object_artifact_contract_v1.hh>
#include <Cellerator/compiler/lto/implement_elf_ceir_sections_v1.hh>
#include <Cellerator/compiler/lto/implement_mach_o_and_coff_strategies_v1.hh>
#include <Cellerator/compiler/lto/implement_object_and_archive_ceir_extraction_v1.hh>
#include <Cellerator/compiler/lto/implement_portable_sidecar_fallback_v1.hh>

namespace cellerator::compiler::lto::v1 {
inline constexpr std::uint32_t object_ceir_contract_version_v1 = 1;
}
