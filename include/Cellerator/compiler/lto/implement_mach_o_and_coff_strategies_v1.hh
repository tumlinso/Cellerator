#pragma once
#include <Cellerator/compiler/lto/freeze_the_ceir_companion_object_artifact_contract_v1.hh>
#include <string>
namespace cellerator::compiler::lto::v1 {
struct platform_ceir_strategy_v1{object_format_v1 format=object_format_v1::sidecar;std::string segment,section,symbol,sidecar_suffix;bool toolchain_available=false,use_sidecar=false;};
[[nodiscard]] platform_ceir_strategy_v1 select_platform_ceir_strategy_v1(object_format_v1,bool toolchain_available);
[[nodiscard]] bool equivalent_platform_content_v1(const ceir_companion_artifact_v1&,const ceir_companion_artifact_v1&)noexcept;
}
