#pragma once
#include <Cellerator/compiler/pass/extension_v1.hh>
#include <Cellerator/compiler/pass/pass_v1.hh>
#include <Cellerator/compiler/pass/self_transform_v1.hh>
#include <Cellerator/compiler/reflection/reflection_v1.hh>
#include <cstdint>
namespace cellerator::compiler::api::v1 {
struct programmable_compiler_surface_v1{std::uint32_t abi_version=1;bool passes=true;bool reflection=true;bool extensions=true;bool same_compilation=true;bool explicit_trust=true;};
[[nodiscard]] programmable_compiler_surface_v1 programmable_compiler_surface() noexcept;
}
