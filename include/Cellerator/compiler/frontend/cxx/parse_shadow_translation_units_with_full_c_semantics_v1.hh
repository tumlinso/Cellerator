#pragma once

#include <Cellerator/compiler/frontend/cxx/create_the_c_compilation_invocation_bridge_v1.hh>
#include <Cellerator/compiler/frontend/cxx/freeze_the_upstream_clang_adapter_boundary_v1.hh>

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace Cellerator::compiler::frontend::cxx {

inline constexpr std::uint32_t shadow_translation_unit_schema_version_v1 = 1;

enum class shadow_translation_unit_status_v1 : std::uint8_t {
    success = 0,
    null_output,
    schema_mismatch,
    unsupported_llvm_major,
    missing_invocation,
    empty_source,
    clang_parse_failed,
    semantic_errors,
};

struct shadow_translation_unit_request_v1 {
    std::uint32_t schema_version = shadow_translation_unit_schema_version_v1;
    std::uint32_t llvm_major = 18;
    const cxx_compilation_invocation_v1* invocation = nullptr;
    std::string virtual_filename = "cellerator-shadow.cc";
    std::string source;
};

class shadow_translation_unit_v1 {
public:
    shadow_translation_unit_v1() noexcept;
    ~shadow_translation_unit_v1();
    shadow_translation_unit_v1(shadow_translation_unit_v1&&) noexcept;
    shadow_translation_unit_v1& operator=(shadow_translation_unit_v1&&) noexcept;

    shadow_translation_unit_v1(const shadow_translation_unit_v1&) = delete;
    shadow_translation_unit_v1& operator=(const shadow_translation_unit_v1&) = delete;

    const upstream_clang_adapter_v1& adapter() const noexcept;
    const std::vector<std::string>& errors() const noexcept;
    const std::vector<std::string>& warnings() const noexcept;
    std::string_view virtual_filename() const noexcept;

private:
    struct implementation;
    std::unique_ptr<implementation> implementation_;

    friend shadow_translation_unit_status_v1 parse_shadow_translation_unit_v1(
        const shadow_translation_unit_request_v1&,
        shadow_translation_unit_v1*) noexcept;
};

shadow_translation_unit_status_v1 parse_shadow_translation_unit_v1(
    const shadow_translation_unit_request_v1& request,
    shadow_translation_unit_v1* translation_unit) noexcept;

}  // namespace Cellerator::compiler::frontend::cxx
