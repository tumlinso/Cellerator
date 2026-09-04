#pragma once

#include <Cellerator/compiler/pass/define_extensible_operation_type_and_attribute_registrat_v1.hh>

#include <cstdint>
#include <string>

namespace cellerator::compiler::pass::v1 {

enum class extension_handling_mode_v1 : std::uint8_t {
    preserve_only = 0,
    inspect_only,
    external_lowered,
    fully_understood,
};

struct extension_capability_request_v1 {
    std::string qualified_name;
    std::uint32_t required_protocols = 0;
    std::uint32_t compiler_protocols = 0;
    std::uint32_t backend_protocols = 0;
    bool external_lowering_available = false;
};

struct extension_capability_receipt_v1 {
    extension_handling_mode_v1 mode = extension_handling_mode_v1::preserve_only;
    std::uint32_t missing_compiler_protocols = 0;
    std::uint32_t missing_backend_protocols = 0;
    std::string diagnostic;
};

[[nodiscard]] extension_capability_receipt_v1 negotiate_extension_capability_v1(
    const extension_capability_request_v1& request);

}  // namespace cellerator::compiler::pass::v1
