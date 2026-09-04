#pragma once

#include <Cellerator/compiler/backend/nvptx/define_direct_ptx_typed_operation_model_v1.hh>
#include <Cellerator/compiler/frontend/parser/parse_native_backend_fragments_v1.hh>

#include <cstdint>
#include <string>
#include <vector>

namespace Cellerator::compiler::backend::nvptx {

enum class native_block_trust_v1 : std::uint8_t {
    safe = 1u,
    trusted,
    unsafe,
};

enum native_block_effect_v1 : std::uint16_t {
    native_block_effect_none_v1 = 0u,
    native_block_effect_read_v1 = 1u << 0u,
    native_block_effect_write_v1 = 1u << 1u,
    native_block_effect_order_v1 = 1u << 2u,
    native_block_effect_synchronize_v1 = 1u << 3u,
    native_block_effect_escape_v1 = 1u << 4u,
    native_block_effect_opaque_v1 = 1u << 5u,
};

struct native_block_value_binding_v1 {
    std::string source_name;
    std::uint32_t ptx_register = 0u;
    direct_ptx_type_v1 type = direct_ptx_type_v1::b32;
};

struct native_block_source_provenance_v1 {
    std::string path;
    std::uint64_t begin_offset = 0u;
    std::uint64_t end_offset = 0u;
    std::string content_sha256;
};

struct inline_native_block_request_v1 {
    frontend::parser::native_backend_fragment_v1 fragment;
    native_block_trust_v1 trust = native_block_trust_v1::safe;
    bool unsafe_acknowledged = false;
    std::vector<native_block_value_binding_v1> inputs;
    std::vector<native_block_value_binding_v1> outputs;
    std::uint16_t declared_effects = native_block_effect_none_v1;
    native_block_source_provenance_v1 provenance;
    const direct_ptx_kernel_v1* typed_ptx_kernel = nullptr;
};

enum class inline_native_block_status_v1 : std::uint8_t {
    success = 0u,
    invalid_fragment,
    invalid_target,
    invalid_binding,
    invalid_contract,
    invalid_provenance,
    unsafe_not_acknowledged,
};

struct inline_native_block_binding_v1 {
    inline_native_block_status_v1 status = inline_native_block_status_v1::invalid_fragment;
    frontend::parser::native_backend_kind_v1 backend =
        frontend::parser::native_backend_kind_v1::generated_cxx;
    std::uint16_t target_sm_major = 0u;
    std::uint16_t target_sm_minor = 0u;
    std::vector<native_block_value_binding_v1> inputs;
    std::vector<native_block_value_binding_v1> outputs;
    std::vector<std::string> clobbers;
    std::uint16_t effects = native_block_effect_none_v1;
    std::string exact_fallback;
    native_block_source_provenance_v1 provenance;
    bool validation_bypassed = false;
    std::string diagnostic;

    explicit operator bool() const noexcept {
        return status == inline_native_block_status_v1::success;
    }
};

[[nodiscard]] inline_native_block_binding_v1 bind_inline_native_block_v1(
    const inline_native_block_request_v1& request);

}  // namespace Cellerator::compiler::backend::nvptx
