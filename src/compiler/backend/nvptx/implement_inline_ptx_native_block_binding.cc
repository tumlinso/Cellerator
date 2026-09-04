#include <Cellerator/compiler/backend/nvptx/implement_inline_ptx_native_block_binding_v1.hh>

#include <algorithm>
#include <cctype>
#include <unordered_map>
#include <unordered_set>

namespace Cellerator::compiler::backend::nvptx {
namespace {

inline_native_block_binding_v1 rejected(const inline_native_block_status_v1 status,
                                        const char* diagnostic) {
    inline_native_block_binding_v1 result;
    result.status = status;
    result.diagnostic = diagnostic;
    return result;
}

bool valid_sha256(const std::string& value) {
    return value.size() == 64u && std::all_of(value.begin(), value.end(), [](const char character) {
        return std::isxdigit(static_cast<unsigned char>(character)) != 0;
    });
}

bool parse_target(const std::string& target, std::uint16_t* major, std::uint16_t* minor) {
    if (target.size() < 5u || target.compare(0u, 3u, "sm_") != 0) return false;
    unsigned number = 0u;
    for (std::size_t offset = 3u; offset < target.size(); ++offset) {
        if (!std::isdigit(static_cast<unsigned char>(target[offset]))) return false;
        number = number * 10u + static_cast<unsigned>(target[offset] - '0');
    }
    if (number < 10u || number > 999u) return false;
    *major = static_cast<std::uint16_t>(number / 10u);
    *minor = static_cast<std::uint16_t>(number % 10u);
    return *major != 0u;
}

bool exact_names(const std::vector<std::string>& declared,
                 const std::vector<native_block_value_binding_v1>& bindings) {
    if (declared.size() != bindings.size()) return false;
    std::unordered_set<std::string> seen;
    for (const auto& name : declared) {
        const auto match = std::find_if(bindings.begin(), bindings.end(), [&](const auto& binding) {
            return binding.source_name == name;
        });
        if (match == bindings.end() || match->ptx_register == 0u ||
            !seen.insert(match->source_name).second) return false;
    }
    return true;
}

bool known_clobbers(const std::vector<std::string>& clobbers) {
    for (const auto& clobber : clobbers) {
        if (clobber != "memory" && clobber != "condition_codes" && clobber != "predicate") {
            return false;
        }
    }
    return true;
}

}  // namespace

inline_native_block_binding_v1 bind_inline_native_block_v1(
    const inline_native_block_request_v1& request) {
    using frontend::parser::native_backend_kind_v1;
    const auto& fragment = request.fragment;
    if (fragment.backend != native_backend_kind_v1::ptx &&
        fragment.backend != native_backend_kind_v1::cuda &&
        fragment.backend != native_backend_kind_v1::raw_native) {
        return rejected(inline_native_block_status_v1::invalid_fragment,
                        "inline native binding requires a PTX, CUDA, or raw-native fragment");
    }
    if (fragment.payload.empty() || fragment.fallback.empty() || fragment.inputs.empty() ||
        fragment.outputs.empty() || (fragment.clobbers.empty() && fragment.effects.empty())) {
        return rejected(inline_native_block_status_v1::invalid_fragment,
                        "payload, typed I/O, effects or clobbers, and exact fallback are required");
    }

    inline_native_block_binding_v1 result;
    result.backend = fragment.backend;
    if (!parse_target(fragment.target, &result.target_sm_major, &result.target_sm_minor)) {
        return rejected(inline_native_block_status_v1::invalid_target,
                        "native target must be an explicit sm_NN predicate");
    }
    if (!exact_names(fragment.inputs, request.inputs) ||
        !exact_names(fragment.outputs, request.outputs)) {
        return rejected(inline_native_block_status_v1::invalid_binding,
                        "every declared input and output requires one typed register binding");
    }
    std::unordered_set<std::uint32_t> output_registers;
    for (const auto& output : request.outputs) {
        if (!output_registers.insert(output.ptx_register).second) {
            return rejected(inline_native_block_status_v1::invalid_binding,
                            "output register bindings must be unique");
        }
    }
    if (request.declared_effects == native_block_effect_none_v1 ||
        (request.declared_effects & ~(native_block_effect_read_v1 | native_block_effect_write_v1 |
                                      native_block_effect_order_v1 |
                                      native_block_effect_synchronize_v1 |
                                      native_block_effect_escape_v1 |
                                      native_block_effect_opaque_v1)) != 0u) {
        return rejected(inline_native_block_status_v1::invalid_contract,
                        "a recognized memory, order, synchronization, escape, or opaque effect is required");
    }
    if (request.provenance.path.empty() || request.provenance.end_offset <= request.provenance.begin_offset ||
        !valid_sha256(request.provenance.content_sha256)) {
        return rejected(inline_native_block_status_v1::invalid_provenance,
                        "source path, ordered range, and SHA-256 provenance are required");
    }
    if (request.trust == native_block_trust_v1::unsafe && !request.unsafe_acknowledged) {
        return rejected(inline_native_block_status_v1::unsafe_not_acknowledged,
                        "unsafe native binding requires explicit acknowledgement");
    }

    const bool bypass = request.trust == native_block_trust_v1::unsafe;
    if (!bypass && !known_clobbers(fragment.clobbers)) {
        return rejected(inline_native_block_status_v1::invalid_contract,
                        "safe and trusted blocks may only name modeled clobbers");
    }
    if (request.trust == native_block_trust_v1::safe) {
        if (fragment.backend == native_backend_kind_v1::ptx && request.typed_ptx_kernel == nullptr) {
            return rejected(inline_native_block_status_v1::invalid_contract,
                            "safe PTX requires a typed operation model");
        }
        if (request.typed_ptx_kernel != nullptr &&
            validate_direct_ptx_kernel_v1(*request.typed_ptx_kernel, result.target_sm_major,
                                          result.target_sm_minor) !=
                direct_ptx_model_status_v1::success) {
            return rejected(inline_native_block_status_v1::invalid_contract,
                            "typed PTX operation model does not validate for the target");
        }
    }

    result.status = inline_native_block_status_v1::success;
    result.inputs = request.inputs;
    result.outputs = request.outputs;
    result.clobbers = fragment.clobbers;
    result.effects = request.declared_effects;
    result.exact_fallback = fragment.fallback;
    result.provenance = request.provenance;
    result.validation_bypassed = bypass;
    result.diagnostic = bypass ? "explicit unsafe block: semantic payload validation bypassed"
                               : request.trust == native_block_trust_v1::trusted
                                     ? "trusted block: declared contracts retained"
                                     : "safe block: typed payload and contracts validated";
    return result;
}

}  // namespace Cellerator::compiler::backend::nvptx
