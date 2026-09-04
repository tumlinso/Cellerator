#include <Cellerator/compiler/frontend/cxx/freeze_the_upstream_clang_adapter_boundary_v1.hh>

#include <array>
#include <utility>

namespace Cellerator::compiler::frontend::cxx {
namespace {

bool supported_llvm_major(std::uint32_t major) noexcept {
    return major == minimum_supported_llvm_major_v1 ||
           major == primary_supported_llvm_major_v1;
}

upstream_clang_adapter_status_v1 validate_object(
    const upstream_clang_object_v1& object,
    upstream_clang_object_kind_v1 expected_kind,
    std::uint32_t expected_llvm_major) noexcept {
    if (object.address == nullptr) {
        return upstream_clang_adapter_status_v1::missing_required_object;
    }
    if (object.kind != expected_kind) {
        return upstream_clang_adapter_status_v1::object_kind_mismatch;
    }
    if (object.llvm_major != expected_llvm_major) {
        return upstream_clang_adapter_status_v1::llvm_version_mismatch;
    }
    return upstream_clang_adapter_status_v1::success;
}

template <class Adapter>
upstream_clang_adapter_status_v1 validate_record(const Adapter& adapter) noexcept {
    if (adapter.schema_version != upstream_clang_adapter_schema_version_v1) {
        return upstream_clang_adapter_status_v1::schema_mismatch;
    }
    if (adapter.record_bytes != sizeof(Adapter)) {
        return upstream_clang_adapter_status_v1::record_size_mismatch;
    }
    if (!supported_llvm_major(adapter.llvm_major)) {
        return upstream_clang_adapter_status_v1::unsupported_llvm_major;
    }

    const std::array<std::pair<const upstream_clang_object_v1*, upstream_clang_object_kind_v1>, 5>
        objects{{
            {&adapter.ast_context, upstream_clang_object_kind_v1::ast_context},
            {&adapter.sema, upstream_clang_object_kind_v1::sema},
            {&adapter.preprocessor, upstream_clang_object_kind_v1::preprocessor},
            {&adapter.diagnostics, upstream_clang_object_kind_v1::diagnostics},
            {&adapter.tooling, upstream_clang_object_kind_v1::tooling},
        }};
    for (const auto& [object, expected_kind] : objects) {
        const auto status = validate_object(*object, expected_kind, adapter.llvm_major);
        if (status != upstream_clang_adapter_status_v1::success) {
            return status;
        }
    }
    return upstream_clang_adapter_status_v1::success;
}

}  // namespace

upstream_clang_adapter_status_v1 bind_upstream_clang_adapter_v1(
    const upstream_clang_adapter_request_v1& request,
    upstream_clang_adapter_v1* adapter) noexcept {
    if (adapter == nullptr) {
        return upstream_clang_adapter_status_v1::null_output;
    }
    const auto status = validate_record(request);
    if (status != upstream_clang_adapter_status_v1::success) {
        return status;
    }
    *adapter = {
        upstream_clang_adapter_schema_version_v1,
        sizeof(upstream_clang_adapter_v1),
        request.llvm_major,
        request.llvm_minor,
        request.ast_context,
        request.sema,
        request.preprocessor,
        request.diagnostics,
        request.tooling,
    };
    return upstream_clang_adapter_status_v1::success;
}

upstream_clang_adapter_status_v1 validate_upstream_clang_adapter_v1(
    const upstream_clang_adapter_v1& adapter) noexcept {
    return validate_record(adapter);
}

}  // namespace Cellerator::compiler::frontend::cxx
