#include <Cellerator/compiler/backend/freeze_the_backend_provider_abi_v1.hh>

namespace cellerator::compiler::backend::v1 {
namespace {

bool valid_string(backend_string_view_v1 value) noexcept {
    return value.data != nullptr && value.size != 0;
}

void diagnose(backend_diagnostic_sink_v1 sink, backend_status_v1 status,
              const char* message, std::size_t size) noexcept {
    if (sink.emit != nullptr) {
        sink.emit(sink.context, status, {message, size});
    }
}

}  // namespace

backend_status_v1 validate_backend_provider_v1(
    const backend_provider_v1& provider) noexcept {
    if (provider.abi_version != backend_provider_abi_version_v1 ||
        provider.struct_size < sizeof(backend_provider_v1)) {
        return backend_status_v1::unsupported_abi;
    }
    if (!valid_string(provider.toolchain.provider) ||
        !valid_string(provider.toolchain.compiler) ||
        !valid_string(provider.toolchain.compiler_version) ||
        !valid_string(provider.toolchain.build_identity)) {
        return backend_status_v1::unavailable_toolchain;
    }
    if (provider.discover_targets == nullptr ||
        provider.query_capabilities == nullptr ||
        provider.realization_admissible == nullptr ||
        provider.emit_object == nullptr) {
        return backend_status_v1::invalid_argument;
    }
    return backend_status_v1::success;
}

backend_status_v1 emit_backend_object_v1(
    const backend_provider_v1& provider, backend_target_v1 target,
    backend_realization_view_v1 realization, backend_object_buffer_v1* object,
    backend_diagnostic_sink_v1 diagnostics) noexcept {
    const auto provider_status = validate_backend_provider_v1(provider);
    if (provider_status != backend_status_v1::success) {
        diagnose(diagnostics, provider_status, "invalid backend provider", 24);
        return provider_status;
    }
    if (!valid_string(target.triple) || realization.data == nullptr ||
        realization.size == 0 || realization.schema_version == 0 ||
        object == nullptr || (object->capacity != 0 && object->data == nullptr)) {
        diagnose(diagnostics, backend_status_v1::invalid_argument,
                 "invalid backend emission request", 32);
        return backend_status_v1::invalid_argument;
    }
    std::uint64_t capabilities = 0;
    auto status = provider.query_capabilities(
        provider.context, target, &capabilities, diagnostics);
    if (status != backend_status_v1::success) {
        return status;
    }
    if ((capabilities & backend_capability_ordinary_object_v1) == 0) {
        diagnose(diagnostics, backend_status_v1::unsupported_target,
                 "target cannot emit ordinary objects", 35);
        return backend_status_v1::unsupported_target;
    }
    status = provider.realization_admissible(
        provider.context, target, realization, diagnostics);
    if (status != backend_status_v1::success) {
        return status;
    }
    return provider.emit_object(
        provider.context, target, realization, object, diagnostics);
}

const backend_provider_abi_receipt_v1& get_backend_provider_abi_receipt_v1() noexcept {
    static constexpr backend_provider_abi_receipt_v1 receipt{
        backend_provider_abi_version_v1,
        sizeof(backend_provider_v1),
        true,
        true,
        true,
    };
    return receipt;
}

}  // namespace cellerator::compiler::backend::v1
