#include <Cellerator/compiler/pass/implement_extension_capability_negotiation_v1.hh>

namespace cellerator::compiler::pass::v1 {

extension_capability_receipt_v1 negotiate_extension_capability_v1(
    const extension_capability_request_v1& request) {
    extension_capability_receipt_v1 receipt;
    receipt.missing_compiler_protocols =
        request.required_protocols & ~request.compiler_protocols;
    receipt.missing_backend_protocols =
        request.required_protocols & ~request.backend_protocols;
    if (request.qualified_name.empty()) {
        receipt.diagnostic = "extension has no qualified name; preserving opaquely";
        return receipt;
    }
    if (receipt.missing_compiler_protocols == 0
        && receipt.missing_backend_protocols == 0) {
        receipt.mode = extension_handling_mode_v1::fully_understood;
        return receipt;
    }
    if (request.external_lowering_available
        && (receipt.missing_backend_protocols & extension_lowering_v1) != 0) {
        receipt.mode = extension_handling_mode_v1::external_lowered;
        receipt.diagnostic = "native lowering unavailable; using declared external lowering";
        return receipt;
    }
    if ((request.compiler_protocols & extension_reflection_v1) != 0) {
        receipt.mode = extension_handling_mode_v1::inspect_only;
        receipt.diagnostic = "required protocols unavailable; extension remains inspectable and opaque";
        return receipt;
    }
    receipt.diagnostic = "required protocols unavailable; extension preserved opaquely";
    return receipt;
}

}  // namespace cellerator::compiler::pass::v1
