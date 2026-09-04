#include <Cellerator/compiler/frontend/source/register_the_pragma_cellerator_preprocessor_contract_v1.hh>

namespace Cellerator::compiler::frontend::source {
namespace {

std::string_view trim(std::string_view value) noexcept {
    while (!value.empty() && (value.front() == ' ' || value.front() == '\t' || value.front() == '\r')) {
        value.remove_prefix(1);
    }
    while (!value.empty() && (value.back() == ' ' || value.back() == '\t' || value.back() == '\r')) {
        value.remove_suffix(1);
    }
    return value;
}

} // namespace

pragma_result_v1 handle_cellerator_pragma_v1(pragma_request_v1 request) {
    pragma_result_v1 result{};
    result.location = request.location;
    if (request.produced_by_macro) {
        result.diagnostic = pragma_diagnostic_v1::macro_produced;
        return result;
    }
    if (request.already_active) {
        result.diagnostic = pragma_diagnostic_v1::duplicate;
        return result;
    }
    if (request.saw_non_directive_token) {
        result.diagnostic = pragma_diagnostic_v1::late_activation;
        return result;
    }
    const auto payload = trim(request.payload);
    if (payload.empty()) {
        result.activate = true;
        result.revision = "0.1";
        return result;
    }
    if (payload == "0.1") {
        result.activate = true;
        result.revision.assign(payload);
        return result;
    }
    if (payload.find_first_of(" \t()[]{}") != std::string_view::npos) {
        result.diagnostic = pragma_diagnostic_v1::malformed;
    } else {
        result.diagnostic = pragma_diagnostic_v1::unknown_version;
    }
    return result;
}

} // namespace Cellerator::compiler::frontend::source
