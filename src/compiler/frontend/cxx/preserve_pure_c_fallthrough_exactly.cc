#include <Cellerator/compiler/frontend/cxx/preserve_pure_c_fallthrough_exactly_v1.hh>

#include <sys/resource.h>

#include <chrono>
#include <string>

namespace Cellerator::compiler::frontend::cxx {
namespace {

bool has_activation(const std::string& source) {
    std::string visible;
    visible.reserve(source.size());
    bool line_comment = false;
    bool block_comment = false;
    bool string_literal = false;
    bool character_literal = false;
    bool escape = false;
    for (std::size_t index = 0; index < source.size(); ++index) {
        const char current = source[index];
        const char next = index + 1 < source.size() ? source[index + 1] : '\0';
        if (line_comment) {
            if (current == '\n') { line_comment = false; visible.push_back(current); }
            continue;
        }
        if (block_comment) {
            if (current == '*' && next == '/') { block_comment = false; ++index; }
            continue;
        }
        if (!string_literal && !character_literal && current == '/' && next == '/') {
            line_comment = true; ++index; continue;
        }
        if (!string_literal && !character_literal && current == '/' && next == '*') {
            block_comment = true; ++index; continue;
        }
        if (!character_literal && current == '"' && !escape) string_literal = !string_literal;
        if (!string_literal && current == '\'' && !escape) character_literal = !character_literal;
        if (string_literal || character_literal) {
            visible.push_back(' ');
            escape = current == '\\' && !escape;
            if (current != '\\') escape = false;
            continue;
        }
        escape = false;
        visible.push_back(current);
    }
    return visible.find("#pragma cellerator") != std::string::npos ||
           visible.find("[[cellerator::") != std::string::npos ||
           visible.find("import cellerator") != std::string::npos ||
           visible.find("cellerator_ir{") != std::string::npos;
}

std::uint64_t peak_rss_kib() noexcept {
    rusage usage{};
    return getrusage(RUSAGE_SELF, &usage) == 0
        ? static_cast<std::uint64_t>(usage.ru_maxrss) : 0;
}

}  // namespace

pure_cxx_fallthrough_status_v1 plan_pure_cxx_fallthrough_v1(
    const pure_cxx_fallthrough_request_v1& request,
    pure_cxx_fallthrough_plan_v1* plan) noexcept {
    if (plan == nullptr || request.original_driver_arguments.empty()) {
        return pure_cxx_fallthrough_status_v1::empty_driver_arguments;
    }
    if (request.schema_version != pure_cxx_fallthrough_schema_version_v1) {
        return pure_cxx_fallthrough_status_v1::schema_mismatch;
    }
    const auto start = std::chrono::steady_clock::now();
    const bool activated = has_activation(request.source);
    plan->mode = activated ? pure_cxx_fallthrough_mode_v1::cellerator_frontend
                           : pure_cxx_fallthrough_mode_v1::direct_driver;
    plan->construct_cellerator_ast_or_ir = activated;
    plan->forwarded_driver_arguments = request.original_driver_arguments;
    plan->frontend_scan_nanoseconds = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start).count());
    plan->peak_resident_kib = peak_rss_kib();
    return pure_cxx_fallthrough_status_v1::success;
}

}  // namespace Cellerator::compiler::frontend::cxx
