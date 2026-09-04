#include <Cellerator/compiler/backend/implement_backend_registry_and_selection_v1.hh>

#include <algorithm>

namespace cellerator::compiler::backend::v1 {
namespace {

std::string_view view(backend_string_view_v1 value) noexcept {
    return value.data == nullptr ? std::string_view{} :
        std::string_view(value.data, value.size);
}

bool usable(const backend_registry_entry_v1& entry) noexcept {
    return !view(entry.name).empty() && !view(entry.source_fragment).empty() &&
        entry.provider != nullptr &&
        validate_backend_provider_v1(*entry.provider) == backend_status_v1::success;
}

}  // namespace

backend_selection_status_v1 backend_registry_v1::register_backend(
    backend_registry_entry_v1 entry) noexcept {
    if (!usable(entry)) return backend_selection_status_v1::invalid_entry;
    try {
        entries_.push_back(entry);
        return backend_selection_status_v1::selected;
    } catch (...) {
        return backend_selection_status_v1::unavailable;
    }
}

backend_selection_result_v1 backend_registry_v1::select(
    const backend_selection_request_v1& request,
    backend_diagnostic_sink_v1 diagnostics) const noexcept {
    const backend_registry_entry_v1* best = nullptr;
    bool ambiguous = false;
    for (const auto& entry : entries_) {
        if (!usable(entry)) continue;
        if (request.policy == backend_selection_policy_v1::force_named &&
            view(entry.name) != view(request.forced_name)) continue;
        std::uint64_t capabilities = 0;
        if (entry.provider->query_capabilities(
                entry.provider->context, request.target, &capabilities, diagnostics) !=
            backend_status_v1::success) continue;
        if ((capabilities & request.required_capabilities) !=
            request.required_capabilities) continue;
        if (best == nullptr || entry.priority > best->priority) {
            best = &entry;
            ambiguous = false;
        } else if (entry.priority == best->priority) {
            ambiguous = true;
        }
    }
    if (best == nullptr) {
        return {request.policy == backend_selection_policy_v1::force_named
                    ? backend_selection_status_v1::forced_backend_unavailable
                    : backend_selection_status_v1::unavailable,
                nullptr, false};
    }
    if (ambiguous && request.policy != backend_selection_policy_v1::force_named) {
        return {backend_selection_status_v1::ambiguous, nullptr, false};
    }
    const bool fallback = request.allow_conventional_fallback && best->priority == 0;
    return {backend_selection_status_v1::selected, best, fallback};
}

}  // namespace cellerator::compiler::backend::v1
