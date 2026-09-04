#include <Cellerator/compiler/tooling/freeze_the_celleratord_architecture_v1.hh>

#include <array>

namespace Cellerator::compiler::tooling {

celleratord_architecture_status_v1 validate_celleratord_architecture_v1(
    const celleratord_architecture_v1 &architecture) noexcept {
    if (architecture.schema_version != celleratord_architecture_schema_v1)
        return celleratord_architecture_status_v1::schema_mismatch;
    if (architecture.clangd_command.empty())
        return celleratord_architecture_status_v1::missing_clangd_command;
    if (architecture.clangd_version_requirement.empty())
        return celleratord_architecture_status_v1::missing_clangd_version;
    if (architecture.compiler_snapshot_version == 0)
        return celleratord_architecture_status_v1::missing_snapshot_version;
    if (architecture.source_map_version == 0)
        return celleratord_architecture_status_v1::missing_source_map_version;
    if (architecture.permanent_clang_fork)
        return celleratord_architecture_status_v1::permanent_fork_forbidden;
    return celleratord_architecture_status_v1::valid;
}

lsp_feature_owner_v1 route_lsp_feature_v1(std::string_view method) noexcept {
    static constexpr std::array<std::string_view, 7> cellerator_methods{{
        "cellerator/ast", "cellerator/ceir", "cellerator/fieldPlan",
        "cellerator/profile", "cellerator/realization", "cellerator/sourceMap",
        "cellerator/verify"
    }};
    for (const auto owned : cellerator_methods)
        if (method == owned)
            return lsp_feature_owner_v1::lib_cellerator_snapshot;
    return lsp_feature_owner_v1::upstream_clangd_worker;
}

bool valid_snapshot_ticket_v1(const celleratord_snapshot_ticket_v1 &ticket) noexcept {
    return ticket.document_revision != 0 && ticket.clangd_document_version >= 0
        && ticket.compiler_snapshot && !ticket.source_map_identity.empty();
}

bool valid_source_mapping_v1(const celleratord_source_mapping_v1 &mapping) noexcept {
    return !mapping.generated_uri.empty() && !mapping.original_uri.empty()
        && mapping.generated_begin <= mapping.generated_end
        && mapping.original_begin <= mapping.original_end;
}

} // namespace Cellerator::compiler::tooling
