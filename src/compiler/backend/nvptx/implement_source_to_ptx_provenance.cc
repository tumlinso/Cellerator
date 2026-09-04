#include <Cellerator/compiler/backend/nvptx/implement_source_to_ptx_provenance_v1.hh>

#include <algorithm>
#include <cctype>
#include <unordered_set>

namespace Cellerator::compiler::backend::nvptx {
namespace {

bool sha256(const std::string& value) {
    return value.size() == 64u && std::all_of(value.begin(), value.end(), [](const char character) {
        return std::isxdigit(static_cast<unsigned char>(character)) != 0;
    });
}

void report(std::string* diagnostic, const char* message) {
    if (diagnostic != nullptr) *diagnostic = message;
}

}  // namespace

source_to_ptx_provenance_status_v1 validate_source_to_ptx_provenance_v1(
    const source_to_ptx_provenance_sidecar_v1& sidecar,
    std::string* diagnostic) noexcept {
    if (sidecar.schema_version != 1u || !sha256(sidecar.ptx_sha256) || sidecar.entries.empty()) {
        report(diagnostic, "sidecar schema, PTX digest, or entries are invalid");
        return source_to_ptx_provenance_status_v1::invalid_header;
    }
    std::unordered_set<std::uint64_t> resource_identities;
    for (const auto& resource : sidecar.resource_reports) {
        if (resource.identity == 0u || resource.assembler_version.empty() ||
            !sha256(resource.image_sha256) || !resource_identities.insert(resource.identity).second) {
            report(diagnostic, "resource report identity or evidence is invalid");
            return source_to_ptx_provenance_status_v1::duplicate_identity;
        }
    }
    std::unordered_set<std::uint64_t> realization_identities;
    for (const auto& entry : sidecar.entries) {
        const auto& source = entry.source;
        if (source.field_identity == 0u || source.operation_identity == 0u || source.path.empty() ||
            source.begin_line == 0u || source.begin_column == 0u || source.end_line < source.begin_line ||
            (source.end_line == source.begin_line && source.end_column <= source.begin_column) ||
            !sha256(source.content_sha256)) {
            report(diagnostic, "source operation range or digest is invalid");
            return source_to_ptx_provenance_status_v1::invalid_source;
        }
        if (entry.semantic_node_identity == 0u || entry.planning_node_identity == 0u ||
            entry.realization_node_identity == 0u) {
            report(diagnostic, "semantic, planning, and realization identities are required");
            return source_to_ptx_provenance_status_v1::invalid_node_identity;
        }
        if (!realization_identities.insert(entry.realization_node_identity).second) {
            report(diagnostic, "realization identity is duplicated");
            return source_to_ptx_provenance_status_v1::duplicate_identity;
        }
        if (entry.ptx_begin_line == 0u || entry.ptx_end_line < entry.ptx_begin_line) {
            report(diagnostic, "PTX line range is invalid");
            return source_to_ptx_provenance_status_v1::invalid_ptx_range;
        }
        if (entry.resource_report_identity == 0u ||
            resource_identities.count(entry.resource_report_identity) == 0u) {
            report(diagnostic, "PTX entry has no assembled resource report");
            return source_to_ptx_provenance_status_v1::invalid_resource_reference;
        }
    }
    if (diagnostic != nullptr) diagnostic->clear();
    return source_to_ptx_provenance_status_v1::success;
}

std::vector<const source_to_ptx_provenance_entry_v1*> find_ptx_for_source_operation_v1(
    const source_to_ptx_provenance_sidecar_v1& sidecar,
    const std::uint64_t field_identity,
    const std::uint64_t operation_identity) noexcept {
    std::vector<const source_to_ptx_provenance_entry_v1*> result;
    for (const auto& entry : sidecar.entries) {
        if (entry.source.field_identity == field_identity &&
            entry.source.operation_identity == operation_identity) result.push_back(&entry);
    }
    std::sort(result.begin(), result.end(), [](const auto* left, const auto* right) {
        return left->ptx_begin_line < right->ptx_begin_line;
    });
    return result;
}

const source_to_ptx_provenance_entry_v1* find_source_for_ptx_line_v1(
    const source_to_ptx_provenance_sidecar_v1& sidecar,
    const std::uint32_t ptx_line) noexcept {
    for (const auto& entry : sidecar.entries) {
        if (ptx_line >= entry.ptx_begin_line && ptx_line <= entry.ptx_end_line) return &entry;
    }
    return nullptr;
}

const assembled_resource_provenance_v1* find_resource_report_v1(
    const source_to_ptx_provenance_sidecar_v1& sidecar,
    const source_to_ptx_provenance_entry_v1& entry) noexcept {
    for (const auto& resource : sidecar.resource_reports) {
        if (resource.identity == entry.resource_report_identity) return &resource;
    }
    return nullptr;
}

}  // namespace Cellerator::compiler::backend::nvptx
