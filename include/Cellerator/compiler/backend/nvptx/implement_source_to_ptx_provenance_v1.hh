#pragma once

#include <Cellerator/compiler/backend/nvptx/implement_ptx_emission_and_ptxas_assembly_v1.hh>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace Cellerator::compiler::backend::nvptx {

struct source_operation_provenance_v1 {
    std::uint64_t field_identity = 0u;
    std::uint64_t operation_identity = 0u;
    std::string path;
    std::uint32_t begin_line = 0u;
    std::uint32_t begin_column = 0u;
    std::uint32_t end_line = 0u;
    std::uint32_t end_column = 0u;
    std::string content_sha256;
};

struct assembled_resource_provenance_v1 {
    std::uint64_t identity = 0u;
    ptxas_resource_diagnostics_v1 resources;
    std::string assembler_version;
    std::string image_sha256;
};

struct source_to_ptx_provenance_entry_v1 {
    source_operation_provenance_v1 source;
    std::uint64_t semantic_node_identity = 0u;
    std::uint64_t planning_node_identity = 0u;
    std::uint64_t realization_node_identity = 0u;
    std::uint32_t ptx_begin_line = 0u;
    std::uint32_t ptx_end_line = 0u;
    std::uint64_t resource_report_identity = 0u;
};

struct source_to_ptx_provenance_sidecar_v1 {
    std::uint32_t schema_version = 1u;
    std::string ptx_sha256;
    std::vector<source_to_ptx_provenance_entry_v1> entries;
    std::vector<assembled_resource_provenance_v1> resource_reports;
};

enum class source_to_ptx_provenance_status_v1 : std::uint8_t {
    success = 0u,
    invalid_header,
    invalid_source,
    invalid_node_identity,
    invalid_ptx_range,
    invalid_resource_reference,
    duplicate_identity,
};

[[nodiscard]] source_to_ptx_provenance_status_v1 validate_source_to_ptx_provenance_v1(
    const source_to_ptx_provenance_sidecar_v1& sidecar,
    std::string* diagnostic = nullptr) noexcept;

[[nodiscard]] std::vector<const source_to_ptx_provenance_entry_v1*>
find_ptx_for_source_operation_v1(const source_to_ptx_provenance_sidecar_v1& sidecar,
                                 std::uint64_t field_identity,
                                 std::uint64_t operation_identity) noexcept;

[[nodiscard]] const source_to_ptx_provenance_entry_v1* find_source_for_ptx_line_v1(
    const source_to_ptx_provenance_sidecar_v1& sidecar,
    std::uint32_t ptx_line) noexcept;

[[nodiscard]] const assembled_resource_provenance_v1* find_resource_report_v1(
    const source_to_ptx_provenance_sidecar_v1& sidecar,
    const source_to_ptx_provenance_entry_v1& entry) noexcept;

}  // namespace Cellerator::compiler::backend::nvptx
