#pragma once
#include <Cellerator/compiler/discovery/import_the_common_jbc_atom_identity_adapters_v1.hh>
#include <cstdint>
#include <vector>
namespace Cellerator::compiler::discovery {
struct operation_trace_event_v1 {
    persistent_atom_identity_v1 relation{}, source_domain{}, destination_domain{};
    persistent_atom_identity_v1 operation{}, numeric_policy{}, profile{}, field{};
};
struct trace_motif_v1 {
    persistent_atom_identity_v1 motif_identity{};
    std::uint32_t sequence_length = 0;
    std::uint32_t occurrence_count = 0;
    std::vector<operation_trace_event_v1> sequence;
};
enum class trace_discovery_status_v1 : std::uint8_t { success=0, invalid_event, invalid_limit };
[[nodiscard]] trace_discovery_status_v1 discover_relation_and_trace_motifs_v1(
    const std::vector<operation_trace_event_v1>& trace, std::uint32_t maximum_length,
    std::uint32_t minimum_occurrences, std::vector<trace_motif_v1>* output) noexcept;
}  // namespace Cellerator::compiler::discovery
