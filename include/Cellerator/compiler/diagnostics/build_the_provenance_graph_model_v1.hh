#pragma once
#include <cstdint>
#include <vector>
namespace cellerator::compiler::diagnostics::v1 {
enum class provenance_kind:std::uint8_t{source=0,ast,semantic_ir,profile_evidence,pass,planning_alternative,selection,realization_stage,generated_source,backend_object,native_symbol};
struct provenance_node{std::uint64_t id=0;provenance_kind kind=provenance_kind::source;};
struct provenance_edge{std::uint64_t from=0,to=0;};
struct provenance_graph{std::vector<provenance_node> nodes;std::vector<provenance_edge> edges;};
[[nodiscard]] bool valid_provenance_graph(const provenance_graph&) noexcept;
[[nodiscard]] std::vector<std::uint64_t> query_provenance(const provenance_graph&,std::uint64_t,bool reverse);
}
