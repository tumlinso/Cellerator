#include <Cellerator/compiler/driver/forward_and_remap_downstream_diagnostics_v1.hh>
namespace cellerator::compiler::driver {
downstream_diagnostic_v1 remap_downstream_diagnostic_v1(downstream_diagnostic_v1 diagnostic, const std::vector<source_map_entry_v1>& maps) { auto remap = [&](source_position_v1& position) { for (const auto& map : maps) if (position.file == map.generated_file && position.line >= map.generated_first_line && position.line <= map.generated_last_line) { position.file = map.source_file; position.line = map.source_first_line + position.line - map.generated_first_line; return; } }; remap(diagnostic.begin); remap(diagnostic.end); return diagnostic; }
}  // namespace cellerator::compiler::driver
