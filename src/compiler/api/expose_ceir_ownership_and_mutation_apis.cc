#include <Cellerator/compiler/api/expose_ceir_ownership_and_mutation_apis_v1.hh>
namespace cellerator::compiler::api::v1 {
ceir_snapshot_v1 parse_ceir_v1(ceir_level_v1 level, std::string text) { ceir_builder_v1 b(level); b.set_text(std::move(text)); b.add_provenance("parsed"); return b.freeze(); }
std::string print_ceir_v1(const ceir_snapshot_v1& module) { return module ? module->text : std::string{}; }
ceir_builder_v1 clone_ceir_v1(const ceir_snapshot_v1& module) { ceir_builder_v1 b(module->level); b.set_text(module->text); for(const auto& p:module->provenance)b.add_provenance(p); return b; }
bool validate_ceir_v1(const ceir_snapshot_v1& module, ceir_validation_v1 mode) noexcept { return module && (mode==ceir_validation_v1::unsafe_mode || !module->text.empty()); }
std::vector<std::uint8_t> serialize_ceir_v1(const ceir_snapshot_v1& module) { if(!module)return {}; std::vector<std::uint8_t> out{static_cast<std::uint8_t>(module->level)}; out.insert(out.end(),module->text.begin(),module->text.end()); return out; }
}
