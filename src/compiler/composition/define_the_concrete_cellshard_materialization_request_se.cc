#include <Cellerator/compiler/composition/define_the_concrete_cellshard_materialization_request_se_v1.hh>
namespace Cellerator::compiler::composition {
std::optional<cellshard_materialization_request_v1> make_cellshard_materialization_request_v1(const portable_schedule_v1&s,std::uint64_t e,std::uint64_t g,std::uint64_t b,std::vector<std::string>t){const auto id=portable_schedule_identity_v1(s);if(!id||!e||!b||t.empty()||s.atom_requirements.empty())return std::nullopt;return cellshard_materialization_request_v1{id,e,g,b,s.atom_requirements,std::move(t),"opaque-cellerator-execution-image-v1"};}
}
