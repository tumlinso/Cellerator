#include <Cellerator/compiler/reflection/implement_reflection_of_realization_ir_v1.hh>
namespace cellerator::compiler::reflection::v1 {
bool validate_reflected_realization_v1(const reflected_realization_v1&r)noexcept{return r.handle.kind==handle_kind_v1::selected_realization&&!r.backend.empty()&&!r.selected_cover.empty()&&!r.extents.empty()&&!r.projections.empty()&&!r.stage_graph.empty()&&r.structure_epoch&&r.value_generation;}
bool realization_is_accelerated_v1(const reflected_realization_v1&r)noexcept{return r.backend=="cuda"||r.backend=="hip";}
}
