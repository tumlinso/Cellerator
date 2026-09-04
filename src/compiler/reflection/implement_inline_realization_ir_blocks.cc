#include <Cellerator/compiler/reflection/implement_inline_realization_ir_blocks_v1.hh>
#include <functional>
namespace cellerator::compiler::reflection::v1 {
inline_realization_status_v1 validate_inline_realization_block_v1(const inline_realization_block_v1&b)noexcept{if(b.backend.empty())return inline_realization_status_v1::missing_backend;if(b.stages.empty())return inline_realization_status_v1::missing_stage;for(const auto&x:b.bindings)if(x.name.empty()||x.type.empty()||!x.identity||!x.generation)return inline_realization_status_v1::invalid_binding;if(b.validation==inline_realization_validation_v1::unchecked&&!b.unsafe_acknowledged)return inline_realization_status_v1::unchecked_not_acknowledged;return inline_realization_status_v1::valid;}
reflected_realization_v1 override_realization_stage_v1(const reflected_realization_v1&r,const inline_realization_block_v1&b,std::size_t i){auto o=r;if(validate_inline_realization_block_v1(b)==inline_realization_status_v1::valid&&i<o.stage_graph.size()){o.backend=b.backend;o.stage_graph[i]=std::hash<std::string>{}(b.stages.front());for(const auto&n:b.native_fragments)o.native_fragments.push_back(std::hash<std::string>{}(n));}return o;}
}
