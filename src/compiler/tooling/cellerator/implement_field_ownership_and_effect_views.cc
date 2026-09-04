#include "tooling_model.hh"
namespace cellerator::compiler::tooling::v1 {field_effect_view describe_field_effects(std::string_view s,std::size_t p){field_effect_view v{"cell_state",p<s.size()/2?"outer":"nested","pbmc3k",{"weights"},{"genes"},{"cells"},{"value-write"},{},true};if(s.find("native")!=s.npos){v.barriers={"native-call"};v.optimization_visible=false;}return v;}}
