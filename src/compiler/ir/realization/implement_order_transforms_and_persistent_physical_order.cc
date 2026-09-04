#include <Cellerator/compiler/ir/realization/implement_order_transforms_and_persistent_physical_order_v1.hh>
#include <algorithm>
namespace cellerator::compiler::ir::realization::v1 {
namespace { order_status_v1 fail(order_status_v1 s,std::string*e,const char*m) noexcept {if(e)*e=m;return s;} }
order_status_v1 validate_order_transform_v1(const order_transform_v1&t,std::string*e) noexcept {
    if(!valid(t.identity)||!valid(t.input.identity)||!valid(t.output.identity))return fail(order_status_v1::invalid_identity,e,"transform and order identities required");
    auto p=t.output_to_input;std::sort(p.begin(),p.end());
    for(std::size_t i=0;i<p.size();++i)if(p[i]!=i)return fail(order_status_v1::invalid_permutation,e,"order map must be a permutation");
    if(t.kind==order_stage_kind_v1::canonicalize&&t.input.order==order_class_v1::canonical)return fail(order_status_v1::redundant_canonicalize,e,"canonical input needs no canonicalization");
    if (e) e->clear();
    return order_status_v1::valid;
}
order_status_v1 validate_persistent_order_chain_v1(const persistent_order_chain_v1&c,std::string*e) noexcept {
    if(c.relations.empty())return fail(order_status_v1::disconnected_chain,e,"relation chain required");
    for(std::size_t i=0;i<c.relations.size();++i){const auto&s=c.relations[i];if(!valid(s.operation)||!valid(s.input.identity)||!valid(s.output.identity))return fail(order_status_v1::invalid_identity,e,"stage identities required");if(i&&!(c.relations[i-1].output.identity==s.input.identity))return fail(order_status_v1::disconnected_chain,e,"adjacent relation orders must connect");}
    for(const auto&t:c.boundary_transforms){auto s=validate_order_transform_v1(t,e);if(s!=order_status_v1::valid)return s;}
    if (e) e->clear();
    return order_status_v1::valid;
}
std::vector<double> apply_order_transform_v1(const std::vector<double>&v,const order_transform_v1&t){if(t.output_to_input.size()!=v.size())return{};std::vector<double>o(v.size());for(std::size_t i=0;i<v.size();++i){if(t.output_to_input[i]>=v.size())return{};o[i]=v[t.output_to_input[i]];}return o;}
} // namespace cellerator::compiler::ir::realization::v1
