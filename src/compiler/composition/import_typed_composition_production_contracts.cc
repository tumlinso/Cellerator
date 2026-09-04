#include <Cellerator/compiler/composition/import_typed_composition_production_contracts_v1.hh>
#include <algorithm>
namespace Cellerator::compiler::composition {
bool validate_typed_production_contract_v1(const typed_production_contract_v1 &c,std::string *error){
 auto fail=[&](const char*m){if(error)*error=m;return false;};
 if(c.stable_name.empty()||c.verifier.empty())return fail("production name and verifier are required");
 if(c.identity_rule.empty()||c.order_rule.empty()||c.generation_rule.empty())return fail("identity, order, and generation rules are required");
 const auto inputs=std::count_if(c.values.begin(),c.values.end(),[](const auto&v){return v.role==production_value_role_v1::input;});
 const auto outputs=std::count_if(c.values.begin(),c.values.end(),[](const auto&v){return v.role==production_value_role_v1::output;});
 if(inputs==0||outputs!=1)return fail("a production requires inputs and exactly one output");
 for(const auto&v:c.values)if(v.name.empty()||v.type.empty()||v.domain_identity.empty()||v.order_identity.empty())return fail("typed values require biological identity and order");
 for(const auto&q:c.coverage){if(q.universe.empty()||q.terms.empty())return fail("coverage equations must be explicit");for(const auto&t:q.terms)if(t.value.empty()||t.coefficient==0)return fail("coverage terms must be nonzero");}
 if(c.effects.empty())return fail("effects must be explicit");
 return true;
}
} // namespace Cellerator::compiler::composition
