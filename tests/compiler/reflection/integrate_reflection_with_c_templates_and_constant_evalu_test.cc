#include <Cellerator/compiler/reflection/integrate_reflection_with_c_templates_and_constant_evalu_v1.hh>
#include <cassert>
using namespace cellerator::compiler::reflection::v1;
using p=reflected_numeric_property_v1<32,true,64>;
static_assert(std::is_same_v<reflected_scalar_v1<p>,float>);static_assert(vectorizable_v1<p>());
int main(){typed_ir_handle_v1<handle_kind_v1::field> h{{1,1,1,1,handle_kind_v1::field,availability_phase_v1::semantic,handle_lifetime_v1::compilation}};assert(h.correct_kind());template_reflection_link_anchor_v1();}
