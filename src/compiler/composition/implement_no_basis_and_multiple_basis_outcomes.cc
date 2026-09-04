#include <Cellerator/compiler/composition/implement_no_basis_and_multiple_basis_outcomes_v1.hh>
namespace Cellerator::compiler::composition {
basis_selection_v1 select_basis_outcome_v1(const std::vector<basis_outcome_v1>&xs,std::string_view profile,double fallback){basis_selection_v1 r;r.reason="no beneficial basis";for(const auto&x:xs)if(x.valid&&x.profile==profile&&x.total_cost<fallback&&(!r.selected||x.total_cost<r.selected->total_cost))r.selected=x;if(r.selected){r.use_basis=true;r.reason=r.selected->external?"selected external offer":"selected profile basis";}return r;}
}
