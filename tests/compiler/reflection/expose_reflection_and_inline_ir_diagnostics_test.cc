#include <Cellerator/compiler/reflection/expose_reflection_and_inline_ir_diagnostics_v1.hh>
#include <cassert>
using namespace cellerator::compiler::reflection::v1;
int main(){reflection_diagnostic_v1 d{reflection_diagnostic_code_v1::stale_handle,"x.cell:4","stale handle","generation 4","generation 5",invalidate_planning_v1|invalidate_realization_v1,false};auto human=format_reflection_diagnostic_v1(d);auto machine=serialize_reflection_diagnostics_v1({d});assert(human=="error R3 x.cell:4: stale handle expected=generation 4 observed=generation 5 invalidates=12");assert(machine.find("\"code\":3")!=std::string::npos&&machine.find("\"invalidations\":12")!=std::string::npos);}
