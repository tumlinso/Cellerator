#include "tooling_model.hh"
namespace cellerator::compiler::tooling::v1 {
biological_hover describe_biological_relation(std::string_view d){biological_hover h{"gene","regulatory","gene:canonical","cell:sample","sparse","forward","f32xf32->f32","values mutable / structure immutable","structure:1","generation:current","model.cell:1"};if(d.find("transpose")!=d.npos)h.orientation="transpose";if(d.find("learned")!=d.npos)h.mutability="values learned / structure immutable";return h;}
}
