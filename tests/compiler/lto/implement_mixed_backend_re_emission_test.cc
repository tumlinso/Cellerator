#include <Cellerator/compiler/lto/implement_mixed_backend_re_emission_v1.hh>
#include <cassert>
using namespace cellerator::compiler::lto::v1;
int main(){auto p=plan_mixed_backend_re_emission_v1({{"plain.o","plain.o",emission_backend_v1::conventional,true,false},{"cpu.ceir","cpu.o",emission_backend_v1::cellerator_cpu,true,true},{"cuda.ceir","cuda.o",emission_backend_v1::cellerator_cuda,false,true},{"native.ceir","native.o",emission_backend_v1::native,false,false}});assert(p.size()==4&&p[0].action==emission_action_v1::retain&&p[1].action==emission_action_v1::reemit&&p[2].action==emission_action_v1::retain&&p[3].action==emission_action_v1::reemit);return p[0].output=="plain.o"?0:1;}
