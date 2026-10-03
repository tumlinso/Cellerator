#include <Cellerator/execution/prepared_host_sweep.hh>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>
namespace ex=cellerator::execution;
namespace ph=ex::prepared_host;
namespace nf=ph::nf;
namespace pg=ph::pg;
int checks=0;
void check(bool v) { ++checks; if(!v) {std::cerr<<"failed check "<<checks<<'\n';std::abort();} }
template<class T> void suite(double tolerance) {
    ex::persistent_axis_identity axis{{1,ex::serialized_record_kind::persistent_axis_identity,sizeof(ex::persistent_axis_identity)},
                                      {1,1},{2,1},{3,1},{4,1}};
    nf::prepared_identity id{{90,1},{20,1},{1},{21,1}};
    nf::instance_binding live{id,{{30,1},{1}},{{31,1},{1}},{}};
    ph::scaled_tanh<T> prepared(id,axis,5);
    check(prepared.preparation_status()==nf::status::success);
    check(prepared.required_elements()==5);
    check(pg::validate_prepared_program_v2(prepared.native_program())==pg::program_status::success);
    auto stages=prepared.native_program().stages;
    check(stages[1].dependency_count==1 && prepared.native_program().dependencies[0]==0);
    std::vector<T> x{1,2,3,4,5},p{.1,.2,.3,.4,.5},z(5),y(5);
    ph::sweep_binding<T> binding{x,p,z,y,axis,&live};
    ph::saved_primal saved{};
    for(int round=0;round<4;++round) {
        x[0]+=T(.1);p[2]+=T(.1);++live.state.generation.value;++live.parameters.generation.value;
        check(prepared.forward(binding,&saved)==pg::program_status::success);
        for(std::size_t i=0;i<5;++i) check(std::abs(double(y[i])-std::tanh(double(T(x[i]*p[i]))))<tolerance);
        check(prepared.primal_is_current(saved,binding));
        check(prepared.native_program().stages==stages);
    }
    auto previous=saved;
    check(prepared.forward(binding,&saved)==pg::program_status::success);
    check(!prepared.primal_is_current(previous,binding));
    auto alternate=binding;alternate.state=p;
    check(!prepared.primal_is_current(saved,alternate));
    ++live.parameters.generation.value;
    check(!prepared.primal_is_current(saved,binding));
    --live.parameters.generation.value;
    ++live.state.generation.value;
    check(!prepared.primal_is_current(saved,binding));
    --live.state.generation.value;
    auto oldz=z,oldy=y;
    const auto attempts=prepared.forward_attempts();
    auto rejected=[&](ph::sweep_binding<T> b,void* stream=nullptr) {
        check(prepared.forward(b,nullptr,stream)==pg::program_status::invalid_argument);
        check(z==oldz && y==oldy);check(prepared.forward_attempts()==attempts);
    };
    auto bad=binding;bad.output={y.data(),4};rejected(bad); // Late output is checked before multiply writes.
    bad=binding;bad.intermediate={z.data(),4};rejected(bad);
    bad=binding;bad.parameters={p.data(),4};rejected(bad);
    bad=binding;bad.output=x;rejected(bad);
    bad=binding;bad.intermediate=p;rejected(bad);
    bad=binding;bad.output=z;rejected(bad);
    bad=binding;bad.axis.order.low++;rejected(bad);
    bad=binding;bad.axis.header.schema_version=0;rejected(bad);
    bad=binding;bad.instance=nullptr;rejected(bad);
    rejected(binding,reinterpret_cast<void*>(1));
    ++live.prepared.epoch.value;rejected(binding);--live.prepared.epoch.value;
    ++live.prepared.structure.low;rejected(binding);--live.prepared.structure.low;
    ++live.prepared.projection.low;rejected(binding);--live.prepared.projection.low;
    auto generation=live.parameters.generation.value;live.parameters.generation.value=0;rejected(binding);live.parameters.generation.value=generation;
    std::fesetround(FE_DOWNWARD);rejected(binding);std::fesetround(FE_TONEAREST);
    auto repeated=binding;repeated.parameters=x;
    check(prepared.forward(repeated)==pg::program_status::success);
    for(std::size_t i=0;i<5;++i) check(std::abs(double(y[i])-std::tanh(double(T(x[i]*x[i]))))<tolerance);
    // Existing direct host numerical route remains usable with the same physical order.
    check(ph::nn::local_forward(ph::nn::local_operation::multiply,std::span<const T>(x),std::span<const T>(p),std::span<T>(z))==ph::nn::local_status::success);
    ph::scaled_tanh<T> empty(id,axis,0);
    check(empty.forward({{},{},{},{},axis,&live})==pg::program_status::success);
}
int main() {suite<float>(2e-7);suite<double>(1e-15);std::cout<<checks<<" prepared host checks passed\n";}
