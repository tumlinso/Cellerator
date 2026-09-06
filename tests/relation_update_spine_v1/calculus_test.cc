#include <Cellerator/compute/operation/relation_calculus.hh>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <utility>
using namespace cellerator;
using namespace compute::relation;
namespace {
unsigned checks = 0;
void check(bool condition, const char* message) {
    ++checks;
    if (!condition) { std::cerr << message << '\n'; std::exit(1); }
}
axis_descriptor axis(std::uint64_t id, std::uint64_t extent) {
    axis_descriptor a{};
    a.identity.header = {execution::biological_abi_version,
        execution::serialized_record_kind::persistent_axis_identity,
        sizeof(execution::persistent_axis_identity)};
    a.identity.domain = {id, 2}; a.identity.order = {id, 3};
    a.identity.geometry = {id, 4}; a.identity.partition = {id, 5};
    a.extent = extent;
    return a;
}
relation_calculus_descriptor fixture() {
    relation_calculus_descriptor c{};
    c.forward.topology = {{7,8}, {1}, axis(10,4), axis(20,5), {30,31}, 9};
    c.forward.dense_width = 16;
    c.transpose = c.forward;
    c.transpose.direction = orientation::transpose;
    return c;
}
template<class F> void reject(F mutate) {
    auto c = fixture(); mutate(c);
    check(!validate(c), "malformed calculus accepted");
    check(!equivalent(fixture(), c), "changed semantic field compared equal");
}
relation_effect_sequence cycle() {
    relation_effect_sequence s{};
    s.initial_generation = {1}; s.count = 6;
    s.stages[0] = {11, relation_effect_kind::forward, 0, {1}, {0}};
    s.stages[1] = {12, relation_effect_kind::transpose, 1, {1}, {0}};
    s.stages[2] = {13, relation_effect_kind::edge_gradient, 1, {1}, {0}};
    s.stages[3] = {14, relation_effect_kind::value_update, 6, {1}, {2}};
    s.stages[4] = {15, relation_effect_kind::publication, 8, {2}, {0}};
    s.stages[5] = {16, relation_effect_kind::forward, 16, {2}, {0}};
    return s;
}
template<class F> void bad_cycle(F mutate) {
    auto s = cycle(); mutate(s); check(!validate(s), "invalid effect order accepted");
}
}
int main() {
    auto c = fixture();
    check(bool(validate(c)), "N16 rejected");
    auto independently_constructed = fixture();
    check(equivalent(c, independently_constructed), "value constructions unequal");
    auto moved = std::move(independently_constructed);
    check(equivalent(c, moved), "move broke value identity");
    independently_constructed.forward.topology.source.extent = 999;
    check(equivalent(c, moved), "copy retained self-referential storage");
    c.gradient = gradient_arithmetic::round_operands_f16_rne;
    check(bool(validate(c)), "explicit half profile rejected");
    check(!equivalent(c, fixture()), "mixed profile compared equal");
    c = fixture(); c.forward.dense_width = c.transpose.dense_width = 1;
    check(bool(validate(c)), "retained N1 rejected");
    c = fixture(); c.transpose.update = output_update::accumulate;
    check(bool(validate(c)), "independent output accumulation rejected");
    c = fixture(); c.update = value_update_kind::delta_add;
    check(bool(validate(c)), "delta update rejected");
    check(!equivalent(c, fixture()), "update kinds compared equal");
    reject([](auto& x){x.forward.direction=orientation(255);});
    reject([](auto& x){std::swap(x.forward,x.transpose);});
    reject([](auto& x){x.gradient=gradient_arithmetic(255);});
    reject([](auto& x){x.update=value_update_kind(255);});
    reject([](auto& x){x.forward.dense_width=x.transpose.dense_width=17;});
    reject([](auto& x){x.forward.input_output_aliasing_legal=true;});
    reject([](auto& x){x.scalar_gradient.channels_per_edge=16;});
    reject([](auto& x){x.scalar_gradient.output=output_update::accumulate;});
    reject([](auto& x){x.scalar_gradient.input_storage=execution::numeric_type::f16;});
    reject([](auto& x){x.scalar_gradient.accumulation=execution::numeric_type::f64;});
    reject([](auto& x){x.scalar_gradient.output_storage=execution::numeric_type::f16;});
    reject([](auto& x){x.scalar_gradient.permit_fma=false;});
    reject([](auto& x){x.scalar_gradient.permit_reassociation=false;});
    reject([](auto& x){x.scalar_gradient.nonfinite=nonfinite_policy::reject;});
    reject([](auto& x){x.forward.arithmetic.permit_fma=x.transpose.arithmetic.permit_fma=false;});
    reject([](auto& x){x.forward.topology.source.extent=std::numeric_limits<std::uint64_t>::max();});
    for (bool source : {false,true}) for (int field=0;field<8;++field) {
        reject([=](auto& x){
            auto& a=source?x.transpose.topology.source:x.transpose.topology.destination;
            switch(field) {
            case 0: ++a.identity.domain.high; break;
            case 1: ++a.identity.order.high; break;
            case 2: ++a.identity.geometry.high; break;
            case 3: ++a.identity.partition.high; break;
            case 4: ++a.extent; break;
            case 5: ++a.identity.header.schema_version; break;
            case 6: ++a.identity.header.byte_count; break;
            case 7: a.identity.header.kind=execution::serialized_record_kind(255); break;
            }
        });
    }
    reject([](auto& x){++x.transpose.topology.identity.high;});
    reject([](auto& x){++x.transpose.topology.epoch.value;});
    reject([](auto& x){++x.transpose.topology.logical_edge_order.low;});
    reject([](auto& x){++x.transpose.topology.edge_count;});
    check(bool(validate(cycle())), "valid dependency diamond rejected");
    auto s=cycle(); auto copied=s; s.stages[0].identity=0;
    check(bool(validate(std::move(copied))), "effect copy/move retained pointers");
    bad_cycle([](auto& x){x.count=0;});
    bad_cycle([](auto& x){x.count=max_relation_effects+1;});
    bad_cycle([](auto& x){x.initial_generation={0};});
    bad_cycle([](auto& x){x.stages[1].identity=x.stages[0].identity;});
    bad_cycle([](auto& x){x.stages[1].identity=0;});
    bad_cycle([](auto& x){x.stages[1].kind=relation_effect_kind(255);});
    bad_cycle([](auto& x){x.stages[0].dependencies=1;});
    bad_cycle([](auto& x){x.stages[0].dependencies=1u<<31;});
    bad_cycle([](auto& x){x.stages[1].dependencies=4;});
    bad_cycle([](auto& x){x.stages[3].dependencies=4;}); // forgot transpose reader
    bad_cycle([](auto& x){x.stages[4].dependencies=4;}); // publication lacks writer
    bad_cycle([](auto& x){x.stages[5].dependencies=8;}); // read lacks publication
    bad_cycle([](auto& x){x.stages[5].reads={1};}); // historical plane no longer exists
    bad_cycle([](auto& x){x.stages[4].reads={1};});
    bad_cycle([](auto& x){x.stages[4].writes={2};});
    bad_cycle([](auto& x){x.stages[3].writes={1};});
    bad_cycle([](auto& x){x.stages[3].writes={0};});
    bad_cycle([](auto& x){x.stages[2].writes={2};});
    bad_cycle([](auto& x){x.count=4;}); // unpublished write
    bad_cycle([](auto& x){std::swap(x.stages[2],x.stages[3]);});
    s=cycle(); s.count=8;
    s.stages[6]={17,relation_effect_kind::value_update,32,{2},{5}};
    s.stages[7]={18,relation_effect_kind::publication,64,{5},{0}};
    check(bool(validate(s)), "second cycle / nonconsecutive generation rejected");
    s.count=9;
    s.stages[8]={19,relation_effect_kind::transpose,128,{5},{0}};
    check(bool(validate(s)), "read after second publication rejected");
    s.stages[8].reads={2};
    check(!validate(s), "old generation after second publication accepted");
    s.initial_generation={std::numeric_limits<std::uint64_t>::max()};
    s.count=2;
    s.stages[0]={1,relation_effect_kind::value_update,0,s.initial_generation,{0}};
    s.stages[1]={2,relation_effect_kind::publication,1,{0},{0}};
    check(!validate(s), "generation overflow accepted");
    std::cout << "ru1_calculus: " << checks << " host checks passed (full_f32, round_operands_f16_rne)\n";
}
