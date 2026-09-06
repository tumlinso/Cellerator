#include <Cellerator/compute/operation/relation_semantics.hh>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <type_traits>
#include <utility>

using namespace cellerator;
using namespace compute::relation;
namespace {
void check(bool value, const char* message) {
    if (!value) { std::cerr << message << '\n'; std::exit(1); }
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
operation_descriptor fixture() {
    operation_descriptor op;
    op.topology = {{7, 8}, {1}, axis(10, 4), axis(20, 5), {30, 31}, 9};
    return op;
}
template<class F> void different(F change) {
    auto x = fixture(); auto y = fixture(); change(y);
    check(!equivalent(x, y), "changed semantic field compared equal");
}
}
int main() {
    static_assert(std::is_trivially_copyable<operation_descriptor>::value);
    static_assert(std::is_standard_layout<operation_descriptor>::value);
    auto x = fixture(); auto y = fixture();
    check(bool(validate(x)), "valid fixture rejected");
    check(equivalent(x, y), "independent construction unequal");
    auto moved = std::move(y); check(equivalent(x, moved), "move lost identity");
    different([](auto& o){o.topology.identity.high ^= 1ull << 63;});
    different([](auto& o){++o.topology.epoch.value;});
    different([](auto& o){++o.topology.logical_edge_order.high;});
    different([](auto& o){++o.topology.edge_count;});
    for (bool source : {false, true}) {
        for (int field = 0; field < 9; ++field) different([=](auto& o){
            auto& a = source ? o.topology.source : o.topology.destination;
            switch(field) {
            case 0: ++a.identity.domain.high; break;
            case 1: ++a.identity.order.high; break;
            case 2: ++a.identity.geometry.high; break;
            case 3: ++a.identity.partition.high; break;
            case 4: ++a.extent; break;
            case 5: ++a.identity.header.schema_version; break;
            case 6: ++a.identity.header.byte_count; break;
            case 7: a.identity.header.kind = execution::serialized_record_kind(0); break;
            case 8: ++a.identity.domain.low; break;
            }
        });
    }
    different([](auto& o){o.direction = orientation::transpose;});
    different([](auto& o){++o.dense_width;});
    different([](auto& o){o.update = output_update::accumulate;});
    different([](auto& o){o.input_output_aliasing_legal = true;});
    different([](auto& o){o.arithmetic.permit_fma = false;});
    different([](auto& o){o.arithmetic.permit_reassociation = false;});
    different([](auto& o){o.arithmetic.nonfinite = nonfinite_policy::reject;});
    for (int field = 0; field < 5; ++field) different([=](auto& o){
        auto& a = o.arithmetic;
        execution::numeric_type* p[] = {&a.relation_storage, &a.input_storage, &a.multiply,
            &a.accumulation, &a.output_storage};
        *p[field] = execution::numeric_type::f64;
    });
    y = x; y.direction = orientation::transpose;
    check(input_axis(x).extent == 4 && result_axis(x).extent == 5
        && input_axis(y).extent == 5 && result_axis(y).extent == 4, "direction reversed");
    y = x; y.topology.source.identity.domain = {};
    check(validate(y).code == status_code::invalid_axis, "missing axis accepted");
    y = x; y.topology.epoch = {};
    check(validate(y).code == status_code::invalid_identity, "zero epoch accepted");
    y = x; y.topology.source.extent = 0;
    check(!validate(y), "edges with absent sources accepted");
    y.topology.edge_count = 0; check(bool(validate(y)), "empty math rejected");
    y.topology.destination.extent = 0; check(bool(validate(y)), "empty axes rejected");
    y = x; y.topology.source.extent = (1ull << 32);
    check(bool(validate(y)), "logical extent narrowed to local32");
    y.topology.source.extent = std::numeric_limits<std::uint64_t>::max();
    check(validate(y).code == status_code::invalid_shape, "byte overflow accepted");
    y = x; y.topology.edge_count = std::numeric_limits<std::uint64_t>::max();
    check(!validate(y), "edge storage overflow accepted");
    y = x; y.dense_width = 17; y.arithmetic.accumulation = execution::numeric_type::f64;
    y.update = output_update::accumulate; y.arithmetic.nonfinite = nonfinite_policy::reject;
    check(bool(validate(y)), "valid capability-limited policy rejected");
    y.update = output_update::affine_accumulate;
    check(validate(y).code == status_code::unsupported_semantics, "undefined affine accepted");
    y = x; y.direction = orientation(255); check(!validate(y), "invalid direction accepted");
    y = x; y.dense_width = 0; check(!validate(y), "zero width accepted");
    y = x; y.arithmetic.nonfinite = nonfinite_policy(255);
    check(!validate(y), "unknown policy accepted");
    y = x; y.arithmetic.multiply = execution::numeric_type::invalid;
    check(!validate(y), "invalid arithmetic accepted");
    // No generation/pointer field exists in the mathematical descriptor. Actual
    // alias and publication checks belong to native bindings, tested by that lane.
    std::cout << "semantic descriptor tests passed\n";
}
