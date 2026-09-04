#include <Cellerator/compiler/sema/implement_relation_endpoint_semantics_v1.hh>

#include <cassert>

int main() {
    using namespace cellerator::compiler::sema::v1;
    const axis_type source{{{1, 2}, 0, false}, 8, {3, 4}, {5, 6}, {7, 8}, 8, {9, 10}};
    const axis_type destination{{{11, 12}, 0, false}, 4, {13, 14}, {15, 16}, {17, 18}, 4, {19, 20}};
    relation_endpoint_semantics relation{source, destination, {21, 22}, {3},
        {23, 24}, {25, 26}, {27, 28}, relation_orientation::forward,
        {29, 30}, relation_mutation_policy::immutable_structure, 17};
    auto runtime = to_runtime_relation(relation);
    assert(agrees_with_runtime_relation(relation, runtime));
    runtime.logical_edge_count = 18;
    assert(!agrees_with_runtime_relation(relation, runtime));
}
