#include <Cellerator/compiler/ir/semantic/deliver_source_to_semantic_ir_vertical_slice_v1.hh>

#include <cassert>
#include <cmath>

using namespace Cellerator::compiler::ir::semantic;

int main() {
    const std::string relation_source = R"cell(#pragma cellerator
field void propagate() <[
    given ce::profile(activated_fibroblast);
::
    response = expression -[regulation]-> target_genes;
]>
)cell";
    const auto relation = lower_cell_source_to_semantic_ir_v1(relation_source);
    assert(relation && relation->field == "propagate");
    assert(relation->profile == "activated_fibroblast");
    assert(relation->operations.size() == 1);
    assert(relation->operations[0].source.line == 5);
    assert(relation->operations[0].source.column == 5);
    assert(relation->operations[0].destination_domain == "target_genes");
    assert(operation_core_compatible_v1(*relation));

    const auto relation_text = write_semantic_ir_v1(*relation);
    const auto relation_round_trip = read_semantic_ir_v1(*relation_text);
    assert(relation_round_trip);
    assert(write_semantic_ir_v1(*relation_round_trip) == relation_text);
    const auto relation_values = execute_semantic_referee_v1(
        *relation_round_trip, {{"expression", 3.0}, {"regulation", 4.0}});
    assert(relation_values && relation_values->at("response") == 12.0);

    const std::string mixed_source = R"cell(#pragma cellerator
field void analyze() <[
    given ce::profile(pbmc3k);
::
    edge_scores = ce::contract_on(support, expression, context);
    response = ce::segment_normalize(edge_scores, segment_scale);
]>
)cell";
    const auto mixed = lower_cell_source_to_semantic_ir_v1(mixed_source);
    assert(mixed && mixed->operations.size() == 2);
    assert(mixed->operations[0].kind ==
           cellerator::compute::operation::v2::operation_kind::contract_on_support);
    assert(mixed->operations[1].kind ==
           cellerator::compute::operation::v2::operation_kind::segment_normalize);
    const auto values = execute_semantic_referee_v1(
        *mixed, {{"support", 1.0}, {"expression", 6.0}, {"context", 2.0},
                 {"segment_scale", 3.0}});
    assert(values && std::abs(values->at("response") - 4.0) < 1e-12);

    const auto text = write_semantic_ir_v1(*mixed);
    const auto round_trip = read_semantic_ir_v1(*text);
    assert(round_trip && write_semantic_ir_v1(*round_trip) == text);
    auto writable_text = *text;
    writable_text.replace(writable_text.find("profile\tpbmc3k"),
                          std::string("profile\tpbmc3k").size(),
                          "profile\tstimulated_pbmc3k");
    const auto edited = read_semantic_ir_v1(writable_text);
    assert(edited && edited->profile == "stimulated_pbmc3k");
    const auto receipt = make_source_linked_receipt_v1(mixed_source, *round_trip);
    assert(receipt && receipt->source_hash != 0 && receipt->semantic_hash != 0);
    assert(receipt->operation_count == 2);
    assert(receipt->exact_source_mapping);
    assert(receipt->operation_core_compatible);

    semantic_vertical_slice_status_v1 status{};
    assert(!lower_cell_source_to_semantic_ir_v1(
        "field void bad() <[\n::\n x = y -[r]-> z;\n]>", &status));
    assert(status == semantic_vertical_slice_status_v1::missing_profile);
}
