#include "minimal_direct_layout.hh"

#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace layout = cellerator::layout;
namespace controls = cellerator::bench::factorized_interactions::controls;

int main() {
    std::vector<layout::FactorizedInteractionMechanism> mechanisms(3);
    mechanisms[0].logical_id = 20;
    mechanisms[0].arity = 3;
    mechanisms[0].inputs = {{1, 1, 3, 0}};
    mechanisms[0].target = 0;
    mechanisms[0].alpha = 1;
    mechanisms[0].bias = 0;
    mechanisms[0].terms = {{2, 1}, {2, 1}};

    mechanisms[1].logical_id = 30;
    mechanisms[1].arity = 2;
    mechanisms[1].inputs = {{1, 2, 0, 0}};
    mechanisms[1].target = 0;
    mechanisms[1].alpha = 1;
    mechanisms[1].bias = 0;
    mechanisms[1].terms = {{2, 1}, {0, 2}};

    mechanisms[2].logical_id = 10;
    mechanisms[2].arity = 4;
    mechanisms[2].inputs = {{0, 3, 3, 3}};
    mechanisms[2].target = 1;
    mechanisms[2].alpha = 0;
    mechanisms[2].bias = 0;

    layout::FactorizedInteractionOptions options;
    options.pattern_id = 11;
    options.structure_id = 13;
    options.structure_epoch = 17;
    // Packet limits are intentionally irrelevant to this canonical-only owner.
    options.max_mechanisms = 0;
    options.max_inputs = 0;
    options.max_contexts = UINT32_MAX;

    controls::MinimalDirectHostLayout prepared(
        mechanisms, {4, 3, 2, 2, 1, 3}, options);
    const auto view = prepared.view();
    assert(view.schema == layout::factorized_interaction_schema);
    assert(view.pattern_id == 11 && view.structure_id == 13 &&
           view.structure_epoch == 17);
    assert(view.extents.inputs == 4 && view.extents.contexts == 3 &&
           view.extents.outputs == 2 && view.extents.alpha_rows == 2 &&
           view.extents.biases == 1 && view.extents.gammas == 3);
    assert(view.mechanism_count == 3 && view.term_count == 4);
    assert(view.mechanisms == prepared.mechanisms().data());
    assert(view.terms == prepared.terms().data());
    assert(view.packet_count == 0 && view.packet_mechanisms == nullptr &&
           view.packets == nullptr && view.role_refs == nullptr);

    const auto& records = prepared.mechanisms();
    assert(records[0].logical_id == 20 && records[0].arity == 3);
    assert(records[0].inputs[0] == 1 && records[0].inputs[1] == 1 &&
           records[0].inputs[2] == 3 && records[0].inputs[3] == 0);
    assert(records[0].term_begin == 0 && records[0].term_count == 2);
    assert(records[1].term_begin == 2 && records[1].term_count == 2);
    assert(records[2].term_begin == 4 && records[2].term_count == 0);
    const auto& terms = prepared.terms();
    assert(terms[0].context == 2 && terms[0].gamma == 1 && terms[0].mechanism == 0);
    assert(terms[1].context == 2 && terms[1].gamma == 1 && terms[1].mechanism == 0);
    assert(terms[2].context == 2 && terms[2].gamma == 1 && terms[2].mechanism == 1);
    assert(terms[3].context == 0 && terms[3].gamma == 2 && terms[3].mechanism == 1);

    assert(view.max_output_mechanisms == 2);
    assert(view.max_input_roles == 4);
    assert(view.max_context_terms == 3);
    assert(view.max_alpha_mechanisms == 2);
    assert(view.max_bias_mechanisms == 3);
    assert(view.max_gamma_terms == 3);
    assert(view.max_mechanism_terms == 2);

    controls::MinimalDirectHostLayout empty({}, {0, 0, 0, 0, 0, 0}, options);
    const auto empty_view = empty.view();
    assert(empty_view.mechanism_count == 0 && empty_view.term_count == 0);
    assert(empty_view.mechanisms == nullptr && empty_view.terms == nullptr);
    assert(empty_view.max_output_mechanisms == 0 &&
           empty_view.max_input_roles == 0 && empty_view.max_context_terms == 0 &&
           empty_view.max_alpha_mechanisms == 0 &&
           empty_view.max_bias_mechanisms == 0 &&
           empty_view.max_gamma_terms == 0 && empty_view.max_mechanism_terms == 0);

    auto duplicate_ids = mechanisms;
    duplicate_ids[2].logical_id = duplicate_ids[1].logical_id;
    bool rejected = false;
    try {
        controls::MinimalDirectHostLayout invalid(
            duplicate_ids, {4, 3, 2, 2, 1, 3}, options);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    assert(rejected);

    auto invalid_role = mechanisms;
    invalid_role[0].inputs[1] = 4;
    rejected = false;
    try {
        controls::MinimalDirectHostLayout invalid(
            invalid_role, {4, 3, 2, 2, 1, 3}, options);
    } catch (const std::out_of_range&) {
        rejected = true;
    }
    assert(rejected);
    return 0;
}
