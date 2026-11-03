#include <Cellerator/layout/factorized_interaction_pattern.hh>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace dep = cellerator::geometry::dependency;
namespace layout = cellerator::layout;

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

layout::FactorizedInteractionMechanism mechanism(
    std::uint64_t id, std::uint32_t arity,
    std::initializer_list<std::uint64_t> inputs, std::uint64_t target) {
    layout::FactorizedInteractionMechanism result;
    result.logical_id = id;
    result.arity = arity;
    result.target = target;
    result.alpha = 0;
    result.bias = 0;
    std::copy(inputs.begin(), inputs.end(), result.inputs.begin());
    return result;
}

dep::ExplicitPattern source_pattern() {
    return dep::ExplicitPattern({
        {201, {{2, 0, 17}, {7, 0, 99}, {2, 0, 17}}},
        {202, {{7, 0, 55}, {2, 0, 56}}}
    }, 110, 220, 330);
}

std::vector<layout::FactorizedInteractionMechanism> valid_records() {
    // The supplied order intentionally differs from ExplicitPattern's stable
    // mechanism order. The checked adapter joins only by explicit logical ID.
    return {
        mechanism(202, 2, {7, 8}, 0),
        mechanism(201, 3, {4, 5, 4}, 1)
    };
}

layout::FactorizedInteractionExtents extents() {
    return {9, 1, 2, 1, 1, 1};
}

void checked_binding_preserves_explicit_identity_and_occurrences() {
    const auto pattern = source_pattern();
    const std::vector<std::uint64_t> incidence_to_input{4, 5, 4, 7, 8};
    auto records = valid_records();
    layout::FactorizedInteractionOptions options;
    options.pattern_id = 1;
    options.structure_id = 2;
    options.structure_epoch = 3;

    const auto prepared = layout::prepare_factorized_interactions_bound(
        pattern, incidence_to_input, records, extents(), options);
    require(prepared.options.pattern_id == pattern.pattern_id()
                && prepared.options.structure_id == pattern.structure_id()
                && prepared.options.structure_epoch == pattern.structure_epoch(),
            "bound preparation derives its structural guards from ExplicitPattern");
    require(prepared.mechanisms.size() == 2
                && prepared.mechanisms[0].logical_id == 202
                && prepared.mechanisms[1].logical_id == 201,
            "packet preparation preserves caller record order after ID-based binding");

    const auto& repeated = prepared.mechanisms[1];
    require(repeated.arity == 3 && repeated.inputs[0] == 4
                && repeated.inputs[1] == 5 && repeated.inputs[2] == 4,
            "incidence binding preserves the ordered repeated role occurrences");
    const auto repeated_packet = prepared.mechanism_packets[1];
    const auto& packet = prepared.packets[repeated_packet];
    const auto packed_it = std::find(
        prepared.packet_mechanisms.begin() + static_cast<std::ptrdiff_t>(packet.mechanism_begin),
        prepared.packet_mechanisms.begin() + static_cast<std::ptrdiff_t>(packet.mechanism_begin
                                                                         + packet.mechanism_count),
        1);
    require(packed_it != prepared.packet_mechanisms.begin()
                            + static_cast<std::ptrdiff_t>(packet.mechanism_begin
                                                         + packet.mechanism_count),
            "packet contains the requested canonical mechanism");
    const auto packed = static_cast<std::uint64_t>(packed_it - prepared.packet_mechanisms.begin());
    for (std::uint32_t role = 0; role < repeated.arity; ++role) {
        const auto ref_index = 4 * packet.mechanism_begin
            + static_cast<std::uint64_t>(role) * packet.mechanism_count
            + (packed - packet.mechanism_begin);
        const auto ref = prepared.role_refs[ref_index];
        require(prepared.packet_inputs[packet.input_begin + ref] == repeated.inputs[role],
                "physical refs reconstruct each explicitly bound incidence");
    }
    layout::validate_factorized_interactions(prepared);
}

template<class Function>
void expect_invalid(Function&& function, const char* message) {
    bool rejected = false;
    try { function(); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, message);
}

void invalid_identity_and_role_bindings_reject() {
    const auto pattern = source_pattern();
    const std::vector<std::uint64_t> incidence_to_input{4, 5, 4, 7, 8};
    const auto base = valid_records();

    auto short_binding = incidence_to_input;
    short_binding.pop_back();
    expect_invalid([&] {
        (void)layout::prepare_factorized_interactions_bound(
            pattern, short_binding, base, extents());
    }, "incidence binding with a wrong extent is rejected");

    auto wrong_id_set = base;
    wrong_id_set[0].logical_id = 999;
    expect_invalid([&] {
        (void)layout::prepare_factorized_interactions_bound(
            pattern, incidence_to_input, wrong_id_set, extents());
    }, "mechanisms with a different stable ID set are rejected");

    auto wrong_arity = base;
    wrong_arity[1].arity = 2;
    expect_invalid([&] {
        (void)layout::prepare_factorized_interactions_bound(
            pattern, incidence_to_input, wrong_arity, extents());
    }, "mechanism arity different from source incidence count is rejected");

    auto wrong_role_order = base;
    wrong_role_order[1].inputs[1] = 4;
    wrong_role_order[1].inputs[2] = 5;
    expect_invalid([&] {
        (void)layout::prepare_factorized_interactions_bound(
            pattern, incidence_to_input, wrong_role_order, extents());
    }, "wrong ordered incidence-to-occurrence binding is rejected");

    auto duplicate_id = base;
    duplicate_id[0].logical_id = duplicate_id[1].logical_id;
    expect_invalid([&] {
        (void)layout::prepare_factorized_interactions_bound(
            pattern, incidence_to_input, duplicate_id, extents());
    }, "duplicate mechanism identities are rejected");
}

} // namespace

int main() {
    try {
        checked_binding_preserves_explicit_identity_and_occurrences();
        invalid_identity_and_role_bindings_reject();
        std::cout << "factorized ExplicitPattern binding checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "factorized ExplicitPattern binding check failed: "
                  << error.what() << '\n';
        return 1;
    }
}
