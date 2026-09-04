#include <Cellerator/compiler/ir/semantic/implement_semantic_canonicalization_and_equivalence_v1.hh>

#include <cassert>

using namespace Cellerator::compiler::ir::semantic;

namespace {

semantic_canonical_record_v1 baseline_record() {
    semantic_canonical_record_v1 record;
    record.operation_identity = {101, 102};
    record.operation_spelling = "ce::relation_apply<gene, cell>";
    record.input_types = {{201, 202}, {203, 204}};
    record.output_types = {{205, 206}};
    record.biological_identities = {{301, 302}, {303, 304}};
    record.numerical = {
        cellerator::execution::numeric_type::f16,
        cellerator::execution::numeric_type::f32,
        cellerator::execution::numeric_type::f32,
        cellerator::execution::numeric_type::f32};
    record.effects = field_effect_reads_ir_v1 | field_effect_writes_ir_v1;
    record.field_identity = 401;
    record.field_boundary = execution_field_boundary_ir_v1::transparent;
    return record;
}

bool equal_fingerprint(semantic_fingerprint_v1 left,
                       semantic_fingerprint_v1 right) {
    return left.low == right.low && left.high == right.high;
}

}  // namespace

int main() {
    const auto baseline = baseline_record();
    auto alternate_spelling = baseline;
    alternate_spelling.operation_spelling =
        "  ce :: relation_apply < gene ,\n cell >  ";

    const auto first = fingerprint_semantic_record_v1(baseline);
    const auto second = fingerprint_semantic_record_v1(alternate_spelling);
    assert(first && second && first->valid());
    assert(equal_fingerprint(*first, *second));
    assert(semantic_equivalent_v1(baseline, alternate_spelling));
    assert(equal_fingerprint(*first, *fingerprint_semantic_record_v1(baseline)));

    auto changed_type = baseline;
    changed_type.input_types.front().low += 1;
    assert(!semantic_equivalent_v1(baseline, changed_type));
    assert(!equal_fingerprint(*first, *fingerprint_semantic_record_v1(changed_type)));

    auto changed_identity = baseline;
    changed_identity.biological_identities.back().high += 1;
    assert(!semantic_equivalent_v1(baseline, changed_identity));

    auto changed_numeric = baseline;
    changed_numeric.numerical.accumulation = cellerator::execution::numeric_type::f64;
    assert(!semantic_equivalent_v1(baseline, changed_numeric));

    auto changed_effect = baseline;
    changed_effect.effects |= field_effect_synchronizes_ir_v1;
    assert(!semantic_equivalent_v1(baseline, changed_effect));

    auto changed_field = baseline;
    changed_field.field_boundary = execution_field_boundary_ir_v1::explicit_boundary;
    assert(!semantic_equivalent_v1(baseline, changed_field));

    semantic_canonicalization_status_v1 status{};
    auto invalid = baseline;
    invalid.input_types.front() = {};
    assert(!fingerprint_semantic_record_v1(invalid, &status));
    assert(status == semantic_canonicalization_status_v1::invalid_type_identity);
}
