#include <Cellerator/compiler/sema/implement_numerical_tuple_semantics_v1.hh>

namespace cellerator::compiler::sema::v1 {
namespace {
bool numeric(execution::numeric_type type) noexcept {
    return type != execution::numeric_type::invalid;
}
}  // namespace

bool valid_numerical_tuple(const numerical_tuple &tuple) noexcept {
    return numeric(tuple.relation_storage) && numeric(tuple.dense_input)
        && numeric(tuple.compute) && numeric(tuple.accumulation)
        && numeric(tuple.output);
}

bool numerical_candidate_legal(const numerical_tuple &tuple,
                               const numerical_candidate_capability &candidate) noexcept {
    if (!valid_numerical_tuple(tuple))
        return false;
    if (candidate.storage != tuple.relation_storage
        || candidate.compute != tuple.compute
        || candidate.accumulation != tuple.accumulation)
        return false;
    if (tuple.nonfinite == nonfinite_contract::propagate
        && !candidate.preserves_nonfinite)
        return false;
    if (tuple.approximation == approximation_contract::forbidden
        && candidate.approximate)
        return false;
    return true;
}

}  // namespace cellerator::compiler::sema::v1
