#include <Cellerator/compiler/sema/implement_state_semantics_v1.hh>

namespace cellerator::compiler::sema::v1 {

state_validation validate_state_view(const state_view &view) noexcept {
    if (view.data == nullptr)
        return state_validation::missing_data;
    if (view.type.rank == 0 || view.type.rank > execution::biological_operand_max_axes
        || view.type.axes == nullptr)
        return state_validation::invalid_rank;
    if (view.type.element_type == execution::numeric_type::invalid)
        return state_validation::invalid_type;
    if (view.type.feature_width == 0)
        return state_validation::invalid_feature_width;
    for (std::uint8_t i = 0; i < view.type.rank; ++i) {
        if (view.type.axes[i].global_extent == 0)
            return state_validation::shape_mismatch;
    }
    return state_validation::ok;
}

state_validation validate_against_dense_operand(
    const state_view &state,
    const execution::dense_tensor_view &operand) noexcept {
    const auto state_status = validate_state_view(state);
    if (state_status != state_validation::ok)
        return state_status;
    if (state.data != operand.data)
        return state_validation::missing_data;
    if (state.type.rank != operand.rank)
        return state_validation::invalid_rank;
    if (state.type.element_type != operand.value_type)
        return state_validation::invalid_type;
    if (state.type.residency_intent != operand.location.residency)
        return state_validation::residency_mismatch;
    for (std::uint8_t i = 0; i < state.type.rank; ++i) {
        if (state.type.axes[i].local_extent != operand.shape[i])
            return state_validation::shape_mismatch;
    }
    return state_validation::ok;
}

state_view bind_pointer(void *pointer, state_type type) noexcept {
    return {pointer, type};
}

}  // namespace cellerator::compiler::sema::v1
