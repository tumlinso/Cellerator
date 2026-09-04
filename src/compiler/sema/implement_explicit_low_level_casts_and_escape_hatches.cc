#include <Cellerator/compiler/sema/implement_explicit_low_level_casts_and_escape_hatches_v1.hh>

namespace cellerator::compiler::sema::v1 {

semantic_cast_result cast_to_semantic_state(
    const ordinary_cxx_view &source,
    state_type destination,
    semantic_cast_mode mode,
    const char *effect_contract) noexcept {
    semantic_cast_result result{{source.data, destination}, semantic_cast_status::ok,
                                mode == semantic_cast_mode::unsafe, effect_contract};
    if (source.data == nullptr) {
        result.status = semantic_cast_status::null_view;
        return result;
    }
    if (source.rank == 0 || source.rank > execution::biological_operand_max_axes
        || destination.rank != source.rank || destination.axes == nullptr) {
        result.status = semantic_cast_status::unrepresentable_rank;
        return result;
    }
    if (source.element_type == execution::numeric_type::invalid
        || destination.element_type == execution::numeric_type::invalid) {
        result.status = semantic_cast_status::unrepresentable_type;
        return result;
    }
    if (mode == semantic_cast_mode::checked
        && (source.element_type != destination.element_type
            || source.residency != destination.residency_intent)) {
        result.status = semantic_cast_status::contract_mismatch;
        return result;
    }
    for (std::uint8_t i = 0; i < source.rank; ++i) {
        if (source.shape[i] != destination.axes[i].local_extent
            && mode == semantic_cast_mode::checked) {
            result.status = semantic_cast_status::contract_mismatch;
            return result;
        }
    }
    return result;
}

}  // namespace cellerator::compiler::sema::v1
