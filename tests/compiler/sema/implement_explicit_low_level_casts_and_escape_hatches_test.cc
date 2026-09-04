#include <Cellerator/compiler/sema/implement_explicit_low_level_casts_and_escape_hatches_v1.hh>

#include <cassert>

int main() {
    using namespace cellerator::compiler::sema::v1;
    float data[4]{};
    axis_type axis{};
    axis.global_extent = axis.local_extent = 4;
    state_type destination{&axis, 1, cellerator::execution::numeric_type::f32, 1,
        cellerator::execution::residency_kind::host};
    ordinary_cxx_view source{data, cellerator::execution::numeric_type::f64,
        cellerator::execution::residency_kind::host, 1, {4}};
    assert(cast_to_semantic_state(source, destination, semantic_cast_mode::checked,
                                  "read").status == semantic_cast_status::contract_mismatch);
    const auto expert = cast_to_semantic_state(source, destination,
        semantic_cast_mode::unsafe, "read_write");
    assert(expert.status == semantic_cast_status::ok && expert.warning);
    source.rank = 5;
    assert(cast_to_semantic_state(source, destination, semantic_cast_mode::unsafe,
                                  "read").status == semantic_cast_status::unrepresentable_rank);
}
