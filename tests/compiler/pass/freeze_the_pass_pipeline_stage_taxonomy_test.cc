#include <Cellerator/compiler/pass/freeze_the_pass_pipeline_stage_taxonomy_v1.hh>

#include <cassert>
#include <set>

namespace cp = cellerator::compiler::pass::v1;
namespace cr = cellerator::compiler::ir::realization::v1;

int main() {
    std::set<std::uint16_t> ids;
    for (std::size_t phase = 0; phase < cp::pipeline_phase_count_v1; ++phase) {
        const auto value = static_cast<cp::pipeline_phase_v1>(phase);
        assert(!cp::pipeline_phase_name_v1(value).empty());
        assert(cp::lowering_resumption_facet_v1(value) != cr::ceir_facet_v1::count);
        for (auto side : {cp::interception_side_v1::before,
                          cp::interception_side_v1::after}) {
            const cp::pipeline_stage_v1 stage{value, side};
            assert(cp::valid_pipeline_stage_v1(stage));
            assert(ids.insert(cp::stable_stage_id_v1(stage)).second);
        }
    }
    assert(ids.size() == cp::pipeline_stage_count_v1);
    assert(!cp::valid_pipeline_stage_v1(
        {cp::pipeline_phase_v1::count, cp::interception_side_v1::before}));
}
