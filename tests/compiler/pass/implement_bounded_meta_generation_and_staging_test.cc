#include <Cellerator/compiler/pass/implement_bounded_meta_generation_and_staging_v1.hh>

#include <cassert>

namespace cp = cellerator::compiler::pass::v1;

namespace {
bool none(const cp::meta_generation_context_v1&,
    std::vector<cp::meta_transform_v1>&) noexcept { return true; }
bool one(const cp::meta_generation_context_v1&,
    std::vector<cp::meta_transform_v1>& out) noexcept {
    out.push_back({"late", cp::pipeline_phase_v1::realization, none});
    return true;
}
bool two(const cp::meta_generation_context_v1&,
    std::vector<cp::meta_transform_v1>& out) noexcept {
    out.push_back({"a", cp::pipeline_phase_v1::discovery, none});
    out.push_back({"b", cp::pipeline_phase_v1::certification, none});
    return true;
}
bool recursive(const cp::meta_generation_context_v1& context,
    std::vector<cp::meta_transform_v1>& out) noexcept {
    out.push_back({"recursive", static_cast<cp::pipeline_phase_v1>(
        static_cast<unsigned>(context.phase) + 1), recursive});
    return true;
}
}

int main() {
    assert(cp::run_bounded_meta_generation_v1({{"zero",
        cp::pipeline_phase_v1::discovery, none}}).execution_order.size() == 1);
    assert(cp::run_bounded_meta_generation_v1({{"one",
        cp::pipeline_phase_v1::discovery, one}}).execution_order.size() == 2);
    assert(cp::run_bounded_meta_generation_v1({{"multiple",
        cp::pipeline_phase_v1::profile_propagation, two}}).execution_order.size() == 3);
    const auto cycle = cp::run_bounded_meta_generation_v1({{"recursive",
        cp::pipeline_phase_v1::profile_propagation, recursive}}, {8, 8});
    assert(cycle.status == cp::meta_generation_status_v1::cycle);
    assert(cycle.diagnostic == "recursive -> recursive");
}
