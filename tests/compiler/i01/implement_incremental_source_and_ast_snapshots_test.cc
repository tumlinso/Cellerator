#include <Cellerator/compiler/tooling/implement_incremental_source_and_ast_snapshots_v1.hh>

#include <cassert>

using namespace Cellerator::compiler::tooling;

int main() {
    const std::vector<snapshot_region_v1> previous = {
        {1, snapshot_artifact_v1::source_tokens, 10, 100, {}, "tokens-a", 1},
        {2, snapshot_artifact_v1::include_state, 20, 200, {1}, "include-a", 1},
        {3, snapshot_artifact_v1::shadow_cpp, 30, 300, {2}, "shadow-a", 1},
        {4, snapshot_artifact_v1::cpp_ast_bridge, 40, 400, {3}, "cpp-ast-a", 1},
        {5, snapshot_artifact_v1::cellerator_ast, 50, 500, {4}, "cell-ast-a", 1},
        {6, snapshot_artifact_v1::semantic_ir, 60, 600, {5}, "sema-a", 1},
        {7, snapshot_artifact_v1::semantic_ir, 70, 700, {}, "unrelated", 1},
    };
    auto local_edit = previous;
    local_edit[0].content_hash = 11;
    local_edit[0].payload = "tokens-b";
    for (std::size_t index = 1; index < 6; ++index)
        local_edit[index].payload += "-rebuilt";

    const auto update = update_incremental_snapshot_v1(previous, local_edit, 2);
    assert(update.metrics.rebuilt == 6);
    assert(update.metrics.dependency_invalidated == 5);
    assert(update.metrics.reused == 1);
    assert(update.metrics.reuse_fraction() > 0.14 && update.metrics.reuse_fraction() < 0.15);
    assert(update.regions[6].payload == "unrelated");
    assert(update.regions[6].generation == 1);

    const auto unchanged = update_incremental_snapshot_v1(previous, previous, 2);
    assert(unchanged.metrics.reused == previous.size());
    assert(unchanged.metrics.reuse_fraction() == 1.0);
}
