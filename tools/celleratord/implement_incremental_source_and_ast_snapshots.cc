#include <Cellerator/compiler/tooling/implement_incremental_source_and_ast_snapshots_v1.hh>

#include <iostream>

int main() {
    using namespace Cellerator::compiler::tooling;
    const std::vector<snapshot_region_v1> snapshot = {
        {1, snapshot_artifact_v1::source_tokens, 1, 1, {}, "tokens", 1}};
    const auto update = update_incremental_snapshot_v1(snapshot, snapshot, 2);
    std::cout << update.metrics.reuse_fraction() << '\n';
    return update.metrics.reused == 1 ? 0 : 1;
}
