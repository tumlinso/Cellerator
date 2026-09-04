#include <Cellerator/compiler/backend/implement_cpu_segment_gate_update_bundle_and_chain_paths_v1.hh>

#include <cassert>
#include <cmath>

namespace cb = cellerator::compiler::backend::v1;

int main() {
    const std::uint64_t offsets[]{0, 2, 5};
    const float input[]{1, 3, 2, 4, 6};
    float reduced[2]{};
    assert(cb::run_cpu_segment_v1(
        {cb::cpu_segment_kind_v1::sum, offsets, 2, input, reduced})
        == cb::cpu_fallback_status_v1::success);
    assert(reduced[0] == 4 && reduced[1] == 12);
    float normalized[5]{};
    assert(cb::run_cpu_segment_v1(
        {cb::cpu_segment_kind_v1::softmax, offsets, 2, input, normalized})
        == cb::cpu_fallback_status_v1::success);
    assert(std::abs(normalized[0] + normalized[1] - 1.0F) < 1e-6F);
    assert(std::abs(normalized[2] + normalized[3] + normalized[4] - 1.0F) < 1e-6F);

    const std::uint8_t predicate[]{1, 0, 1};
    float gated[3]{};
    assert(cb::run_cpu_gate_v1({cb::cpu_gate_kind_v1::predicate,
               input, predicate, gated, 3}) == cb::cpu_fallback_status_v1::success);
    assert(gated[0] == 1 && gated[1] == 0 && gated[2] == 2);

    float values[]{1, 2, 3};
    const std::uint64_t indices[]{2, 0};
    const float updates[]{5, -1};
    assert(cb::run_cpu_sparse_update_v1({values, 3, indices, updates, 2})
        == cb::cpu_fallback_status_v1::success);
    assert(values[0] == 0 && values[2] == 8);

    const float other[]{10, 20, 30};
    const float* members[]{values, other};
    float bundle[3]{};
    assert(cb::run_cpu_bundle_v1({members, 2, 3, bundle})
        == cb::cpu_fallback_status_v1::success);
    assert(bundle[0] == 10 && bundle[1] == 22 && bundle[2] == 38);

    const cb::cpu_chain_stage_v1 stages[]{{2, 1}, {3, -2}};
    float chain[3]{};
    assert(cb::run_cpu_chain_v1({values, chain, 3, stages, 2})
        == cb::cpu_fallback_status_v1::success);
    assert(chain[0] == 1 && chain[1] == 13 && chain[2] == 49);
}
