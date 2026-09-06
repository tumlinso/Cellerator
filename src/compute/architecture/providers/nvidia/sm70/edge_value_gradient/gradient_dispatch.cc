#include <Cellerator/compute/architecture/providers/nvidia/sm70/edge_value_gradient/hybrid_gradient.cuh>

namespace cellerator::compute::architecture::providers::nvidia::sm70::edge_value_gradient {
contract::status_v1 select_gradient_route(bool half_rounded,
    gradient_choice choice, std::uint64_t tile_count,
    gradient_selection &selection) noexcept {
    using contract::status_v1;
    selection = {};
    switch (choice) {
        case gradient_choice::automatic:
            selection.reason = half_rounded
                ? "rounded sparse: no measured hybrid crossover promoted"
                : "full-f32 sparse: preserves requested operand arithmetic";
            return status_v1::success;
        case gradient_choice::force_sparse:
            selection.reason = half_rounded
                ? "forced rounded sparse: fresh RNE packs and complete sparse dot"
                : "forced full-f32 sparse: complete f32 dot";
            return status_v1::success;
        case gradient_choice::force_hybrid:
            if (!half_rounded) {
                selection.reason = "WMMA requires explicit half-rounded operands";
                return status_v1::unsupported;
            }
            if (!tile_count || tile_count > absent_slot / 256u) {
                selection.reason = "no legal nonempty bounded rectangular cover";
                return status_v1::unsupported;
            }
            selection.use_hybrid = true;
            selection.reason = "forced exact gathered WMMA with sparse residual";
            return status_v1::success;
    }
    selection.reason = "unknown gradient choice";
    return status_v1::invalid_argument;
}
}
