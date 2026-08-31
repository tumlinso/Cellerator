#include "Cellerator/profiling/static_markers.h"

namespace cellerator::profiling {

bool validate_static_marker_registry_v1(
        const static_profile_marker_v1* markers,
        std::uint64_t marker_count) noexcept {
    if (marker_count != 0 && markers == nullptr) return false;
    for (std::uint64_t i = 0; i < marker_count; ++i) {
        const auto& marker = markers[i];
        if (marker.correlation_id == 0 || marker.kernel_symbol_id == 0 ||
            marker.candidate_name == nullptr || marker.candidate_name[0] == 0 ||
            marker.stage_name == nullptr || marker.stage_name[0] == 0 ||
            marker.kernel_symbol == nullptr || marker.kernel_symbol[0] == 0 ||
            (i != 0 && (markers[i - 1].correlation_id >= marker.correlation_id ||
                        markers[i - 1].kernel_symbol_id >=
                                marker.kernel_symbol_id))) return false;
    }
    return true;
}

}  // namespace cellerator::profiling
