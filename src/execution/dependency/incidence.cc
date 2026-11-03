#include <Cellerator/execution/dependency/incidence.hh>

#include <utility>

namespace cellerator::execution {

PreparedIncidence::PreparedIncidence(ActiveWorklist worklist,
                                     ::cellerator::layout::EntityLayout layout)
    : worklist_(std::move(worklist)), layout_(std::move(layout)), offsets_{0} {
    for (const auto ordinal : worklist_.ordinals()) {
        const auto& edge = worklist_.pattern()->edges()[ordinal];
        for (std::uint32_t i = 0; i < edge.participant_count; ++i) {
            const auto& participant = worklist_.pattern()->participants()[edge.participant_begin + i];
            slots_.push_back(layout_.slot(participant.entity_id));
            roles_.push_back(participant.role);
        }
        offsets_.push_back(slots_.size());
    }
}

} // namespace cellerator::execution
