#pragma once

#include <Cellerator/execution/dependency/workset.hh>
#include <Cellerator/layout/entity_layout.hh>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace cellerator::execution {

// Host-side physical realization of selected logical pattern incidences.
// Topology, activity and value order are frozen; numerical values are supplied
// at invocation. Repeated incidences remain repeated.
class PreparedIncidence final {
public:
    PreparedIncidence(ActiveWorklist worklist, ::cellerator::layout::EntityLayout layout);

    const ActiveWorklist& worklist() const noexcept { return worklist_; }
    const ::cellerator::layout::EntityLayout& layout() const noexcept { return layout_; }
    const std::vector<std::size_t>& offsets() const noexcept { return offsets_; }
    const std::vector<std::size_t>& slots() const noexcept { return slots_; }
    const std::vector<std::uint32_t>& roles() const noexcept { return roles_; }
    std::size_t edge_count() const noexcept { return worklist_.size(); }
    std::size_t incidence_count() const noexcept { return slots_.size(); }

private:
    ActiveWorklist worklist_;
    ::cellerator::layout::EntityLayout layout_;
    std::vector<std::size_t> offsets_, slots_;
    std::vector<std::uint32_t> roles_;
};

} // namespace cellerator::execution
