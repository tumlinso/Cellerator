#pragma once

#include "supernode_reduce.cuh"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace cellerator::compute::graph {

inline void detect_branch_supernodes(
    const SupernodeDag &dag,
    SupernodeTable *supernodes,
    float min_branch_mass,
    float min_branch_ratio) {
    if (supernodes == nullptr) throw std::invalid_argument("detect_branch_supernodes requires a table");
    if (dag.rows != supernodes->count) throw std::invalid_argument("dag.rows must match supernode count");
    if (min_branch_mass < 0.0f || min_branch_ratio < 0.0f) {
        throw std::invalid_argument("branch thresholds must be non-negative");
    }

    std::fill(supernodes->is_branch.begin(), supernodes->is_branch.end(), static_cast<std::uint8_t>(0));
    for (std::uint32_t row = 0; row < dag.rows; ++row) {
        std::vector<float> masses;
        for (std::uint32_t edge = dag.row_ptr[row]; edge < dag.row_ptr[row + 1u]; ++edge) masses.push_back(dag.mass[edge]);
        if (masses.size() < 2u) continue;
        std::sort(masses.begin(), masses.end(), std::greater<float>());
        const float total = masses[0] + masses[1];
        if (masses[1] >= min_branch_mass && masses[1] >= total * min_branch_ratio) {
            supernodes->is_branch[row] = 1u;
        }
    }
}

} // namespace cellerator::compute::graph
