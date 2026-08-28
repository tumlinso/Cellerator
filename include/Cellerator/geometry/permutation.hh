#pragma once

#include "Cellerator/geometry/format.hh"

namespace cellpack {

inline bool build_inverse_permutation(const u32 *permutation, u32 count, u32 *inverse) {
    if ((count != 0u && permutation == nullptr) || (count != 0u && inverse == nullptr)) return false;
    for (u32 i = 0; i < count; ++i) inverse[i] = invalid_id;
    for (u32 i = 0; i < count; ++i) {
        const u32 original = permutation[i];
        if (original >= count || inverse[original] != invalid_id) return false;
        inverse[original] = i;
    }
    return true;
}

inline bool validate_permutation(const u32 *permutation, u32 count) {
    if (count != 0u && permutation == nullptr) return false;
    for (u32 i = 0; i < count; ++i) {
        if (permutation[i] >= count) return false;
        for (u32 j = i + 1u; j < count; ++j) {
            if (permutation[i] == permutation[j]) return false;
        }
    }
    return true;
}

inline bool validate_inverse_permutation(const u32 *permutation, const u32 *inverse, u32 count) {
    if ((count != 0u && permutation == nullptr) || (count != 0u && inverse == nullptr)) return false;
    for (u32 permuted = 0; permuted < count; ++permuted) {
        const u32 original = permutation[permuted];
        if (original >= count || inverse[original] != permuted) return false;
    }
    return true;
}

inline bool is_identity_permutation(const u32 *permutation, u32 count) {
    if (count != 0u && permutation == nullptr) return false;
    for (u32 i = 0; i < count; ++i) {
        if (permutation[i] != i) return false;
    }
    return true;
}

} // namespace cellpack
