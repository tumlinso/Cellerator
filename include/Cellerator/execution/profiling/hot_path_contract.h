#pragma once

#include <cstdint>

#ifndef CELLERATOR_ENABLE_PROFILING_MARKERS
#define CELLERATOR_ENABLE_PROFILING_MARKERS 0
#endif

namespace cellerator::profiling {

struct disabled_hot_path_contract_v1 {
    bool marker_arguments_evaluated = false;
    bool callback_dereference = false;
    bool dynamic_string_work = false;
    bool allocation = false;
    bool synchronization = false;
    std::uint8_t reserved[3]{};
};

inline constexpr bool profiling_markers_compiled_v1 =
        CELLERATOR_ENABLE_PROFILING_MARKERS != 0;

inline constexpr disabled_hot_path_contract_v1
disabled_profiling_hot_path_contract_v1() noexcept {
    return {};
}

#if !CELLERATOR_ENABLE_PROFILING_MARKERS
static_assert(!profiling_markers_compiled_v1);
static_assert(!disabled_profiling_hot_path_contract_v1().marker_arguments_evaluated);
static_assert(!disabled_profiling_hot_path_contract_v1().callback_dereference);
static_assert(!disabled_profiling_hot_path_contract_v1().dynamic_string_work);
static_assert(!disabled_profiling_hot_path_contract_v1().allocation);
static_assert(!disabled_profiling_hot_path_contract_v1().synchronization);
#endif

}  // namespace cellerator::profiling
