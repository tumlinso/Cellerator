#pragma once

#include <cstdint>

#ifndef CELLERATOR_ENABLE_PROFILING_MARKERS
#define CELLERATOR_ENABLE_PROFILING_MARKERS 0
#endif

namespace cellerator::profiling {

struct static_profile_marker_v1 {
    std::uint64_t correlation_id = 0;
    std::uint64_t kernel_symbol_id = 0;
    const char* candidate_name = nullptr;
    const char* stage_name = nullptr;
    const char* kernel_symbol = nullptr;
};

using marker_callback_v1 = void (*)(
        const static_profile_marker_v1& marker, void* context) noexcept;

struct marker_sink_v1 {
    marker_callback_v1 begin = nullptr;
    marker_callback_v1 end = nullptr;
    void* context = nullptr;
};

bool validate_static_marker_registry_v1(
        const static_profile_marker_v1* markers,
        std::uint64_t marker_count) noexcept;

#if CELLERATOR_ENABLE_PROFILING_MARKERS
inline void emit_profile_begin_v1(const marker_sink_v1& sink,
                                  const static_profile_marker_v1& marker) noexcept {
    if (sink.begin != nullptr) sink.begin(marker, sink.context);
}
inline void emit_profile_end_v1(const marker_sink_v1& sink,
                                const static_profile_marker_v1& marker) noexcept {
    if (sink.end != nullptr) sink.end(marker, sink.context);
}
#define CELLERATOR_PROFILE_BEGIN_V1(sink, marker) \
    ::cellerator::profiling::emit_profile_begin_v1((sink), (marker))
#define CELLERATOR_PROFILE_END_V1(sink, marker) \
    ::cellerator::profiling::emit_profile_end_v1((sink), (marker))
#else
#define CELLERATOR_PROFILE_BEGIN_V1(sink, marker) ((void)0)
#define CELLERATOR_PROFILE_END_V1(sink, marker) ((void)0)
#endif

}  // namespace cellerator::profiling
