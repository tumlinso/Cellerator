include_guard(GLOBAL)

# Provider inclusion and provider-local tuning are independent controls.  A
# tuning profile never enables a provider, and selecting a provider never
# silently opts into approximate arithmetic.
set(CELLERATOR_PROVIDER_SELECTION "baseline" CACHE STRING
    "Compiled CUDA provider set: baseline, generic, sm70, or all")
set_property(CACHE CELLERATOR_PROVIDER_SELECTION PROPERTY STRINGS
    baseline generic sm70 all)

set(CELLERATOR_PROVIDER_TUNING_PROFILE "precise" CACHE STRING
    "Provider-local CUDA tuning profile: precise, throughput, or profiling")
set_property(CACHE CELLERATOR_PROVIDER_TUNING_PROFILE PROPERTY STRINGS
    precise throughput profiling)

option(CELLERATOR_ENABLE_EXPERIMENTAL_CANDIDATES
    "Compile explicitly requestable, never-auto-promoted provider candidates"
    OFF)
option(CELLERATOR_ENABLE_CUDA_PROFILING_MARKERS
    "Compile provider-local profiler marker call sites" OFF)
option(CELLERATOR_PROVIDER_APPROXIMATE_MATH
    "Permit provider-local approximate CUDA math where the candidate numerical contract allows it"
    OFF)
set(CELLERATOR_PROVIDER_CACHE_POLICY "default" CACHE STRING
    "Provider-local PTX cache policy: default, ca, or cg")
set_property(CACHE CELLERATOR_PROVIDER_CACHE_POLICY PROPERTY STRINGS
    default ca cg)
set(CELLERATOR_PROVIDER_MAX_REGISTERS "0" CACHE STRING
    "Optional provider-local CUDA register ceiling; zero leaves it unset")

cellerator_require_choice(CELLERATOR_PROVIDER_SELECTION
    baseline generic sm70 all)
cellerator_require_choice(CELLERATOR_PROVIDER_TUNING_PROFILE
    precise throughput profiling)
cellerator_require_choice(CELLERATOR_PROVIDER_CACHE_POLICY default ca cg)
if(NOT CELLERATOR_PROVIDER_MAX_REGISTERS MATCHES "^[0-9]+$")
    message(FATAL_ERROR
        "CELLERATOR_PROVIDER_MAX_REGISTERS must be a non-negative integer")
endif()

set(CELLERATOR_PROVIDER_INCLUDE_GENERIC 0)
set(CELLERATOR_PROVIDER_INCLUDE_SM70 0)
if(CELLERATOR_PROVIDER_SELECTION STREQUAL "baseline")
    # Preserve the pre-CE-EXOP source-linked inventory while making its binary
    # content explicit and independent of the configuring host.
    set(CELLERATOR_PROVIDER_INCLUDE_GENERIC 1)
    set(CELLERATOR_PROVIDER_INCLUDE_SM70 1)
elseif(CELLERATOR_PROVIDER_SELECTION STREQUAL "generic")
    set(CELLERATOR_PROVIDER_INCLUDE_GENERIC 1)
elseif(CELLERATOR_PROVIDER_SELECTION STREQUAL "sm70")
    set(CELLERATOR_PROVIDER_INCLUDE_SM70 1)
else()
    set(CELLERATOR_PROVIDER_INCLUDE_GENERIC 1)
    set(CELLERATOR_PROVIDER_INCLUDE_SM70 1)
endif()

set(CELLERATOR_PROVIDER_TUNING_PRECISE 0)
set(CELLERATOR_PROVIDER_TUNING_THROUGHPUT 0)
set(CELLERATOR_PROVIDER_TUNING_PROFILING 0)
if(CELLERATOR_PROVIDER_TUNING_PROFILE STREQUAL "precise")
    set(CELLERATOR_PROVIDER_TUNING_PRECISE 1)
elseif(CELLERATOR_PROVIDER_TUNING_PROFILE STREQUAL "throughput")
    set(CELLERATOR_PROVIDER_TUNING_THROUGHPUT 1)
else()
    set(CELLERATOR_PROVIDER_TUNING_PROFILING 1)
endif()

message(STATUS
    "Cellerator providers: selection=${CELLERATOR_PROVIDER_SELECTION}, "
    "tuning=${CELLERATOR_PROVIDER_TUNING_PROFILE}")

add_library(cellerator_provider_build_policy INTERFACE)
add_library(Cellerator::provider_build_policy ALIAS
    cellerator_provider_build_policy)
target_compile_definitions(cellerator_provider_build_policy INTERFACE
    CELLERATOR_ENABLE_EXPERIMENTAL_CANDIDATES=$<BOOL:${CELLERATOR_ENABLE_EXPERIMENTAL_CANDIDATES}>
    CELLERATOR_ENABLE_CUDA_PROFILING_MARKERS=$<BOOL:${CELLERATOR_ENABLE_CUDA_PROFILING_MARKERS}>
    CELLERATOR_PROVIDER_APPROXIMATE_MATH=$<BOOL:${CELLERATOR_PROVIDER_APPROXIMATE_MATH}>
    CELLERATOR_PROVIDER_TUNING_PRECISE=${CELLERATOR_PROVIDER_TUNING_PRECISE}
    CELLERATOR_PROVIDER_TUNING_THROUGHPUT=${CELLERATOR_PROVIDER_TUNING_THROUGHPUT}
    CELLERATOR_PROVIDER_TUNING_PROFILING=${CELLERATOR_PROVIDER_TUNING_PROFILING})

# Apply provider-local policy only to a concrete provider target.  Precise is
# the default in every profile.  Approximate math, cache forcing, register
# ceilings, line information, and markers each require an explicit cache
# input and never leak into unrelated CUDA targets.
function(cellerator_apply_provider_build_policy target)
    if(NOT TARGET ${target})
        message(FATAL_ERROR "unknown CUDA provider target: ${target}")
    endif()
    target_link_libraries(${target} PRIVATE
        Cellerator::provider_build_policy)

    set(provider_cuda_flags -O3 --expt-relaxed-constexpr
        --expt-extended-lambda)
    if(CELLERATOR_PROVIDER_APPROXIMATE_MATH)
        list(APPEND provider_cuda_flags --use_fast_math)
    endif()
    if(NOT CELLERATOR_PROVIDER_CACHE_POLICY STREQUAL "default")
        list(APPEND provider_cuda_flags
            "-Xptxas=-dlcm=${CELLERATOR_PROVIDER_CACHE_POLICY}")
    endif()
    if(CELLERATOR_PROVIDER_MAX_REGISTERS GREATER 0)
        list(APPEND provider_cuda_flags
            "--maxrregcount=${CELLERATOR_PROVIDER_MAX_REGISTERS}")
    endif()
    if(CELLERATOR_ENABLE_CUDA_LINEINFO
       OR CELLERATOR_PROVIDER_TUNING_PROFILING)
        list(APPEND provider_cuda_flags -lineinfo)
    endif()
    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:${provider_cuda_flags}>)
endfunction()
