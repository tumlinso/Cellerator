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

cellerator_require_choice(CELLERATOR_PROVIDER_SELECTION
    baseline generic sm70 all)
cellerator_require_choice(CELLERATOR_PROVIDER_TUNING_PROFILE
    precise throughput profiling)

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
