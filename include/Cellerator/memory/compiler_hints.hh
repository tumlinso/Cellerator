#pragma once

// These hints are valid only after a checked boundary established the assumed
// condition. They never replace capacity, alignment, identity, or residency
// validation.
#if defined(__CUDACC__)
#define CELLERATOR_RESTRICT __restrict__
#define CELLERATOR_FORCEINLINE __forceinline__
#define CELLERATOR_NOINLINE __noinline__
#elif defined(__clang__) || defined(__GNUC__)
#define CELLERATOR_RESTRICT __restrict__
#define CELLERATOR_FORCEINLINE inline __attribute__((always_inline))
#define CELLERATOR_NOINLINE __attribute__((noinline))
#else
#define CELLERATOR_RESTRICT
#define CELLERATOR_FORCEINLINE inline
#define CELLERATOR_NOINLINE
#endif

#if defined(__clang__)
#define CELLERATOR_ASSUME(condition) __builtin_assume(condition)
#elif defined(__GNUC__) || defined(__CUDACC__)
#define CELLERATOR_ASSUME(condition) \
    do { if (!(condition)) __builtin_unreachable(); } while (false)
#else
#define CELLERATOR_ASSUME(condition) ((void) 0)
#endif

#if defined(__clang__) || defined(__GNUC__)
#define CELLERATOR_ASSUME_ALIGNED(pointer, alignment) \
    __builtin_assume_aligned((pointer), (alignment))
#define CELLERATOR_LIKELY(condition) __builtin_expect(!!(condition), 1)
#define CELLERATOR_UNLIKELY(condition) __builtin_expect(!!(condition), 0)
#else
#define CELLERATOR_ASSUME_ALIGNED(pointer, alignment) (pointer)
#define CELLERATOR_LIKELY(condition) (condition)
#define CELLERATOR_UNLIKELY(condition) (condition)
#endif
