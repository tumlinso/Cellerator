#pragma once

#ifndef CELLERATOR_ENABLE_CUDA
#ifdef CELLSHARD_ENABLE_CUDA
#define CELLERATOR_ENABLE_CUDA CELLSHARD_ENABLE_CUDA
#else
#define CELLERATOR_ENABLE_CUDA 1
#endif
#endif

// A host-only build can use SDK types when the toolkit is visible.
#if CELLERATOR_ENABLE_CUDA || (__has_include(<cuda_runtime.h>) && __has_include(<cuda_fp16.h>))
#define CELLERATOR_DETAIL_USE_CUDA_SDK_TYPES 1
#else
#define CELLERATOR_DETAIL_USE_CUDA_SDK_TYPES 0
#endif
