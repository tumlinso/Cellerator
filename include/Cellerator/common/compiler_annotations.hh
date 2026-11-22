#pragma once

#if defined(__CUDACC__)
#define CELLERATOR_HOST_DEVICE __host__ __device__
#define CELLERATOR_CUDA_FORCEINLINE __forceinline__
#else
#define CELLERATOR_HOST_DEVICE
#define CELLERATOR_CUDA_FORCEINLINE inline
#endif
