#pragma once

#ifdef CELLERATOR_ENABLE_CUDA
#include <Cellerator/common/cuda_compat.hh>
#endif 

#ifndef __host__
#define __host__
#endif

#ifndef __device__
#define __device__
#endif

#ifndef __global__
#define __global__
#endif

#ifndef __shared__
#define __shared__
#endif
