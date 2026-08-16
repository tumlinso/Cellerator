#pragma once

#ifndef CELLERATOR_DIST_HAS_NCCL
#define CELLERATOR_DIST_HAS_NCCL 0
#endif
#if CELLERATOR_DIST_HAS_NCCL
#if defined(__has_include)
#if __has_include(<nccl.h>)
#include <nccl.h>
#elif __has_include("/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/comm_libs/12.9/nccl/include/nccl.h")
#include "/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/comm_libs/12.9/nccl/include/nccl.h"
#else
#error "CELLERATOR_DIST_HAS_NCCL requires nccl.h"
#endif
#else
#include <nccl.h>
#endif
#endif
