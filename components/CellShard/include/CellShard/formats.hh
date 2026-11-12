#pragma once

#ifndef CELLSHARD_ENABLE_CELLERATOR_QUANTIZED
#define CELLSHARD_ENABLE_CELLERATOR_QUANTIZED 1
#endif

#include <Cellerator/tensor/base/dense.cuh>
#include <Cellerator/tensor/base/compressed.cuh>
#include <Cellerator/tensor/base/blocked_ell.cuh>
#if CELLSHARD_ENABLE_CELLERATOR_QUANTIZED
#include "formats/quantized_blocked_ell.cuh"
#endif
#include <Cellerator/tensor/base/sliced_ell.cuh>
#include <Cellerator/tensor/base/triplet.cuh>
#include <Cellerator/tensor/base/diagonal.cuh>
