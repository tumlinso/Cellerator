#pragma once

#ifndef CELLSHARD_ENABLE_CELLERATOR_QUANTIZED
#define CELLSHARD_ENABLE_CELLERATOR_QUANTIZED 1
#endif

#include <CellShard/formats/dense.cuh>
#include <CellShard/formats/compressed.cuh>
#include <Cellerator/tensor/base/blocked_ell.cuh>
#if CELLSHARD_ENABLE_CELLERATOR_QUANTIZED
#include "formats/quantized_blocked_ell.cuh"
#endif
#include <Cellerator/tensor/base/sliced_ell.cuh>
#include <CellShard/formats/triplet.cuh>
#include <Cellerator/tensor/base/diagonal.cuh>
