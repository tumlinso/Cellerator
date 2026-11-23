#pragma once

#include "cuda_compat.cuh"
#include <Cellerator/common/real/format_codes.hh>
#include <Cellerator/common/real/fp16.cuh>

namespace cellshard::real {

using storage_t = ::cellerator::real::f16_t;
using ::cellerator::real::value_f16;
inline constexpr int storage_code = value_f16;

}

namespace real = ::cellshard::real;
