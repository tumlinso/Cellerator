#pragma once

#include "real.cuh"
#include <Cellerator/common/index_types.hh>

namespace cellshard::types {

using ::cellerator::types::dim_t;
using ::cellerator::types::idx_t;
using ::cellerator::types::nnz_t;
using ::cellerator::types::ptr_t;
using ::cellerator::types::u32;
using ::cellerator::types::u64;
using storage_value_t = real::storage_t;
using ::cellerator::real::value_f32;

template<class T> struct value_code;
template<> struct value_code<u32> { enum { code = 1 }; };
template<> struct value_code<storage_value_t> { enum { code = real::storage_code }; };

}
