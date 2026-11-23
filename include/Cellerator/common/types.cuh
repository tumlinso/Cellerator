#pragma once

#include "index_types.hh"
#include "real.cuh"

namespace cellerator::types {

enum {
    value_u32 = 1,
    value_i32 = 2
};

static constexpr int value_f16 = real::value_f16;
static constexpr int value_f32 = real::value_f32;
static constexpr int value_f64 = real::value_f64;
static constexpr int value_bf16 = real::value_bf16;
static constexpr int value_fp8_e4m3 = real::value_fp8_e4m3;
static constexpr int value_fp8_e5m2 = real::value_fp8_e5m2;

static constexpr int has_bf16_types = real::has_bf16_types;
static constexpr int has_fp8_types = real::has_fp8_types;

using shard_idx_t = unsigned long;

using storage_value_t = real::storage_t;
using compute_value_t = real::compute_t;
using accum_value_t = real::accum_t;

template<typename T>
struct value_code : real::code_of<T> {};

template<>
struct value_code<u32> {
    enum { code = value_u32 };
};

template<>
struct value_code<i32> {
    enum { code = value_i32 };
};

template<int Code>
struct value_type : real::type_of<Code> {};

template<>
struct value_type<value_u32> {
    using type = u32;
};

template<>
struct value_type<value_i32> {
    using type = i32;
};

} // namespace cellerator::types
