#pragma once

#include <stdint.h>
#include <Cellerator/common/real/fp16.hh>
#include "CCell/detail/native_views.hh"

namespace velocitypp::detail {

constexpr size_t floating_width(ccell::detail::DType type) noexcept {
    using ccell::detail::DType;
    return type == DType::float16 ? sizeof(uint16_t) : type == DType::float32 ? sizeof(float) :
           type == DType::float64 ? sizeof(double) : 0;
}

// Call after validating the floating encoding. Integer indices use their own loads.
template<class Run> auto with_source_type(ccell::detail::DType type, Run run) {
    using ccell::detail::DType;
    if (type == DType::float16) return run.template operator()<uint16_t>();
    if (type == DType::float32) return run.template operator()<float>();
    return run.template operator()<double>();
}

template<class Storage> auto source_value(const unsigned char* p) noexcept {
    Storage value;
    __builtin_memcpy(&value, p, sizeof(Storage));
    if constexpr (__is_same(Storage, uint16_t)) return cellerator::real::load_f16(value);
    else return value;
}

}
