#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace Cellerator::compiler::frontend::cxx {

inline constexpr std::uint32_t pure_cxx_fallthrough_schema_version_v1 = 1;

enum class pure_cxx_fallthrough_mode_v1 : std::uint8_t {
    direct_driver = 1,
    cellerator_frontend,
};

enum class pure_cxx_fallthrough_status_v1 : std::uint8_t {
    success = 0,
    schema_mismatch,
    empty_driver_arguments,
};

struct pure_cxx_fallthrough_request_v1 {
    std::uint32_t schema_version = pure_cxx_fallthrough_schema_version_v1;
    std::string source;
    std::vector<std::string> original_driver_arguments;
};

struct pure_cxx_fallthrough_plan_v1 {
    pure_cxx_fallthrough_mode_v1 mode = pure_cxx_fallthrough_mode_v1::direct_driver;
    bool construct_cellerator_ast_or_ir = false;
    std::vector<std::string> forwarded_driver_arguments;
    std::uint64_t frontend_scan_nanoseconds = 0;
    std::uint64_t peak_resident_kib = 0;
};

pure_cxx_fallthrough_status_v1 plan_pure_cxx_fallthrough_v1(
    const pure_cxx_fallthrough_request_v1& request,
    pure_cxx_fallthrough_plan_v1* plan) noexcept;

}  // namespace Cellerator::compiler::frontend::cxx
