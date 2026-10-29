#pragma once

namespace Cellerator::compiler::build {

struct language_standards_v1 {
    int compiler_implementation;
    int tooling_implementation;
    int public_header_minimum;
    int legacy_runtime;
    int legacy_cuda;
    bool implementation_mode_is_private;
};

inline constexpr language_standards_v1 language_standards_contract_v1{
    23, 23, 17, 17, 17, true,
};

}  // namespace Cellerator::compiler::build
