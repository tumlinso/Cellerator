#pragma once

#include <cstdint>
#include <string>

namespace cellerator::compiler::sema::v1 {

enum class semantic_mismatch : std::uint8_t {
    none = 0,
    domain,
    order,
    structure_generation,
    value_generation,
    support_generation,
    numerical_policy,
    operation_resolution
};

struct semantic_explanation {
    semantic_mismatch mismatch = semantic_mismatch::none;
    std::string subject;
    std::string expected;
    std::string actual;
    std::string diagnostic;

    explicit operator bool() const noexcept { return mismatch == semantic_mismatch::none; }
};

semantic_explanation explain_semantic_compatibility(
    semantic_mismatch mismatch,
    std::string subject,
    std::string expected,
    std::string actual);
const char *semantic_mismatch_name(semantic_mismatch mismatch) noexcept;

}  // namespace cellerator::compiler::sema::v1
