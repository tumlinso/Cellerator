#pragma once

#include <string_view>

namespace Cellerator::compiler::build {

struct llvm_clang_capability_v1 {
    bool available;
    std::string_view llvm_version;
    std::string_view abi_identity;
    bool upstream_package;
    bool permanent_fork_required;
};

[[nodiscard]] constexpr bool usable(const llvm_clang_capability_v1& value) {
    return value.available && !value.llvm_version.empty() &&
           !value.abi_identity.empty() && value.upstream_package &&
           !value.permanent_fork_required;
}

}  // namespace Cellerator::compiler::build
