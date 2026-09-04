#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace cellerator::compiler::tooling::v1 {

struct completion_item {
    std::string spelling;
    std::string category;
};

[[nodiscard]] std::vector<completion_item>
complete_cellerator_syntax(std::string_view source, std::size_t cursor);

}  // namespace cellerator::compiler::tooling::v1
