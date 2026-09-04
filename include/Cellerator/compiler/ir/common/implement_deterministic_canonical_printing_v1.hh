#pragma once

#include <Cellerator/compiler/ir/common/implement_common_operation_and_extension_records_v1.hh>

#include <cstdint>
#include <string>
#include <vector>

namespace cellerator::compiler::ir::text {

struct print_document {
    std::uint16_t major{1u};
    std::uint16_t minor{0u};
    std::vector<common_operation> operations;
};

std::string canonical_print(const print_document &document);
std::string pretty_print(const print_document &document);

} // namespace cellerator::compiler::ir::text
