#include <Cellerator/compiler/ast/bind_c_ast_references_safely_v1.hh>

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

using namespace Cellerator::compiler::ast;

int main() {
    try {
        static_assert(std::is_trivially_copyable_v<cxx_ast_reference_v1>);
        static_assert(std::is_trivially_copyable_v<cxx_ast_reference_key_v1>);
        static_assert(sizeof(cxx_ast_reference_v1) == 24);

        std::vector<cxx_ast_reference_key_v1> keys{
            {cxx_ast_entity_kind_v1::constant, 7, 50, 500, 15},
            {cxx_ast_entity_kind_v1::declaration, 7, 10, 100, 11},
            {cxx_ast_entity_kind_v1::expression, 7, 20, 200, 12},
            {cxx_ast_entity_kind_v1::type, 7, 30, 300, 13},
            {cxx_ast_entity_kind_v1::template_entity, 7, 40, 400, 14},
        };
        std::string error;
        auto first = freeze_cxx_ast_references_v1(91, 1, 18, keys, &error);
        if (!first || first->size() != keys.size()) {
            throw std::runtime_error(error);
        }
        std::reverse(keys.begin(), keys.end());
        auto reparse = freeze_cxx_ast_references_v1(91, 2, 18, keys, &error);
        if (!reparse) {
            throw std::runtime_error(error);
        }
        const auto old_reference = first->reference(keys.front());
        const auto new_reference = reparse->reference(keys.front());
        if (!old_reference || !new_reference || old_reference->slot != new_reference->slot ||
            reparse->resolve(*old_reference) != nullptr || first->resolve(*new_reference) != nullptr) {
            throw std::runtime_error("reparse generation invalidation is incorrect");
        }

        auto upgraded = freeze_cxx_ast_references_v1(91, 3, 19, keys, &error);
        if (!upgraded || upgraded->resolve(*new_reference) != nullptr) {
            throw std::runtime_error("adapter version invalidation is incorrect");
        }
        auto removed_keys = keys;
        removed_keys.pop_back();
        auto rebuilt = freeze_cxx_ast_references_v1(91, 4, 19, removed_keys, &error);
        if (!rebuilt) {
            throw std::runtime_error(error);
        }
        const auto remap = rebuild_cxx_ast_references_v1(*first, *rebuilt);
        if (remap.size() != first->size() ||
            std::count_if(remap.begin(), remap.end(), [](const auto& item) {
                return item.replacement.has_value();
            }) != static_cast<std::ptrdiff_t>(removed_keys.size())) {
            throw std::runtime_error("rebuild remap did not conservatively invalidate removal");
        }
        auto duplicate = keys;
        duplicate.push_back(keys.front());
        if (freeze_cxx_ast_references_v1(91, 5, 19, std::move(duplicate), &error)) {
            throw std::runtime_error("duplicate adapter key was accepted");
        }

        std::cout << "stable_references=" << first->size()
                  << " rebuilt=" << removed_keys.size() << " invalidated=1\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
