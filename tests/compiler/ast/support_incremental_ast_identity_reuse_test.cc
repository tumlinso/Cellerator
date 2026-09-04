#include <Cellerator/compiler/ast/support_incremental_ast_identity_reuse_v1.hh>

#include <cassert>
#include <iostream>

using namespace Cellerator::compiler::ast;

int main() {
    std::vector<incremental_ast_identity_v1> previous;
    std::vector<incremental_ast_identity_v1> current;
    constexpr std::size_t count = 10'000;
    previous.reserve(count);
    current.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        const bool macro = index % 100 == 0;
        const bool templ = index % 137 == 0;
        incremental_ast_identity_v1 record{
            index + 1, incremental_identity_level_v1::subtree, index + 10,
            index + 100, index + 200, macro, templ, macro ? 77U : 0U, templ ? 88U : 0U};
        previous.push_back(record);
        record.identity = 0;
        if (index >= 4000 && index < 4020) ++record.content_hash;
        if (index == 5000) ++record.macro_environment_hash;
        if (index == 5069) ++record.template_environment_hash;
        current.push_back(record);
    }
    std::string error;
    auto result = reuse_incremental_ast_identities_v1(previous, current, &error);
    assert(result && error.empty());
    assert(result->metrics.total == count);
    assert(result->metrics.invalidated == 22);
    assert(result->metrics.created == 22);
    assert(result->metrics.reused == count - 22);
    assert(result->metrics.reused_fraction() > 0.99);
    assert(result->identities.front().identity == previous.front().identity);
    assert(result->identities[4000].identity > count);

    auto missing_macro_environment = current;
    missing_macro_environment[100].macro_environment_hash = 0;
    auto conservative = reuse_incremental_ast_identities_v1(previous,
                                                             missing_macro_environment, &error);
    assert(conservative);
    assert(conservative->identities[100].identity > count);

    std::cout << "nodes=" << count << " reused=" << result->metrics.reused
              << " reused_fraction=" << result->metrics.reused_fraction() << '\n';
}
