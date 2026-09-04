#include <Cellerator/compiler/sema/freeze_compiler_semantic_type_categories_v1.hh>

#include <cassert>
#include <cstdint>
#include <set>
#include <string>

int main() {
    using namespace cellerator::compiler::sema::v1;

    assert(semantic_type_category_count() == 15u);
    std::set<std::string> spellings;
    for (std::uint32_t i = 0; i < semantic_type_category_count(); ++i) {
        const auto &descriptor = semantic_type_categories()[i];
        assert(is_valid_semantic_type_descriptor(descriptor));
        assert(spellings.emplace(descriptor.spelling).second);
        assert(find_semantic_type_category(descriptor.category) == &descriptor);
    }

    // Owners, allocators, containers, and layouts are deliberately absent.
    assert(spellings.count("buffer") == 0u);
    assert(spellings.count("csr") == 0u);
    assert(spellings.count("projection") == 0u);
}
