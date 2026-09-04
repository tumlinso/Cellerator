#include <Cellerator/compiler/frontend/source/define_generated_identifier_hygiene_v1.hh>

#include <array>
#include <iostream>
#include <set>
#include <stdexcept>

using namespace Cellerator::compiler::frontend::source;

int main() {
    try {
        std::set<std::string> generated;
        for (auto domain : std::array{generated_identifier_domain_v1::local_symbol,
                                      generated_identifier_domain_v1::type,
                                      generated_identifier_domain_v1::module,
                                      generated_identifier_domain_v1::link_name}) {
            const auto id = make_generated_identifier_v1(domain, "same canonical source");
            if (!id.emitted_after_preprocessing || !is_reserved_generated_identifier_v1(id.spelling) ||
                !generated.insert(id.spelling).second) throw std::runtime_error("identifier domain collision");
        }
        for (auto user : std::array<std::string_view, 5>{"cellerator", "generated_v1", "shadow", "module", "CELLERATOR_MACRO"})
            if (is_reserved_generated_identifier_v1(user) || generated.count(std::string(user)))
                throw std::runtime_error("legal user identifier collided");
        const auto a = make_generated_identifier_v1(generated_identifier_domain_v1::local_symbol, "a");
        const auto b = make_generated_identifier_v1(generated_identifier_domain_v1::local_symbol, "b");
        if (a.content_hash == b.content_hash || a.spelling == b.spelling) throw std::runtime_error("content identity collapsed");
        std::cout << "validated reserved content-derived identifier hygiene\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
