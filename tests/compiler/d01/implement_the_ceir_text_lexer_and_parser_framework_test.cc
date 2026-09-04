#include <Cellerator/compiler/ir/common/implement_the_ceir_text_lexer_and_parser_framework_v1.hh>

#include <cassert>
#include <random>
#include <string>

using namespace cellerator::compiler::ir::text;

int main() {
    parser ceir;
    ceir.register_dialect("semantic", [](std::string_view name) {
        return name == "semantic.apply";
    });
    const std::string source = "include \"types.ceir\"\nimport semantic\nsemantic.apply { bad ??? }";
    const auto unit = ceir.parse(source);
    assert(unit.includes == std::vector<std::string>{"types.ceir"});
    assert(unit.imports == std::vector<std::string>{"semantic"});
    assert(unit.operations.size() == 1u && unit.operations[0].inline_block);
    assert(unit.operations[0].range.begin == source.find("semantic.apply"));

    const auto malformed = ceir.parse("include ; unknown.op { \"unterminated");
    assert(!malformed.diagnostics.empty());
    for (const auto &diagnostic : malformed.diagnostics)
        assert(diagnostic.range.begin <= diagnostic.range.end);

    std::mt19937 random(19u);
    std::string fuzz(128u, ' ');
    for (unsigned iteration = 0; iteration < 1000u; ++iteration) {
        for (auto &character : fuzz)
            character = static_cast<char>(random() & 0x7fu);
        const auto parsed = ceir.parse(fuzz);
        for (const auto &diagnostic : parsed.diagnostics)
            assert(diagnostic.range.end <= fuzz.size());
    }
}
