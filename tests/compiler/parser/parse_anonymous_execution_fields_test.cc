#include <Cellerator/compiler/frontend/parser/parse_anonymous_execution_fields_v1.hh>

#include <cassert>
#include <string>

using namespace Cellerator::compiler::frontend::parser;

int main() {
    const std::string source = R"cpp(
template<class T> bool ordinary(T x) { return x < [&]{ return 1; }(); }
auto shifted = value << 2; [[maybe_unused]] auto text = "<[ not a field ]>";
<[
    given ce::profile(pbmc);
    prefer ce::minimum_latency;
::
    if (a < b) { output = lambda([&] { return value >> 1; }); }
    <[ nested = input; ]>
]>
)cpp";
    const auto parsed = parse_anonymous_execution_fields_v1(source);
    assert(parsed.accepted());
    // Ordinary separated '< [' remains delegated to C++.
    assert(parsed.fields.size() == 1);
    const auto &field = parsed.fields.back();
    assert(field.planning_attributes.size() == 2);
    assert(field.captured_cxx_statements.size() == 2);
    assert(field.nested_fields.size() == 1);
    assert(source.substr(field.range.begin, 2) == "<[");
    assert(source.substr(field.range.end - 2, 2) == "]>");

    for (int index = 0; index != 128; ++index) {
        const std::string prefix = index % 2 ? "template<class T> " : "[[nodiscard]] ";
        const auto fuzzed = parse_anonymous_execution_fields_v1(
            prefix + "auto x = (a < b) ? (v >> 1) : f([&]{return 0;}); <[ x += 1; ]>");
        assert(fuzzed.accepted());
        assert(fuzzed.fields.size() == 1);
    }
    assert(!parse_anonymous_execution_fields_v1("<[ missing").accepted());
}
