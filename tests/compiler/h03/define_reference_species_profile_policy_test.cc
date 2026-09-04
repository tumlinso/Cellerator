#include <array>
#include <cassert>
#include <string_view>
struct profile { std::string_view id; unsigned taxonomy; std::string_view warning; bool automatic; };
int main() {
    constexpr std::array<profile,3> profiles{{
        {"homo_sapiens.test.v1",9606,"TEST ONLY",false},
        {"mus_musculus.test.v1",10090,"TEST ONLY",false},
        {"rattus_norvegicus.test.v1",10116,"TEST ONLY",false}}};
    for (const auto& p : profiles) assert(!p.automatic && !p.warning.empty() && p.taxonomy != 0 && p.id.find("test.v1") != std::string_view::npos);
}
