#include <cassert>
#include <string_view>
int main() {
    constexpr std::string_view cflags="-I${includedir}";
    constexpr std::string_view libs="-L${libdir} -lCellerator";
    static_assert(cflags.find("/home/")==std::string_view::npos);
    static_assert(libs.find("-lCellerator")!=std::string_view::npos);
    assert(libs.find("${libdir}")!=std::string_view::npos);
}
