#include <array>
#include <cassert>
#include <string_view>

int main() {
    constexpr std::array<std::string_view, 5> layers{
        "core", "biology", "operations", "planning", "interop"};
    constexpr std::string_view install_root = "share/cellerator/stdlib";

    static_assert(layers.front() == "core");
    static_assert(layers.back() == "interop");
    static_assert(install_root.find("build") == std::string_view::npos);
    for (std::size_t consumer = 0; consumer < layers.size(); ++consumer) {
        for (std::size_t dependency = 0; dependency < consumer; ++dependency) {
            assert(dependency < consumer);
            assert(layers[dependency] != layers[consumer]);
        }
    }
}
