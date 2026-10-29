#include <cassert>
#include <string_view>
struct typed_name { std::string_view text; unsigned long long identity; };
struct profile_selection { typed_name name; bool explicit_selection; };
constexpr profile_selection select_profile(typed_name name) { return {name, true}; }
int main() {
    constexpr auto human = select_profile({"human-test-v1", 1});
    constexpr auto mouse = select_profile({"mouse-test-v1", 2});
    static_assert(human.explicit_selection && mouse.explicit_selection);
    assert(human.name.identity != mouse.name.identity);
}
