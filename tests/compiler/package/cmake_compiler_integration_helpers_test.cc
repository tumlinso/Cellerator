#include <array>
#include <cassert>
#include <string_view>
int main() {
    constexpr std::array<std::string_view,6> contract{"PROFILE","BACKEND","STDLIB","DEPFILE","OPTIONS","LTO"};
    for (auto item: contract) assert(!item.empty());
    static_assert(contract[0] == "PROFILE");
}
