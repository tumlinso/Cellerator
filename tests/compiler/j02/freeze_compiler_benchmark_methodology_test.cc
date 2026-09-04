#include <array>
#include <cassert>
#include <string_view>
int main() {
    constexpr unsigned warmups=2, repeats=11;
    constexpr std::array<std::string_view,8> identity{"hardware","topology","toolchain","source","benchmark","binary","profile","input"};
    constexpr std::array<std::string_view,5> statistics{"median","mad","minimum","maximum","bootstrap_95"};
    static_assert(warmups>=2 && repeats>=11 && repeats%2==1);
    for(auto item: identity) assert(!item.empty());
    for(auto item: statistics) assert(!item.empty());
}
