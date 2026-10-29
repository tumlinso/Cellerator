#include <algorithm>
#include <array>
#include <cassert>
#include <string_view>
int main() {
    std::array<std::string_view,4> files{"stdlib/z.cell","profiles/p","backends/cuda","schemas/ceir"};
    auto first=files; auto second=files;
    std::sort(first.begin(),first.end()); std::sort(second.begin(),second.end());
    assert(first==second);
    for(auto path:first) assert(path.front()!='/' && path.find("/home/")==std::string_view::npos);
}
