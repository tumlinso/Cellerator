#include <array>
#include <cassert>
#include <string_view>
int main(){ constexpr std::array<std::string_view,6> mechanisms{"support_sketch","signature","atom_plane","fragment","grammar_basis","exact_scan"}; constexpr std::array<std::string_view,3> baselines{"matched_generic","matched_null","no_basis"}; static_assert(mechanisms.size()*baselines.size()==18); for(auto x:mechanisms) assert(!x.empty()); }
