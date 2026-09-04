#include <array>
#include <cassert>
#include <string_view>
int main(){ constexpr std::array<std::string_view,4> levels{"semantic","planning","realization","executable"}; constexpr std::array<std::string_view,9> phases{"construct","canonicalize","text_parse","text_print","binary_load","binary_store","map","unknown_extension","strip_provenance"}; static_assert(levels.size()*phases.size()==36); for(auto x:phases) assert(!x.empty()); }
