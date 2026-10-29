#include <array>
#include <cassert>
#include <string_view>
int main(){ constexpr std::array<std::string_view,10> operations{"session_start","concurrent_parse","cancel","editor_start","diagnostics","completion","hover","ir_query","candidate_query","peak_memory"}; constexpr bool ordinary_cxx_guard=true; static_assert(operations.size()==10 && ordinary_cxx_guard); assert(operations[6]=="hover"); }
