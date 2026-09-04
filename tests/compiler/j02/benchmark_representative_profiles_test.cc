#include <array>
#include <cassert>
#include <cmath>
#include <string_view>
int main(){ constexpr std::array<std::string_view,9> phases{"build","exact_scan","sketch","load","mapped_query","state_transfer","branch_join","multi_state","peak_memory"}; const double exact=8.0, estimate=7.5; assert(std::abs(exact-estimate)==0.5); static_assert(phases.size()==9); }
