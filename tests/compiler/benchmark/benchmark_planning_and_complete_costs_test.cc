#include <array>
#include <cassert>
#include <string_view>
int main(){ constexpr std::array<std::string_view,8> metrics{"decomposition_portfolio","candidate_count","transition_cost","oracle_regret","external_cost","cache_hit","resumption","bounded_failure"}; static_assert(metrics.size()==8); assert(metrics.back()=="bounded_failure"); }
