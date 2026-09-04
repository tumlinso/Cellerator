#include <array>
#include <cassert>
#include <string_view>
int main(){
 constexpr std::array<std::string_view,3> corpora{"small","medium","template-heavy"};
 constexpr std::array<std::string_view,8> metrics{"driver","preprocess","compile","link","peak_rss","depfile","object_size","diagnostics"};
 static_assert(corpora.size()*2==6 && metrics.size()==8);
 for(auto x:corpora) assert(!x.empty()); for(auto x:metrics) assert(!x.empty());
}
