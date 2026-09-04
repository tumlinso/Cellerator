#include <Cellerator/compiler/tooling/forward_completion_hover_navigation_and_rename_v1.hh>
#include <algorithm>
namespace Cellerator::compiler::tooling {
language_feature_result_v1 merge_language_features_v1(language_feature_v1 feature,
 std::vector<language_feature_item_v1> ordinary,std::vector<language_feature_item_v1> cellerator){
 language_feature_result_v1 out{feature,std::move(ordinary)};
 for(auto &item:cellerator){ item.cellerator=true; auto duplicate=std::find_if(out.items.begin(),out.items.end(),[&](const auto&x){return x.symbol==item.symbol&&x.uri==item.uri&&x.begin==item.begin&&x.end==item.end;}); if(duplicate==out.items.end()) out.items.push_back(std::move(item)); }
 return out;
}
bool rename_is_consistent_v1(const language_feature_result_v1 &result,std::string_view old_name){
 if(result.feature!=language_feature_v1::rename||result.items.empty()) return false;
 return std::all_of(result.items.begin(),result.items.end(),[&](const auto&i){return i.symbol==old_name&&i.begin<=i.end&&!i.uri.empty();});
}
} // namespace Cellerator::compiler::tooling
