#include <Cellerator/compiler/tooling/forward_completion_hover_navigation_and_rename_v1.hh>
#include <cassert>
using namespace Cellerator::compiler::tooling;
int main(){
 language_feature_item_v1 cpp{"vector","std::vector","a.cc",1,7,false};
 language_feature_item_v1 field{"gene","field Gene","a.cell",10,14,true};
 for(auto feature:{language_feature_v1::completion,language_feature_v1::hover,language_feature_v1::definition,language_feature_v1::references}){auto r=merge_language_features_v1(feature,{cpp},{field});assert(r.items.size()==2);assert(!r.items[0].cellerator&&r.items[1].cellerator);}
 auto rename=merge_language_features_v1(language_feature_v1::rename,{{"gene","","a.cc",2,6,false}},{{"gene","","a.cell",10,14,true}});assert(rename_is_consistent_v1(rename,"gene"));
 auto dedup=merge_language_features_v1(language_feature_v1::completion,{cpp},{cpp});assert(dedup.items.size()==1);
}
