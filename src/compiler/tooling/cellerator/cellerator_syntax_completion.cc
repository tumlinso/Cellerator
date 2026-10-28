#include "tooling_model.hh"

#include <array>

namespace cellerator::compiler::tooling::v1 {
namespace {
constexpr std::array<std::pair<std::string_view,std::string_view>,16> vocabulary{{
 {"domain","declaration"},{"axis","declaration"},{"relation","declaration"},{"field","declaration"},
 {"operation","declaration"},{"effect","semantic"},{"persistent","storage"},{"profile","profile"},
 {"reflect","reflection"},{"pass","extension"},{"ceir","inline-ir"},{"native","native-block"},
 {"reads","effect"},{"writes","effect"},{"support","relation"},{"orientation","relation"}}};
}
std::vector<completion_item> complete_cellerator_syntax(std::string_view source,std::size_t cursor){
 cursor=cursor>source.size()?source.size():cursor;std::size_t begin=cursor;
 while(begin&&((source[begin-1]>='a'&&source[begin-1]<='z')||source[begin-1]=='_'))--begin;
 const auto prefix=source.substr(begin,cursor-begin);std::vector<completion_item> out;
 for(auto [word,kind]:vocabulary)if(word.substr(0,prefix.size())==prefix)out.push_back({std::string(word),std::string(kind)});
 return out;
}
}
