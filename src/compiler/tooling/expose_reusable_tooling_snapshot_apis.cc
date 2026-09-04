#include <Cellerator/compiler/tooling/expose_reusable_tooling_snapshot_apis_v1.hh>
#include <algorithm>
namespace Cellerator::compiler::tooling {
tooling_snapshot_v1::tooling_snapshot_v1(tooling_snapshot_data_v1 d):data_(std::make_shared<const tooling_snapshot_data_v1>(std::move(d))){}
std::uint64_t tooling_snapshot_v1::revision()const{return data_->revision;}
std::optional<tooling_symbol_v1> tooling_snapshot_v1::symbol_at(std::uint64_t o)const{auto i=std::find_if(data_->symbols.begin(),data_->symbols.end(),[&](const auto&s){return o>=s.begin&&o<s.end;});return i==data_->symbols.end()?std::nullopt:std::optional<tooling_symbol_v1>(*i);}
const std::vector<std::string>&tooling_snapshot_v1::diagnostics()const{return data_->diagnostics;}
tooling_cancellation_v1::tooling_cancellation_v1():state_(std::make_shared<std::atomic_bool>(false)){}void tooling_cancellation_v1::cancel()const{state_->store(true);}bool tooling_cancellation_v1::cancelled()const{return state_->load();}
void request_background_compile_v1(const background_compile_hook_v1&h,std::string s,tooling_cancellation_v1 c){if(h&&!c.cancelled())h(std::move(s),c);}
} // namespace Cellerator::compiler::tooling
