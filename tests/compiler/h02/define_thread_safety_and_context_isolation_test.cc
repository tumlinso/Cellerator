#include <Cellerator/sdk/define_thread_safety_and_context_isolation_v1.hh>
#include <atomic>
#include <cassert>
#include <thread>
#include <vector>
namespace api=cellerator::compiler::api::v1;
int main(){
 auto registry=std::make_shared<const api::immutable_registry_v1>(api::immutable_registry_v1::entries_type{{"cell","builtin"}});
 api::context_builder_v1 builder(registry);builder.set("target","cpu");auto a=builder.finish();
 api::isolated_context_v1 b(registry);b.set("target","cuda");
 std::atomic<bool> ok{true};std::vector<std::thread> readers;
 for(int i=0;i<8;++i)readers.emplace_back([&]{for(int n=0;n<1000;++n)if(a->get("target")!="cpu"||b.get("target")!="cuda"||*registry->find("cell")!="builtin")ok=false;});
 for(auto& t:readers)t.join();assert(ok);assert(a->get("target")!=b.get("target"));
 api::backend_process_isolation_v1 backend{42,false};assert(backend.process_id==42&&!backend.shares_mutable_compiler_state);
}
