#include <Cellerator/compiler/backend/compile_generated_c_into_ordinary_objects_v1.hh>
#include <dlfcn.h>
#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <spawn.h>
#include <string>
#include <sys/wait.h>
#include <thread>
#include <vector>

extern char** environ;
namespace fs=std::filesystem;
using namespace cellerator::compiler::backend::v1;
using region_fn=float(*)(float,void(*)(int)) noexcept;

static std::uint64_t hash(std::string_view s){std::uint64_t h=1469598103934665603ULL;for(unsigned char c:s){h^=c;h*=1099511628211ULL;}return h;}
static bool spawn_wait(std::vector<std::string>&args){
  std::vector<char*>v;for(auto&s:args)v.push_back(s.data());v.push_back(nullptr);
  pid_t p{};if(posix_spawnp(&p,args[0].c_str(),nullptr,nullptr,v.data(),environ)!=0)return false;
  int st{};return waitpid(p,&st,0)>=0&&WIFEXITED(st)&&WEXITSTATUS(st)==0;
}
struct Module {
  void* handle=nullptr;region_fn call=nullptr;std::string path;std::atomic<unsigned>*closed=nullptr;
  ~Module(){if(handle){dlclose(handle);if(closed)++*closed;}}
};
using Lease=std::shared_ptr<Module>;

static Lease compile_and_load(const fs::path&dir,std::uint64_t id,float factor,
                              bool inject_failure,std::atomic<unsigned>*closed){
  fs::create_directories(dir);
  const std::string body=inject_failure?"this is intentionally invalid C++":
    "extern \"C\" float cellerator_region(float x, void(*hook)(int)) noexcept { if(hook){hook(0);hook(1);} return x*"+std::to_string(factor)+"f; }\n";
  const auto key=hash(body+"|e6-cpu-generated-module-v1");const auto stem=std::to_string(id)+"-"+std::to_string(key);
  const auto src=dir/(stem+".cc"),obj=dir/(stem+".o"),dep=dir/(stem+".d"),so=dir/(stem+".so");
  {std::ofstream f(src);f<<body;}
  compile_generated_cpp_request_v1 req{"g++",src.string(),obj.string(),dep.string(),dir.string(),{"-fPIC"},{},{}};
  compile_generated_cpp_receipt_v1 receipt{};
  if(compile_generated_cpp_object_v1(req,&receipt)!=compile_object_status_v1::success)return {};
  std::vector<std::string> args{"g++","-shared",obj.string(),"-o",so.string()};
  if(!spawn_wait(args))return {};
  void*h=dlopen(so.c_str(),RTLD_NOW|RTLD_LOCAL);if(!h)return {};
  auto fn=reinterpret_cast<region_fn>(dlsym(h,"cellerator_region"));if(!fn){dlclose(h);return {};}
  auto m=std::make_shared<Module>();m->handle=h;m->call=fn;m->path=so.string();m->closed=closed;return m;
}

struct Request { std::uint64_t id=0,epoch=0,value_generation=0;float factor=1;bool fail=false;std::string key; };
struct Completion { Request request;Lease module;bool failed=false,cancelled=false; };
class Worker {
 public:
  explicit Worker(fs::path dir,std::atomic<unsigned>*closed):dir_(std::move(dir)),closed_(closed),thread_([this]{run();}){}
  ~Worker(){ {std::lock_guard l(mu_);stop_=true;release_gate_=true;}cv_.notify_all();thread_.join(); }
  struct Enqueue {std::uint64_t id=0;bool accepted=false,duplicate=false,full=false;};
  Enqueue submit(Request r){
    std::lock_guard l(mu_);
    if(auto it=by_key_.find(r.key);it!=by_key_.end())return {it->second,true,true,false};
    if(pending_){return {0,false,false,true};}
    r.id=++next_id_;by_key_[r.key]=r.id;
    if(busy_){pending_=r;}else pending_=r;
    ++submitted_;cv_.notify_all();return {r.id,true,false,false};
  }
  bool cancel(std::uint64_t id){
    std::lock_guard l(mu_);if(pending_&&pending_->id==id){
      completed_[id]=Completion{*pending_,{},false,true};pending_.reset();++cancelled_;cv_.notify_all();return true;}return false;
  }
  void hold_next(){std::lock_guard l(mu_);hold_next_=true;}
  void release(){std::lock_guard l(mu_);release_gate_=true;cv_.notify_all();}
  void wait_started(std::uint64_t id){std::unique_lock l(mu_);cv_.wait(l,[&]{return active_id_==id||completed_.contains(id);});}
  Completion wait(std::uint64_t id){std::unique_lock l(mu_);cv_.wait(l,[&]{return completed_.contains(id);});auto c=completed_.at(id);completed_.erase(id);if(auto it=by_key_.find(c.request.key);it!=by_key_.end()&&it->second==id)by_key_.erase(it);return c;}
  unsigned submitted()const{return submitted_;}unsigned cancelled()const{return cancelled_;}
 private:
  void run(){
    std::unique_lock l(mu_);
    while(true){cv_.wait(l,[&]{return stop_||pending_.has_value();});if(stop_)return;
      Request r=*pending_;pending_.reset();busy_=true;active_id_=r.id;const bool should_hold=hold_next_;hold_next_=false;cv_.notify_all();
      if(should_hold){cv_.wait(l,[&]{return release_gate_||stop_;});release_gate_=false;}
      if(stop_)return;
      l.unlock();auto mod=compile_and_load(dir_,r.id,r.factor,r.fail,closed_);l.lock();
      completed_[r.id]=Completion{r,mod,!mod,false};busy_=false;active_id_=0;cv_.notify_all();
    }
  }
  fs::path dir_;std::atomic<unsigned>*closed_=nullptr;mutable std::mutex mu_;std::condition_variable cv_;
  bool stop_=false,busy_=false,hold_next_=false,release_gate_=false;
  std::optional<Request> pending_;std::uint64_t next_id_=0,active_id_=0;
  std::map<std::string,std::uint64_t> by_key_;std::map<std::uint64_t,Completion> completed_;
  unsigned submitted_=0,cancelled_=0;
  std::thread thread_;
};

struct Publication {
  Lease active;std::uint64_t structure_epoch=0,value_generation=0;unsigned accepted=0,rejected_stale=0,rejected_failure=0;
  bool publish(const Completion&c){
    if(c.cancelled||c.failed||!c.module){++rejected_failure;return false;}
    if(c.request.epoch!=structure_epoch){++rejected_stale;return false;}
    active=c.module;++accepted;return true;
  }
  float call(float value)const{return active?active->call(value,nullptr):value;}
};

struct CallGate {std::mutex mu;std::condition_variable cv;bool entered=false,release=false,exited=false;};
static CallGate* gate_ptr=nullptr;
static void call_hook(int phase){auto*g=gate_ptr;std::unique_lock l(g->mu);if(phase==0){g->entered=true;g->cv.notify_all();g->cv.wait(l,[&]{return g->release;});}else{g->exited=true;g->cv.notify_all();}}

int main(int argc,char**argv){
  if(argc!=2)return 2;
  const fs::path root=argv[1];fs::create_directories(root);
  std::atomic<unsigned> closed{0};
  auto generic=compile_and_load(root/"initial",0,1.0f,false,&closed);if(!generic)return 3;
  Publication pub;pub.structure_epoch=7;pub.value_generation=4;pub.active=generic;generic.reset();
  Worker worker(root/"worker",&closed);

  // Old executable is executing from a real .so while its lease is captured.
  CallGate gate;gate_ptr=&gate;std::weak_ptr<Module> old_weak=pub.active;
  float old_result=0;auto old_lease=pub.active;
  std::thread in_flight([lease=old_lease,&old_result]{old_result=lease->call(3.0f,&call_hook);});
  {std::unique_lock l(gate.mu);gate.cv.wait(l,[&]{return gate.entered;});}
  old_lease.reset();

  // Compilation is deliberately blocked after dequeue. Duplicate request
  // coalesces, one distinct item fills the pending slot, and a third rejects.
  worker.hold_next();
  Request replacement{0,7,4,2.0f,false,"structure7-contractA"};
  auto first=worker.submit(replacement);assert(first.accepted&&!first.duplicate);
  worker.wait_started(first.id);
  auto duplicate=worker.submit(replacement);assert(duplicate.accepted&&duplicate.duplicate&&duplicate.id==first.id);
  Request queued{0,7,4,3.0f,false,"structure7-contractB"};
  auto second=worker.submit(queued);assert(second.accepted);
  Request overflow{0,7,4,4.0f,false,"structure7-contractC"};
  auto full=worker.submit(overflow);assert(!full.accepted&&full.full);
  assert(worker.cancel(second.id));
  assert(worker.wait(second.id).cancelled);
  auto cancelled_retry=worker.submit(queued);assert(cancelled_retry.accepted&&!cancelled_retry.duplicate&&cancelled_retry.id!=second.id);
  // Numerical values may advance while compilation runs; structure/contract
  // remain compatible and the module accepts values at call time.
  ++pub.value_generation;
  worker.release();auto done=worker.wait(first.id);assert(!done.failed);
  auto success_retry=worker.submit(replacement);assert(success_retry.accepted&&!success_retry.duplicate&&success_retry.id!=first.id);
  auto cancelled_retry_result=worker.wait(cancelled_retry.id);assert(!cancelled_retry_result.failed);
  auto success_retry_result=worker.wait(success_retry.id);assert(!success_retry_result.failed);
  assert(pub.publish(done));
  assert(pub.call(3.0f)==6.0f);
  assert(!old_weak.expired()); // the old call still pins code after publication
  {std::lock_guard l(gate.mu);gate.release=true;}gate.cv.notify_all();in_flight.join();
  assert(old_result==3.0f&&old_weak.expired());

  // A genuinely compiled replacement for old epoch becomes stale before
  // publication. The generic-compatible active artifact remains available.
  worker.hold_next();Request stale{0,7,pub.value_generation,5.0f,false,"structure7-stale"};
  auto stale_req=worker.submit(stale);assert(stale_req.accepted);worker.wait_started(stale_req.id);
  ++pub.structure_epoch;pub.active.reset();worker.release();auto stale_done=worker.wait(stale_req.id);
  assert(!stale_done.failed&&!pub.publish(stale_done));
  stale_done.module.reset();
  assert(pub.call(3.0f)==3.0f);

  // Compilation failure and cancellation preserve the current executable.
  worker.hold_next();Request cancel{0,pub.structure_epoch,pub.value_generation,9.0f,false,"queued-cancel"};
  auto cancel_req=worker.submit(cancel);assert(cancel_req.accepted);worker.wait_started(cancel_req.id);
  Request cancelled_pending{0,pub.structure_epoch,pub.value_generation,8.0f,false,"pending-cancel"};
  auto pending=worker.submit(cancelled_pending);assert(pending.accepted);assert(worker.cancel(pending.id));assert(worker.wait(pending.id).cancelled);
  worker.release();(void)worker.wait(cancel_req.id);
  Request failure{0,pub.structure_epoch,pub.value_generation,1.0f,true,"compile-failure"};
  auto fail_req=worker.submit(failure);assert(fail_req.accepted);auto failed=worker.wait(fail_req.id);
  assert(failed.failed&&!pub.publish(failed));
  auto failure_retry=worker.submit(failure);assert(failure_retry.accepted&&!failure_retry.duplicate&&failure_retry.id!=fail_req.id);
  auto failure_retry_result=worker.wait(failure_retry.id);assert(failure_retry_result.failed);
  assert(pub.call(3.0f)==3.0f);

  std::cout<<"old_call_result="<<old_result<<" old_code_retired_after_completion=1\n"
    <<"replacement_published="<<pub.accepted<<" stale_rejected="<<pub.rejected_stale
    <<" failed_rejected="<<pub.rejected_failure<<" value_generation_now="<<pub.value_generation
    <<" structural_epoch_now="<<pub.structure_epoch<<" duplicate_coalesced=1 pending_capacity_rejected=1"
    <<" pending_cancelled="<<worker.cancelled()<<" retry_after_success_cancel_failure=pass generic_fallback_callable=1"
    <<" module_close_count="<<closed.load()<<"\n";
}
