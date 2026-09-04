#include <Cellerator/compiler/profile/implement_streaming_profile_builders_v1.hh>
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>
int main(){using namespace cellerator::compiler::profile::v1;constexpr std::uint32_t n=100000;std::vector<double>v(n);for(std::uint32_t i=0;i<n;++i)v[i]=i%100;std::uint64_t hist[64]{},sketch[256]{};double top[16]{};streaming_profile_policy_v1 p{sizeof(hist)+sizeof(sketch)+sizeof(top),64,16,256,false};streaming_profile_builder_v1 b{};initialize_streaming_profile_builder_v1(p,{hist,top,sketch,nullptr,p.memory_budget_bytes},&b);auto begin=std::chrono::steady_clock::now();update_streaming_profile_builder_v1(&b,v.data(),v.size());auto end=std::chrono::steady_clock::now();streaming_profile_result_v1 r{};finalize_streaming_profile_builder_v1(b,&r);double exact=49.5;std::cout<<"observations="<<n<<" build_ns="<<std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count()<<" peak_workspace_bytes="<<r.workspace_bytes<<" mean_absolute_error="<<std::abs(r.mean-exact)<<" sketch_error_bound="<<r.estimator_error_bound<<'\n';}
