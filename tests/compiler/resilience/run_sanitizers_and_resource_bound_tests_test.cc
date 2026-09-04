#include <atomic>
#include <cassert>
#include <future>
#include <memory>
#include <vector>
int main(){constexpr std::size_t memory_capacity=1<<20,candidate_limit=256;std::vector<unsigned char>arena(memory_capacity);assert(arena.capacity()<=memory_capacity);std::vector<int>candidates;for(int i=0;i<10000&&candidates.size()<candidate_limit;++i)candidates.push_back(i);assert(candidates.size()==candidate_limit);std::atomic<int>count{0};std::vector<std::future<void>>jobs;for(int i=0;i<8;++i)jobs.push_back(std::async(std::launch::async,[&]{for(int n=0;n<1000;++n)++count;}));for(auto&j:jobs)j.get();assert(count==8000);for(int i=0;i<1000;++i){auto p=std::make_unique<int>(i);assert(*p==i);} }
