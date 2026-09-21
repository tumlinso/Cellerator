#include "epoch_fixture.cuh"
int main()try{int count=0;cuda_ok(cudaGetDeviceCount(&count));require(count==1,"one leased GPU required");epoch_test();std::cout<<"V05 safe epoch replacement, unaffected sibling and actual address-reuse stale-ticket rejection passed\n";}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
