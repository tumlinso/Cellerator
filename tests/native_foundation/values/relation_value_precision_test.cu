#include "precision_fixture.cuh"
int main()try{int count=0;cuda_ok(cudaGetDeviceCount(&count));require(count==1,"one leased GPU required");async_test();test(1);test(16);f32_test(1,false);f32_test(1,true);f32_test(16,true);std::cout<<"V04 true f32 values and explicit derived f16 generations; V01/V03 regressions passed\n";}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
