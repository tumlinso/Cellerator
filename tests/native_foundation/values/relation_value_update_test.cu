#include "update_fixture.cuh"
int main()try{int count=0;cuda_ok(cudaGetDeviceCount(&count));require(count==1,"one leased GPU required");update_test(false);update_test(true,true);update_test(true,false);std::cout<<"V06 real f16/f32 delta and gradient updates, response invalidation and independent branching passed\n";}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
