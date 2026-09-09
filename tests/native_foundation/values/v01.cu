#include "fixture.cuh"
int main()try{
    int count=0;cuda_ok(cudaGetDeviceCount(&count));require(count==1,"exactly one leased visible device required");
    cudaDeviceProp properties{};cuda_ok(cudaGetDeviceProperties(&properties,0));require(properties.major==7 && properties.minor==0,"real sm70 device required");
    test(1);test(16);
    std::cout<<"{\"task\":\"CE-NF1-V01\",\"actual_cuda\":true,\"shared_structure\":true,\"independent_instances\":2,\"widths\":[1,16]}\n";
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
