#include <Cellerator/compiler/backend/nvcc/deliver_the_first_nvcc_object_milestone_v1.hh>
#include <Cellerator/compiler/backend/nvcc/generate_custom_relation_kernels_where_selected_v1.hh>

#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

int main() {
    using namespace cellerator::compiler::backend::nvcc::v1;

    const auto kernel = generate_custom_relation_kernel(
        {"profile_relation", 32, 48, 4, 9, 16, true, false, false, false});
    assert(kernel);
    const auto milestone = make_first_nvcc_object_milestone(
        {21, 22, 23, 70, *kernel});
    assert(milestone);
    assert(milestone->conventional_fallback_retained);

    const auto directory = std::filesystem::current_path() / "f03_013_artifacts";
    std::filesystem::create_directories(directory);
    const auto source = directory / "relation.cu";
    const auto object = directory / "relation.o";
    const auto executable = directory / "relation_test";
    std::ofstream output(source);
    output << "#include <cuda_runtime.h>\n#include <cmath>\n#include <vector>\n"
           << milestone->cuda_source
           << "int main(){std::vector<float> h(48),o(32);"
              "for(unsigned i=0;i<h.size();++i)h[i]=float(i)+0.25f;"
              "float *d_in=nullptr,*d_out=nullptr;"
              "if(cudaMalloc(&d_in,h.size()*sizeof(float))!=cudaSuccess)return 2;"
              "if(cudaMalloc(&d_out,o.size()*sizeof(float))!=cudaSuccess)return 3;"
              "cudaMemcpy(d_in,h.data(),h.size()*sizeof(float),cudaMemcpyHostToDevice);"
              "profile_relation<<<1,32>>>(d_in,d_out);"
              "if(cudaDeviceSynchronize()!=cudaSuccess)return 4;"
              "cudaMemcpy(o.data(),d_out,o.size()*sizeof(float),cudaMemcpyDeviceToHost);"
              "for(unsigned i=0;i<o.size();++i)if(o[i]!=h[i%48])return 5;"
              "cudaFree(d_out);cudaFree(d_in);return 0;}\n";
    output.close();

    const std::string compile = "/usr/bin/nvcc -std=c++17 -arch=sm_70 -c " +
        source.string() + " -o " + object.string();
    const std::string link = "/usr/bin/nvcc -arch=sm_70 " + object.string() +
        " -o " + executable.string();
    assert(std::system(compile.c_str()) == 0);
    assert(std::system(link.c_str()) == 0);
    assert(std::system(executable.c_str()) == 0);
}
