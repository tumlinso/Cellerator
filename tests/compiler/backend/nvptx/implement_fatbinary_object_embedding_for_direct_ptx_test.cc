#include <Cellerator/compiler/backend/nvptx/implement_fatbinary_object_embedding_for_direct_ptx_v1.hh>

#include <cassert>
#include <iostream>

using namespace Cellerator::compiler::backend::nvptx;

extern "C" const unsigned char _binary_embedded_minimal_kernel_cubin_start[];
extern "C" const unsigned char _binary_embedded_minimal_kernel_cubin_end[];

int main() {
    const auto plan = make_cuda_object_embedding_plan_v1(
        "/usr/bin/objcopy", "embedded_minimal_kernel.cubin", "embedded_minimal_kernel.o",
        embedded_cuda_image_kind_v1::cubin);
    assert(plan.executable == "/usr/bin/objcopy" && plan.section_name == ".nv_cellerator_cubin" &&
           plan.start_symbol == "_binary_embedded_minimal_kernel_cubin_start");

    embedded_cuda_image_v1 image;
    image.begin = _binary_embedded_minimal_kernel_cubin_start;
    image.end = _binary_embedded_minimal_kernel_cubin_end;
    image.kind = embedded_cuda_image_kind_v1::cubin;
    image.kernel_symbol = "embedded_minimal_kernel";
    const auto launched = launch_embedded_cuda_image_v1(image, {});
    assert(launched && launched.image_bytes > 0u);

    std::cout << "plain C++ linked and launched embedded cubin bytes="
              << launched.image_bytes << '\n';
}
