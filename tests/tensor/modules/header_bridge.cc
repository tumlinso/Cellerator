#include <Cellerator/tensor/base/compressed.hh>

float read_compressed(cellerator::tensor::base::CompressedView<uint32_t, const float, uint64_t> view) {
    return view.values[view.structure.major_ptr[0]];
}
