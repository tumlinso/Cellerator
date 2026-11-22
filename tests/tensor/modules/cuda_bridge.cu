#include <Cellerator/tensor/block_masked/view.hh>

float read_block_masked(cellerator::tensor::block_masked::View<const float> view) {
    return cellerator::tensor::block_masked::try_load(view, 0, 1).value;
}
