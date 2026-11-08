#include <anndata/device_abi.h>

extern "C" std::uint64_t bio_device_rows(anndata::device::DenseView view) {
    return view.rows;
}

__global__ void bio_device_row_kernel(anndata::device::DenseView view, std::uint64_t* out) {
    if (!threadIdx.x && !blockIdx.x) *out = view.rows;
}
