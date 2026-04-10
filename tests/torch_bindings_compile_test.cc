#include "../src/torch/bindings.hh"

#include "../extern/CellShard/src/sharded/sharded_host.cuh"

#include <torch/torch.h>

int main() {
    using cellerator::torch_bindings::ExportOptions;
    using cellerator::torch_bindings::export_as_tensor;

    auto *part_a = new cellshard::sparse::compressed;
    auto *part_b = new cellshard::sparse::compressed;
    cellshard::sharded<cellshard::sparse::compressed> view;

    cellshard::sparse::init(part_a, 2, 4, 3, cellshard::sparse::compressed_by_row);
    if (!cellshard::sparse::allocate(part_a)) return 1;
    part_a->majorPtr[0] = 0;
    part_a->majorPtr[1] = 2;
    part_a->majorPtr[2] = 3;
    part_a->minorIdx[0] = 0;
    part_a->minorIdx[1] = 3;
    part_a->minorIdx[2] = 1;
    part_a->val[0] = __float2half(1.0f);
    part_a->val[1] = __float2half(2.0f);
    part_a->val[2] = __float2half(3.0f);

    cellshard::sparse::init(part_b, 1, 4, 2, cellshard::sparse::compressed_by_row);
    if (!cellshard::sparse::allocate(part_b)) {
        cellshard::destroy(part_a);
        return 1;
    }
    part_b->majorPtr[0] = 0;
    part_b->majorPtr[1] = 2;
    part_b->minorIdx[0] = 0;
    part_b->minorIdx[1] = 2;
    part_b->val[0] = __float2half(4.0f);
    part_b->val[1] = __float2half(5.0f);

    torch::Tensor part_tensor = export_as_tensor(*part_a);
    if (part_tensor.layout() != torch::kSparseCsr) {
        cellshard::destroy(part_a);
        cellshard::destroy(part_b);
        return 1;
    }

    cellshard::init(&view);
    if (!cellshard::append_part(&view, part_a) || !cellshard::append_part(&view, part_b)) {
        cellshard::clear(&view);
        return 1;
    }

    ExportOptions options;
    options.value_dtype = torch::kFloat32;
    torch::Tensor stitched = export_as_tensor(view, options);
    const bool ok = stitched.layout() == torch::kSparseCsr
        && stitched.size(0) == 3
        && stitched.size(1) == 4
        && stitched.values().dtype() == torch::kFloat32;

    cellshard::clear(&view);
    return ok ? 0 : 1;
}
