#include <stdint.h>

import cellerator.tensor;

namespace base = cellerator::tensor::base;
namespace bm = cellerator::tensor::block_masked;

float read_compressed(base::CompressedView<uint32_t, const float, uint64_t> view);
#if defined(CELLERATOR_TENSOR_MODULE_TEST_CUDA)
float read_block_masked(bm::View<const float> view);
#endif

int main() {
    const uint32_t rows[]{0}, cols[]{1};
    const uint64_t major[]{0, 1};
    const float values[]{3.0f};
    const base::CompressedStructureView<uint32_t, uint64_t> csr{
        1, 2, 1, base::CompressedAxis::row, major, cols};
    const base::CooView<uint32_t, const float> coo{1, 2, 1, rows, cols, values};
    if (read_compressed({csr, values}) != 3.0f || coo.col_idx[0] != 1 ||
        coo.values[0] != 3.0f) return 1;

    const uint32_t offsets[]{0, 1}, ids[]{0}, row_masks[]{1}, feature_masks[]{2};
    const uint32_t feature_blocks[]{0, 2};
    const bm::LayoutView<> layout{
        offsets, ids, row_masks, offsets, feature_masks, offsets, feature_blocks,
        32, 1, 2, 1, 1, 1, 1, 1};
    const bm::View<const float> blocked{layout, values};
    const bm::LoadResult<float> stored = bm::try_load(blocked, 0, 1);
    if (stored.status != bm::LoadStatus::stored || stored.value != 3.0f ||
        bm::try_load(blocked, 0, 0).status != bm::LoadStatus::absent) return 2;
    uint64_t offset = 1;
    if (!bm::value_offset(layout, 0, 1, &offset) || offset != 0) return 3;
#if defined(CELLERATOR_TENSOR_MODULE_TEST_CUDA)
    if (read_block_masked(blocked) != 3.0f) return 4;
#endif
    return 0;
}
