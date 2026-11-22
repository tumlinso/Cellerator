#pragma once

namespace cellerator::tensor::base {

// Each row block has ell_cols / block_size block slots. An empty slot
// has this column index. Values use ell_cols scalar slots per logical row.
enum { blocked_ell_invalid_col = 0xffffffffu };

} // namespace cellerator::tensor::base
