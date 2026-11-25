#include <stddef.h>
#include <stdint.h>
#include <Cellerator/memory/view.hh>

import cellerator.preprocess;

#include "modeling.hh"

int main() {
    using namespace cellerator::memory;
    const float values[] = {1, 3, 3, 1};
    const const_matrix_view<float> counts{values, 2, 2, 2, {domain::host}};
    return ccell_example::normalized_readout(counts) ? 0 : 1;
}
