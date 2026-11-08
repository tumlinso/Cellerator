#pragma once
#include "bridge.hh"

namespace velocity_cuda_bridge {
struct SensitivityOptions { double band = .5, prediction_limit = .1, r2_limit = .05; };
struct SensitivityOutput {
    DenseView state{}, low{}, high{}, delta{}, prediction_change{}, r2_change{};
    DenseView expanded_status{}, contracted_status{}, status_change{}, selection_change{};
};

// report arrays address the full prepared gene axis; individual null views are
// omitted. The stage provides rows*block score scratch and rows mask scratch.
int sensitivity(std::uint64_t rows, std::uint64_t first, std::uint64_t count,
    const std::uint64_t* source_genes, DenseView fit_mask, DenseView gene_mask,
    Fields, Result, Options, SensitivityOptions, SensitivityOutput, bool fp32, void* stream);
}
