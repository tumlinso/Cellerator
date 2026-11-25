#pragma once

// Development-only caller-owned storage helpers for precision tests and drivers.
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace velocitypp_test {

inline bool prepare(const cellerator::velocity::Input& input, std::vector<std::size_t>& indices,
                    std::vector<cellerator::velocity::Entry>& entries, cellerator::velocity::PreparedInput& prepared) {
    cellerator::velocity::PreparationSize needed;
    if (!cellerator::velocity::preparation_size(input, needed)) return false;
    indices.resize(needed.indices);
    entries.resize(needed.entries);
    return bool(cellerator::velocity::prepare(input,
        {indices.data(), indices.size(), entries.data(), entries.size()}, prepared));
}

template<class T> struct WorkspaceStorage {
    std::vector<T> scalars;
    std::vector<std::uint8_t> mask;
    cellerator::velocity::Workspace<T> workspace;

    bool bind(const cellerator::velocity::PreparedInput& input, std::size_t block_genes,
              bool stochastic, std::size_t workspace_bytes = 1u << 20,
              bool unspliced_second = false) {
        cellerator::velocity::RunOptions options;
        options.workspace_bytes = workspace_bytes;
        options.block_genes = block_genes;
        options.unspliced_second = unspliced_second;
        cellerator::velocity::WorkspaceSize needed;
        if (!cellerator::velocity::workspace_size<T>(input, options, stochastic, needed)) return false;
        scalars.resize(needed.scalars);
        mask.resize(needed.masks);
        return bool(cellerator::velocity::bind_workspace(input, options, stochastic, scalars.data(), scalars.size(),
                                               mask.data(), mask.size(), workspace));
    }
};

template<class T> struct FitStorage {
    std::vector<T> gamma, offset, r2;
    std::unique_ptr<bool[]> selected;
    std::vector<std::uint8_t> status;
    cellerator::velocity::BasicFitResult<T> result;

    explicit FitStorage(std::size_t genes)
        : gamma(genes), offset(genes), r2(genes), selected(new bool[genes]{}), status(genes) {
        result = {gamma.data(), offset.data(), r2.data(), selected.get(), status.data(), genes, false};
    }
};

template<class T> struct StochasticStorage {
    FitStorage<T> initialization;
    std::vector<T> gamma, offset, offset2, first_scale, second_scale, objective;
    std::vector<std::uint8_t> status;
    cellerator::velocity::BasicStochasticFitResult<T> result;

    explicit StochasticStorage(std::size_t genes)
        : initialization(genes), gamma(genes), offset(genes), offset2(genes), first_scale(genes),
          second_scale(genes), objective(genes), status(genes) {
        result = {initialization.result, gamma.data(), offset.data(), offset2.data(), first_scale.data(),
                  second_scale.data(), objective.data(), status.data(), genes, false};
    }
};

} // namespace velocitypp_test
