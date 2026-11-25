#pragma once
#include "support.hh"
#include <memory>
#include <stdexcept>
#include <vector>

inline void require_status(cellerator::velocity::Status s) {
    if (!s) throw std::runtime_error("cellerator::velocity status " + std::to_string(int(s.code)));
}
struct PreparedStorage {
    std::vector<size_t> indices;
    std::vector<cellerator::velocity::Entry> entries;
    cellerator::velocity::PreparedInput value;
    PreparedStorage(cellerator::velocity::AcquiredMatrix s, cellerator::velocity::AcquiredMatrix u, cellerator::velocity::AcquiredMatrix w,
                    const size_t* genes=nullptr, size_t gene_count=0,
                    const uint8_t* fit_mask=nullptr, const uint8_t* gene_mask=nullptr,
                    ptrdiff_t fit_stride=1, ptrdiff_t gene_stride=1) {
        cellerator::velocity::Input input{s,u,w};
        input.genes=genes; input.gene_count=gene_count; input.fit_mask=fit_mask; input.gene_mask=gene_mask;
        input.fit_mask_stride=fit_stride; input.gene_mask_stride=gene_stride;
        cellerator::velocity::PreparationSize size;
        require_status(cellerator::velocity::preparation_size(input,size));
        indices.resize(size.indices); entries.resize(size.entries);
        require_status(cellerator::velocity::prepare(input,{indices.data(),indices.size(),entries.data(),entries.size()},value));
    }
};
template<class T> struct StochasticStorage {
    std::vector<T> scalars, ig, io, ir, gamma, offset, offset2, first, second, objective;
    std::vector<uint8_t> masks, initial_status, status;
    std::unique_ptr<bool[]> selected;
    cellerator::velocity::Workspace<T> workspace;
    cellerator::velocity::BasicStochasticFitResult<T> result;
    StochasticStorage(const cellerator::velocity::PreparedInput& p, cellerator::velocity::RunOptions options = {}) {
        cellerator::velocity::WorkspaceSize size;
        require_status(cellerator::velocity::workspace_size<T>(p,options,true,size));
        scalars.resize(size.scalars); masks.resize(size.masks);
        require_status(cellerator::velocity::bind_workspace(p,options,true,scalars.data(),scalars.size(),masks.data(),masks.size(),workspace));
        auto n=p.genes;
        ig.resize(n);io.resize(n);ir.resize(n);gamma.resize(n);offset.resize(n);offset2.resize(n);
        first.resize(n);second.resize(n);objective.resize(n);initial_status.resize(n);status.resize(n);
        selected=std::make_unique<bool[]>(n);
        result.initialization={ig.data(),io.data(),ir.data(),selected.get(),initial_status.data(),n};
        result.gamma=gamma.data();result.offset=offset.data();result.offset2=offset2.data();
        result.first_scale=first.data();result.second_scale=second.data();result.objective=objective.data();
        result.status=status.data();result.capacity=n;
    }
    auto& run(const cellerator::velocity::PreparedInput& p, cellerator::velocity::StochasticOptions options = {}) {
        require_status(cellerator::velocity::reference::stochastic(p,workspace,result,options)); return result;
    }
};
