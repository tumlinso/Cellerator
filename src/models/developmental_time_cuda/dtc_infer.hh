#pragma once

#include "dtc_model.hh"

namespace cellerator::models::developmental_time_cuda {

inline torch::Tensor infer_time(
    DevelopmentalTimeCudaModel &model,
    const torch::Tensor &sparse_csr_batch) {
    return predict_time(model, sparse_csr_batch);
}

} // namespace cellerator::models::developmental_time_cuda
