#pragma once

#include <Cellerator/core/quantized/layout.cuh>
#include <Cellerator/core/quantized/packing.cuh>

#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace celleratorch::quantize {

namespace msq = ::cellerator::core::quantized;

// Primitive/training wrapper around the lower-level quantized backend.

struct GeneQuantizerConfig {
    std::int64_t input_genes = 0;
    std::int64_t bits = 1;
    double scale_floor = 1.0e-4;
    double init_scale = 1.0;
    double init_offset = 0.0;
};

struct GeneQuantizerLossConfig {
    double reconstruction_weight = 1.0;
    double future_weight = 0.25;
    double range_weight = 0.01;
    double min_dynamic_range = 0.25;
};

struct GeneQuantizerTrainConfig {
    double learning_rate = 1.0e-2;
    double weight_decay = 1.0e-4;
    double loss_scale = 128.0;
    double max_grad_norm = 1.0;
    bool clip_gradients = true;
    bool skip_non_finite_updates = true;
};

struct QuantizerBatch {
    torch::Tensor features;
    torch::Tensor cell_indices;
};

struct GeneQuantizerOutput {
    torch::Tensor reconstruction;
    torch::Tensor codes;
    torch::Tensor scale;
    torch::Tensor offset;
    std::int64_t bits = 0;
};

struct GeneQuantizerLoss {
    torch::Tensor total;
    torch::Tensor reconstruction;
    torch::Tensor future_consistency;
    torch::Tensor range_regularization;
};

struct GeneQuantizerTrainStep {
    GeneQuantizerOutput output;
    GeneQuantizerLoss loss;
};

struct PackedDenseQuantizedMatrix {
    std::int64_t bits = 0;
    int rows = 0;
    int cols = 0;
    int nnz = 0;
    int rows_per_block = 0;
    int block_count = 0;
    std::vector<int> row_ptr;
    std::vector<int> packed_row_ptr;
    std::vector<int> col_idx;
    std::vector<int> block_row_ptr;
    std::vector<float> scale;
    std::vector<float> offset;
    std::vector<unsigned char> packed_values;
};

namespace detail {

inline void validate_bits_(std::int64_t bits) {
    if (bits != 1 && bits != 2 && bits != 4 && bits != 8) {
        throw std::invalid_argument("GeneQuantizerConfig.bits must be 1, 2, 4, or 8");
    }
}

inline std::int64_t max_code_for_bits_(std::int64_t bits) {
    validate_bits_(bits);
    return (static_cast<std::int64_t>(1) << bits) - 1;
}

inline double inverse_softplus_(double value) {
    const double clamped = std::max(value, 1.0e-12);
    return std::log(std::exp(clamped) - 1.0);
}

inline torch::Tensor dense_f32_(const torch::Tensor &tensor) {
    // Real cost: sparse inputs become dense here.
    if (tensor.layout() == torch::kSparse || tensor.layout() == torch::kSparseCsr) {
        return tensor.to_dense().to(torch::kFloat32);
    }
    return tensor.to(torch::kFloat32);
}

inline torch::Tensor contiguous_f32_cpu_(const torch::Tensor &tensor) {
    return tensor.to(torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)).contiguous();
}

template<typename Fn>
inline decltype(auto) with_bits_(std::int64_t bits, Fn &&fn) {
    switch (bits) {
        case 1:
            return std::forward<Fn>(fn)(std::integral_constant<int, 1>{});
        case 2:
            return std::forward<Fn>(fn)(std::integral_constant<int, 2>{});
        case 4:
            return std::forward<Fn>(fn)(std::integral_constant<int, 4>{});
        case 8:
            return std::forward<Fn>(fn)(std::integral_constant<int, 8>{});
        default:
            throw std::invalid_argument("quantizer bit width must be 1, 2, 4, or 8");
    }
}

template<int Bits>
inline PackedDenseQuantizedMatrix pack_dense_quantized_matrix_impl_(
    const torch::Tensor &dense_values,
    const torch::Tensor &scale,
    const torch::Tensor &offset,
    int rows_per_block) {
    // Host-side export helper, not a per-step training path.
    torch::Tensor dense_cpu = contiguous_f32_cpu_(dense_f32_(dense_values));
    torch::Tensor scale_cpu = contiguous_f32_cpu_(scale.view({ -1 }));
    torch::Tensor offset_cpu = contiguous_f32_cpu_(offset.view({ -1 }));
    if (dense_cpu.dim() != 2) throw std::invalid_argument("pack_dense_quantized_matrix requires a rank-2 dense tensor");
    if (scale_cpu.dim() != 1 || offset_cpu.dim() != 1) {
        throw std::invalid_argument("pack_dense_quantized_matrix requires rank-1 scale and offset tensors");
    }

    const int rows = static_cast<int>(dense_cpu.size(0));
    const int cols = static_cast<int>(dense_cpu.size(1));
    if (cols != scale_cpu.size(0) || cols != offset_cpu.size(0)) {
        throw std::invalid_argument("pack_dense_quantized_matrix scale and offset length must match the gene dimension");
    }
    if (rows_per_block <= 0) throw std::invalid_argument("rows_per_block must be > 0");

    PackedDenseQuantizedMatrix packed;
    packed.bits = Bits;
    packed.rows = rows;
    packed.cols = cols;
    packed.nnz = rows * cols;
    packed.rows_per_block = rows_per_block;
    packed.block_count = msq::block_count_for_rows(rows, rows_per_block);
    packed.row_ptr.resize(static_cast<std::size_t>(rows) + 1u);
    packed.packed_row_ptr.resize(static_cast<std::size_t>(rows) + 1u);
    packed.col_idx.resize(static_cast<std::size_t>(packed.nnz));
    packed.block_row_ptr.resize(static_cast<std::size_t>(packed.block_count) + 1u);
    packed.scale.resize(static_cast<std::size_t>(cols));
    packed.offset.resize(static_cast<std::size_t>(cols));

    for (int row = 0; row <= rows; ++row) {
        packed.row_ptr[static_cast<std::size_t>(row)] = row * cols;
    }
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            packed.col_idx[static_cast<std::size_t>(row * cols + col)] = col;
        }
    }
    std::memcpy(
        packed.scale.data(),
        scale_cpu.data_ptr<float>(),
        static_cast<std::size_t>(cols) * sizeof(float));
    std::memcpy(
        packed.offset.data(),
        offset_cpu.data_ptr<float>(),
        static_cast<std::size_t>(cols) * sizeof(float));

    msq::build_packed_row_ptr<Bits>(packed.row_ptr.data(), rows, packed.packed_row_ptr.data());
    msq::build_uniform_block_row_ptr(rows, rows_per_block, packed.block_row_ptr.data());
    packed.packed_values.assign(static_cast<std::size_t>(packed.packed_row_ptr.back()), 0u);

    auto matrix = msq::make_matrix<Bits>(
        packed.rows,
        packed.cols,
        packed.nnz,
        packed.block_count,
        packed.row_ptr.data(),
        packed.packed_row_ptr.data(),
        packed.col_idx.data(),
        packed.block_row_ptr.data(),
        packed.packed_values.data(),
        msq::make_per_gene_affine(packed.scale.data(), packed.offset.data()));

    if (msq::pack_nnz_values(&matrix, dense_cpu.data_ptr<float>()) != 0) {
        throw std::runtime_error("pack_dense_quantized_matrix failed to pack dense values");
    }
    return packed;
}

template<int Bits>
inline torch::Tensor unpack_dense_quantized_matrix_impl_(const PackedDenseQuantizedMatrix &packed) {
    std::vector<float> unpacked(static_cast<std::size_t>(packed.nnz), 0.0f);
    auto matrix = msq::make_matrix<Bits>(
        packed.rows,
        packed.cols,
        packed.nnz,
        packed.block_count,
        packed.row_ptr.data(),
        packed.packed_row_ptr.data(),
        packed.col_idx.data(),
        packed.block_row_ptr.data(),
        const_cast<unsigned char *>(packed.packed_values.data()),
        msq::make_per_gene_affine(packed.scale.data(), packed.offset.data()));

    if (msq::unpack_nnz_values(&matrix, unpacked.data()) != 0) {
        throw std::runtime_error("unpack_dense_quantized_matrix failed to unpack dense values");
    }

    torch::Tensor tensor = torch::empty(
        { packed.rows, packed.cols },
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    if (!unpacked.empty()) {
        std::memcpy(tensor.data_ptr<float>(), unpacked.data(), unpacked.size() * sizeof(float));
    }
    return tensor;
}

} // namespace detail

class GeneQuantizerModelImpl : public torch::nn::Module {
public:
    explicit GeneQuantizerModelImpl(GeneQuantizerConfig config = GeneQuantizerConfig())
        : config_(std::move(config)) {
        if (config_.input_genes <= 0) throw std::invalid_argument("GeneQuantizerConfig.input_genes must be > 0");
        detail::validate_bits_(config_.bits);
        if (config_.scale_floor <= 0.0) throw std::invalid_argument("GeneQuantizerConfig.scale_floor must be > 0");
        if (config_.init_scale <= config_.scale_floor) {
            throw std::invalid_argument("GeneQuantizerConfig.init_scale must be > scale_floor");
        }

        log_scale_ = register_parameter(
            "log_scale",
            torch::empty({ config_.input_genes }, torch::TensorOptions().dtype(torch::kFloat32)));
        offset_ = register_parameter(
            "offset",
            torch::empty({ config_.input_genes }, torch::TensorOptions().dtype(torch::kFloat32)));
        reset_parameters();
    }

    void reset_parameters() {
        torch::NoGradGuard no_grad;
        const float scale_init = static_cast<float>(detail::inverse_softplus_(config_.init_scale - config_.scale_floor));
        log_scale_.fill_(scale_init);
        offset_.fill_(static_cast<float>(config_.init_offset));
    }

    GeneQuantizerOutput forward(const torch::Tensor &features) {
        // Main overhead is chosen at the first line: sparse inputs densify
        // before the quantization math runs.
        torch::Tensor dense = detail::dense_f32_(features);
        if (dense.dim() != 2) throw std::invalid_argument("GeneQuantizerModel expects a rank-2 feature tensor");
        if (dense.size(1) != config_.input_genes) {
            throw std::invalid_argument("GeneQuantizerModel gene dimension does not match config");
        }

        const float max_code = static_cast<float>(detail::max_code_for_bits_(config_.bits));
        torch::Tensor scale = torch::nn::functional::softplus(log_scale_) + config_.scale_floor;
        torch::Tensor offset = offset_.to(torch::kFloat32);
        torch::Tensor centered = dense.to(torch::kFloat32) - offset.unsqueeze(0);
        torch::Tensor relaxed = centered / scale.unsqueeze(0);
        relaxed = torch::clamp(relaxed, 0.0, max_code);
        torch::Tensor rounded = torch::round(relaxed);
        torch::Tensor quantized = is_training() ? relaxed + (rounded - relaxed).detach() : rounded;
        torch::Tensor reconstruction = offset.unsqueeze(0) + quantized * scale.unsqueeze(0);

        return GeneQuantizerOutput{
            std::move(reconstruction),
            rounded.to(torch::kInt64),
            std::move(scale),
            std::move(offset),
            config_.bits
        };
    }

    torch::Tensor resolved_scale() const {
        return torch::nn::functional::softplus(log_scale_) + config_.scale_floor;
    }

    torch::Tensor resolved_offset() const {
        return offset_;
    }

    torch::Tensor log_scale_parameter() const {
        return log_scale_;
    }

    torch::Tensor offset_parameter() const {
        return offset_;
    }

    std::int64_t bits() const {
        return config_.bits;
    }

    double scale_floor() const {
        return config_.scale_floor;
    }

private:
    GeneQuantizerConfig config_;
    torch::Tensor log_scale_;
    torch::Tensor offset_;
};

TORCH_MODULE(GeneQuantizerModel);

namespace detail {

inline bool can_use_sparse_cuda_quantizer_(const torch::Tensor &features) {
    return features.defined()
        && features.is_cuda()
        && features.layout() == torch::kSparseCsr
        && features.dim() == 2;
}

struct SparseCudaLossBundle {
    torch::Tensor reconstruction;
    torch::Tensor range_regularization;
};

SparseCudaLossBundle sparse_cuda_reconstruction_range_backward_(
    GeneQuantizerModel &model,
    const QuantizerBatch &batch,
    const GeneQuantizerLossConfig &config);

} // namespace detail

inline GeneQuantizerLoss compute_gene_quantizer_loss(
    const GeneQuantizerOutput &output,
    const QuantizerBatch &batch,
    const GeneQuantizerLossConfig &config = GeneQuantizerLossConfig()) {
    if (config.reconstruction_weight < 0.0 || config.future_weight < 0.0 || config.range_weight < 0.0) {
        throw std::invalid_argument("GeneQuantizerLossConfig weights must be >= 0");
    }
    if (config.min_dynamic_range < 0.0) {
        throw std::invalid_argument("GeneQuantizerLossConfig.min_dynamic_range must be >= 0");
    }

    torch::Tensor reconstruction = output.reconstruction.to(torch::kFloat32);
    torch::Tensor scalar_zero = torch::zeros({}, torch::TensorOptions().dtype(torch::kFloat32).device(reconstruction.device()));
    torch::Tensor future_loss = scalar_zero.clone();
    torch::Tensor reconstruction_loss = scalar_zero.clone();
    torch::Tensor range_loss = scalar_zero.clone();
    torch::Tensor target;

    const bool need_dense_target = config.reconstruction_weight > 0.0 || config.range_weight > 0.0;
    if (need_dense_target) {
        target = detail::dense_f32_(batch.features).to(reconstruction.device(), torch::kFloat32);
        if (target.dim() != 2 || target.sizes() != reconstruction.sizes()) {
            throw std::invalid_argument("QuantizerBatch.features must densify to the same shape as reconstruction");
        }
    }

    if (config.reconstruction_weight > 0.0) {
        reconstruction_loss = torch::pow(reconstruction - target, 2).mean();
    }

    if (config.range_weight > 0.0) {
        torch::Tensor dynamic_range = output.scale.to(torch::kFloat32)
            * static_cast<float>(std::max<std::int64_t>(1, detail::max_code_for_bits_(output.bits)));
        torch::Tensor dynamic_range_loss = torch::pow(torch::relu(
            torch::full_like(dynamic_range, static_cast<float>(config.min_dynamic_range)) - dynamic_range), 2).mean();
        torch::Tensor gene_floor = std::get<0>(target.min(0, false));
        torch::Tensor offset_anchor_loss = torch::pow(output.offset.to(torch::kFloat32) - gene_floor, 2).mean();
        range_loss = dynamic_range_loss + offset_anchor_loss;
    }

    torch::Tensor total = config.reconstruction_weight * reconstruction_loss
        + config.future_weight * future_loss
        + config.range_weight * range_loss;
    return GeneQuantizerLoss{
        std::move(total),
        std::move(reconstruction_loss),
        std::move(future_loss),
        std::move(range_loss)
    };
}

inline torch::optim::AdamW make_gene_quantizer_optimizer(
    GeneQuantizerModel &model,
    const GeneQuantizerTrainConfig &config = GeneQuantizerTrainConfig()) {
    return torch::optim::AdamW(
        model->parameters(),
        torch::optim::AdamWOptions(config.learning_rate).weight_decay(config.weight_decay));
}

inline GeneQuantizerTrainStep train_gene_quantizer_step(
    GeneQuantizerModel &model,
    torch::optim::Optimizer &optimizer,
    const QuantizerBatch &batch,
    const GeneQuantizerLossConfig &loss_config = GeneQuantizerLossConfig(),
    const GeneQuantizerTrainConfig &train_config = GeneQuantizerTrainConfig()) {
    if (train_config.loss_scale <= 0.0) throw std::invalid_argument("loss_scale must be > 0");

    model->train();
    optimizer.zero_grad();

    if (detail::can_use_sparse_cuda_quantizer_(batch.features)) {
        detail::SparseCudaLossBundle sparse_loss =
            detail::sparse_cuda_reconstruction_range_backward_(model, batch, loss_config);
        torch::Tensor future_loss = torch::zeros(
            {},
            torch::TensorOptions().dtype(torch::kFloat32).device(batch.features.device()));

        if (train_config.skip_non_finite_updates) {
            for (const torch::Tensor &param : model->parameters()) {
                if (param.grad().defined() && !torch::isfinite(param.grad()).all().item<bool>()) {
                    optimizer.zero_grad();
                    torch::NoGradGuard no_grad;
                    GeneQuantizerOutput output = model->forward(batch.features);
                    torch::Tensor total = loss_config.reconstruction_weight * sparse_loss.reconstruction
                        + loss_config.future_weight * future_loss
                        + loss_config.range_weight * sparse_loss.range_regularization;
                    return GeneQuantizerTrainStep{
                        std::move(output),
                        GeneQuantizerLoss{
                            std::move(total),
                            std::move(sparse_loss.reconstruction),
                            std::move(future_loss),
                            std::move(sparse_loss.range_regularization)
                        }
                    };
                }
            }
        }

        if (train_config.clip_gradients) {
            torch::nn::utils::clip_grad_norm_(model->parameters(), train_config.max_grad_norm);
        }
        optimizer.step();

        torch::NoGradGuard no_grad;
        GeneQuantizerOutput output = model->forward(batch.features);
        torch::Tensor total = loss_config.reconstruction_weight * sparse_loss.reconstruction
            + loss_config.future_weight * future_loss
            + loss_config.range_weight * sparse_loss.range_regularization;
        return GeneQuantizerTrainStep{
            std::move(output),
            GeneQuantizerLoss{
                std::move(total),
                std::move(sparse_loss.reconstruction),
                std::move(future_loss),
                std::move(sparse_loss.range_regularization)
            }
        };
    }

    // Extra step overhead comes from manual unscale, optional finite-gradient
    // checks, and optional full-parameter gradient clipping.
    GeneQuantizerOutput output = model->forward(batch.features);
    GeneQuantizerLoss loss = compute_gene_quantizer_loss(output, batch, loss_config);
    (loss.total * train_config.loss_scale).backward();

    for (torch::Tensor &param : model->parameters()) {
        if (param.grad().defined()) param.grad().div_(train_config.loss_scale);
    }
    if (train_config.skip_non_finite_updates) {
        for (const torch::Tensor &param : model->parameters()) {
            if (param.grad().defined() && !torch::isfinite(param.grad()).all().item<bool>()) {
                optimizer.zero_grad();
                return GeneQuantizerTrainStep{ std::move(output), std::move(loss) };
            }
        }
    }
    if (train_config.clip_gradients) {
        torch::nn::utils::clip_grad_norm_(model->parameters(), train_config.max_grad_norm);
    }
    optimizer.step();
    return GeneQuantizerTrainStep{ std::move(output), std::move(loss) };
}

inline GeneQuantizerTrainStep evaluate_gene_quantizer_step(
    GeneQuantizerModel &model,
    const QuantizerBatch &batch,
    const GeneQuantizerLossConfig &loss_config = GeneQuantizerLossConfig()) {
    torch::NoGradGuard no_grad;
    model->eval();

    GeneQuantizerOutput output = model->forward(batch.features);
    GeneQuantizerLoss loss = compute_gene_quantizer_loss(output, batch, loss_config);
    return GeneQuantizerTrainStep{ std::move(output), std::move(loss) };
}

inline PackedDenseQuantizedMatrix pack_dense_quantized_matrix(
    const torch::Tensor &dense_values,
    const torch::Tensor &scale,
    const torch::Tensor &offset,
    std::int64_t bits,
    int rows_per_block = 32) {
    return detail::with_bits_(bits, [&](auto bit_tag) {
        return detail::pack_dense_quantized_matrix_impl_<decltype(bit_tag)::value>(
            dense_values,
            scale,
            offset,
            rows_per_block);
    });
}

inline torch::Tensor unpack_dense_quantized_matrix(const PackedDenseQuantizedMatrix &packed) {
    return detail::with_bits_(packed.bits, [&](auto bit_tag) {
        return detail::unpack_dense_quantized_matrix_impl_<decltype(bit_tag)::value>(packed);
    });
}

} // namespace celleratorch::quantize
