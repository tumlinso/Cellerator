#include <Cellerator/models/quantize.hh>

#include <torch/torch.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace quant = ::cellerator::models::quantize;

namespace {

void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

quant::GeneQuantizerOutput train_model(
    std::int64_t bits,
    const quant::QuantizerBatch &batch,
    int steps) {
    quant::GeneQuantizerConfig model_config;
    model_config.input_genes = batch.features.size(1);
    model_config.bits = bits;
    model_config.scale_floor = 1.0e-3;
    model_config.init_scale = 1.0;
    model_config.init_offset = 0.0;

    quant::GeneQuantizerModel model(model_config);
    model->to(batch.features.device());
    quant::GeneQuantizerTrainConfig train_config;
    train_config.learning_rate = 5.0e-2;
    train_config.weight_decay = 0.0;
    train_config.loss_scale = 32.0;

    quant::GeneQuantizerLossConfig loss_config;
    loss_config.reconstruction_weight = 1.0;
    loss_config.future_weight = 0.0;
    loss_config.range_weight = 0.01;
    loss_config.min_dynamic_range = bits == 1 ? 0.50 : 1.25;

    auto optimizer = quant::make_gene_quantizer_optimizer(model, train_config);
    for (int step = 0; step < steps; ++step) {
        quant::train_gene_quantizer_step(model, optimizer, batch, loss_config, train_config);
    }

    return quant::evaluate_gene_quantizer_step(model, batch, loss_config).output;
}

} // namespace

int main() {
    require(torch::cuda::is_available(), "quantizeModelTest requires CUDA");
    torch::manual_seed(0);

    const torch::Tensor features = torch::tensor(
        {
            {0.00f, 10.00f},
            {4.00f, 10.25f},
            {8.00f, 10.50f},
            {8.50f, 10.75f},
        },
        torch::TensorOptions().dtype(torch::kFloat32));
    const torch::Tensor cell_indices = torch::tensor(
        {100, 101, 102, 103},
        torch::TensorOptions().dtype(torch::kInt64));

    const quant::QuantizerBatch batch{ features, cell_indices };
    const quant::GeneQuantizerOutput four_bit_output = train_model(4, batch, 250);

    const float scale0 = four_bit_output.scale.index({0}).item<float>();
    const float scale1 = four_bit_output.scale.index({1}).item<float>();
    const float offset0 = four_bit_output.offset.index({0}).item<float>();
    const float offset1 = four_bit_output.offset.index({1}).item<float>();
    require(std::fabs(scale0 - scale1) > 0.10f, "per-gene scales should learn different values");
    require(std::fabs(offset0 - offset1) > 1.00f, "per-gene offsets should learn different values");

    const quant::PackedDenseQuantizedMatrix packed_four_bit = quant::pack_dense_quantized_matrix(
        features,
        four_bit_output.scale,
        four_bit_output.offset,
        4,
        2);
    const torch::Tensor unpacked_four_bit = quant::unpack_dense_quantized_matrix(packed_four_bit);
    require(torch::allclose(
        unpacked_four_bit,
        four_bit_output.reconstruction.to(torch::kCPU),
        1.0e-5,
        1.0e-5), "4-bit quantized pack/unpack should match quantizer reconstruction");

    const quant::GeneQuantizerOutput binary_output = train_model(1, batch, 200);
    require(binary_output.codes.min().item<std::int64_t>() >= 0, "binary codes must be non-negative");
    require(binary_output.codes.max().item<std::int64_t>() <= 1, "binary mode must emit only 0/1 codes");

    const quant::PackedDenseQuantizedMatrix packed_binary = quant::pack_dense_quantized_matrix(
        features,
        binary_output.scale,
        binary_output.offset,
        1,
        2);
    const torch::Tensor unpacked_binary = quant::unpack_dense_quantized_matrix(packed_binary);
    require(torch::allclose(
        unpacked_binary,
        binary_output.reconstruction.to(torch::kCPU),
        1.0e-5,
        1.0e-5), "binary quantized pack/unpack should match quantizer reconstruction");

    const torch::Tensor sparse_cuda = features
        .to(torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA))
        .to_sparse_csr();
    const quant::QuantizerBatch sparse_batch{
        sparse_cuda,
        cell_indices.to(torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA))
    };

    quant::GeneQuantizerConfig sparse_config;
    sparse_config.input_genes = sparse_batch.features.size(1);
    sparse_config.bits = 4;
    sparse_config.scale_floor = 1.0e-3;
    sparse_config.init_scale = 1.0;
    sparse_config.init_offset = 0.0;
    quant::GeneQuantizerModel sparse_model(sparse_config);
    sparse_model->to(torch::kCUDA);

    quant::GeneQuantizerTrainConfig sparse_train;
    sparse_train.learning_rate = 5.0e-2;
    sparse_train.weight_decay = 0.0;
    sparse_train.loss_scale = 32.0;
    sparse_train.clip_gradients = true;

    quant::GeneQuantizerLossConfig sparse_loss_cfg;
    sparse_loss_cfg.reconstruction_weight = 1.0;
    sparse_loss_cfg.future_weight = 0.0;
    sparse_loss_cfg.range_weight = 0.01;
    sparse_loss_cfg.min_dynamic_range = 1.25;

    auto sparse_optimizer = quant::make_gene_quantizer_optimizer(sparse_model, sparse_train);
    const float initial_loss = quant::evaluate_gene_quantizer_step(
        sparse_model,
        sparse_batch,
        sparse_loss_cfg).loss.total.item<float>();
    for (int step = 0; step < 200; ++step) {
        quant::train_gene_quantizer_step(
            sparse_model,
            sparse_optimizer,
            sparse_batch,
            sparse_loss_cfg,
            sparse_train);
    }
    const quant::GeneQuantizerTrainStep sparse_eval = quant::evaluate_gene_quantizer_step(
        sparse_model,
        sparse_batch,
        sparse_loss_cfg);
    const float final_loss = sparse_eval.loss.total.item<float>();
    require(final_loss < initial_loss, "sparse CUDA quantizer training should reduce loss");
    require(sparse_eval.output.reconstruction.is_cuda(), "sparse CUDA quantizer reconstruction should stay on device");
    require(sparse_eval.output.codes.min().item<std::int64_t>() >= 0, "sparse CUDA codes must be non-negative");
    require(sparse_eval.output.codes.max().item<std::int64_t>() <= 15, "sparse CUDA 4-bit codes must stay in range");

    return 0;
}
