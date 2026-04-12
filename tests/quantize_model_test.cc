#include "../src/models/quantize/quantize.hh"

#include <torch/torch.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace quant = ::cellerator::models::quantize;
namespace fn = ::cellerator::compute::neighbors::forward_neighbors;

namespace {

fn::ForwardNeighborRecordBatch make_record_batch(
    const torch::Tensor &cell_indices,
    const torch::Tensor &developmental_time,
    const torch::Tensor &latent_unit,
    const torch::Tensor &embryo_ids) {
    const torch::Tensor ids_cpu = cell_indices.to(torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU)).contiguous();
    const torch::Tensor time_cpu = developmental_time.to(torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)).contiguous();
    const torch::Tensor latent_cpu = latent_unit.to(torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)).contiguous();
    const torch::Tensor embryo_cpu = embryo_ids.to(torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU)).contiguous();

    fn::ForwardNeighborRecordBatch batch;
    batch.latent_dim = latent_cpu.dim() == 2 ? latent_cpu.size(1) : 0;
    batch.cell_indices.resize(static_cast<std::size_t>(ids_cpu.size(0)));
    batch.developmental_time.resize(static_cast<std::size_t>(time_cpu.size(0)));
    batch.latent_unit.resize(static_cast<std::size_t>(latent_cpu.numel()));
    batch.embryo_ids.resize(static_cast<std::size_t>(embryo_cpu.size(0)));

    if (!batch.cell_indices.empty()) {
        std::memcpy(batch.cell_indices.data(), ids_cpu.data_ptr<std::int64_t>(), batch.cell_indices.size() * sizeof(std::int64_t));
        std::memcpy(batch.developmental_time.data(), time_cpu.data_ptr<float>(), batch.developmental_time.size() * sizeof(float));
        std::memcpy(batch.latent_unit.data(), latent_cpu.data_ptr<float>(), batch.latent_unit.size() * sizeof(float));
        std::memcpy(batch.embryo_ids.data(), embryo_cpu.data_ptr<std::int64_t>(), batch.embryo_ids.size() * sizeof(std::int64_t));
    }
    return batch;
}

void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

float rowwise_masked_mse(
    const torch::Tensor &lhs,
    const torch::Tensor &rhs,
    const torch::Tensor &valid_rows) {
    torch::Tensor per_row = torch::pow(lhs.to(torch::kFloat32) - rhs.to(torch::kFloat32), 2).mean(1);
    return per_row.masked_select(valid_rows.to(torch::kBool)).mean().item<float>();
}

fn::ForwardNeighborIndex build_test_index(const torch::Tensor &cell_indices) {
    return fn::build_forward_neighbor_index(make_record_batch(
        cell_indices,
        torch::tensor({0.10f, 0.20f, 0.30f, 0.40f}, torch::TensorOptions().dtype(torch::kFloat32)),
        torch::tensor(
            {
                {1.00f, 0.00f},
                {0.99f, 0.01f},
                {0.98f, 0.02f},
                {0.97f, 0.03f},
            },
            torch::TensorOptions().dtype(torch::kFloat32)),
        torch::tensor({0, 0, 0, 0}, torch::TensorOptions().dtype(torch::kInt64))));
}

quant::QuantizerForwardSupervision build_supervision(
    const fn::ForwardNeighborIndex &index,
    const torch::Tensor &cell_indices,
    const torch::Tensor &features) {
    quant::QuantizerForwardSupervision supervision;
    supervision.neighbor_index = &index;
    supervision.reference_cell_indices = cell_indices;
    supervision.reference_features = features;
    supervision.search_config.backend = fn::ForwardNeighborBackend::exact_windowed;
    supervision.search_config.embryo_policy = fn::ForwardNeighborEmbryoPolicy::same_embryo_only;
    supervision.search_config.top_k = 1;
    supervision.search_config.candidate_k = 1;
    supervision.search_config.time_window.min_delta = 0.0f;
    supervision.search_config.time_window.max_delta = 0.11f;
    return supervision;
}

quant::GeneQuantizerOutput train_model(
    std::int64_t bits,
    const quant::QuantizerBatch &batch,
    const quant::QuantizerForwardSupervision *supervision,
    double future_weight,
    int steps) {
    quant::GeneQuantizerConfig model_config;
    model_config.input_genes = batch.features.size(1);
    model_config.bits = bits;
    model_config.scale_floor = 1.0e-3;
    model_config.init_scale = 1.0;
    model_config.init_offset = 0.0;

    quant::GeneQuantizerModel model(model_config);
    quant::GeneQuantizerTrainConfig train_config;
    train_config.learning_rate = 5.0e-2;
    train_config.weight_decay = 0.0;
    train_config.loss_scale = 32.0;

    quant::GeneQuantizerLossConfig loss_config;
    loss_config.reconstruction_weight = 1.0;
    loss_config.future_weight = future_weight;
    loss_config.range_weight = 0.01;
    loss_config.min_dynamic_range = bits == 1 ? 0.50 : 1.25;

    auto optimizer = quant::make_gene_quantizer_optimizer(model, train_config);
    for (int step = 0; step < steps; ++step) {
        quant::train_gene_quantizer_step(model, optimizer, batch, loss_config, train_config, supervision);
    }

    return quant::evaluate_gene_quantizer_step(model, batch, loss_config, supervision).output;
}

} // namespace

int main() {
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
    const fn::ForwardNeighborIndex index = build_test_index(cell_indices);
    const quant::QuantizerForwardSupervision supervision = build_supervision(index, cell_indices, features);
    const quant::ForwardNeighborTarget future_target =
        quant::build_forward_neighbor_target(batch.cell_indices, supervision);

    require(future_target.valid_rows.index({0}).item<bool>(), "row 0 should have a forward neighbor");
    require(future_target.valid_rows.index({1}).item<bool>(), "row 1 should have a forward neighbor");
    require(future_target.valid_rows.index({2}).item<bool>(), "row 2 should have a forward neighbor");
    require(!future_target.valid_rows.index({3}).item<bool>(), "last row should not have a forward neighbor");
    require(future_target.neighbor_cell_indices.index({0, 0}).item<std::int64_t>() == 101, "row 0 neighbor should come from forward_neighbors");
    require(future_target.neighbor_cell_indices.index({1, 0}).item<std::int64_t>() == 102, "row 1 neighbor should come from forward_neighbors");
    require(future_target.neighbor_cell_indices.index({2, 0}).item<std::int64_t>() == 103, "row 2 neighbor should come from forward_neighbors");

    const quant::GeneQuantizerOutput four_bit_without_future = train_model(4, batch, nullptr, 0.0, 250);
    const quant::GeneQuantizerOutput four_bit_with_future = train_model(4, batch, &supervision, 2.0, 250);

    const float scale0 = four_bit_with_future.scale.index({0}).item<float>();
    const float scale1 = four_bit_with_future.scale.index({1}).item<float>();
    const float offset0 = four_bit_with_future.offset.index({0}).item<float>();
    const float offset1 = four_bit_with_future.offset.index({1}).item<float>();
    require(std::fabs(scale0 - scale1) > 0.10f, "per-gene scales should learn different values");
    require(std::fabs(offset0 - offset1) > 1.00f, "per-gene offsets should learn different values");

    const float future_mse_without = rowwise_masked_mse(
        four_bit_without_future.reconstruction,
        future_target.target_features,
        future_target.valid_rows);
    const float future_mse_with = rowwise_masked_mse(
        four_bit_with_future.reconstruction,
        future_target.target_features,
        future_target.valid_rows);
    require(future_mse_with < future_mse_without, "forward-neighbor supervision should improve future consistency");

    const quant::PackedDenseQuantizedMatrix packed_four_bit = quant::pack_dense_quantized_matrix(
        features,
        four_bit_with_future.scale,
        four_bit_with_future.offset,
        4,
        2);
    const torch::Tensor unpacked_four_bit = quant::unpack_dense_quantized_matrix(packed_four_bit);
    require(torch::allclose(
        unpacked_four_bit,
        four_bit_with_future.reconstruction.to(torch::kCPU),
        1.0e-5,
        1.0e-5), "4-bit quantized pack/unpack should match quantizer reconstruction");

    const quant::GeneQuantizerOutput binary_output = train_model(1, batch, &supervision, 1.0, 200);
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

    return 0;
}
