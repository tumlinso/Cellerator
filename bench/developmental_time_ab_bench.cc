#include <Cellerator/models/developmental_time.hh>
#include <Cellerator/models/developmental_time_cuda.hh>
#include "benchmark_mutex.hh"
#include "cellerator_cuda_mode.hh"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace dt = ::cellerator::models::developmental_time;
namespace dtc = ::cellerator::models::developmental_time_cuda;

namespace {

struct bench_config {
    std::string impl = "baseline";
    std::string mode = "train";
    std::string scenario_id = "small";
    std::int64_t rows = 2048;
    std::int64_t cols = 4096;
    std::int64_t nnz_per_row = 32;
    std::int64_t stem_dim = 128;
    std::int64_t hidden_dim = 64;
    std::int64_t time_bins = 8;
    std::int64_t warmup = 2;
    std::int64_t iters = 10;
    std::string out_dir;
};

void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

std::int64_t parse_i64(const char *value, const char *label) {
    char *end = nullptr;
    const long long parsed = std::strtoll(value, &end, 10);
    if (value == nullptr || *value == '\0' || end == nullptr || *end != '\0') {
        throw std::invalid_argument(std::string("invalid integer for ") + label);
    }
    return static_cast<std::int64_t>(parsed);
}

bench_config parse_args(int argc, char **argv) {
    bench_config cfg;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        auto require_value = [&](const char *label) -> const char * {
            if (i + 1 >= argc) throw std::invalid_argument(std::string("missing value for ") + label);
            return argv[++i];
        };
        if (arg == "--impl") {
            cfg.impl = require_value("--impl");
        } else if (arg == "--mode") {
            cfg.mode = require_value("--mode");
        } else if (arg == "--scenario-id") {
            cfg.scenario_id = require_value("--scenario-id");
        } else if (arg == "--rows") {
            cfg.rows = parse_i64(require_value("--rows"), "--rows");
        } else if (arg == "--cols") {
            cfg.cols = parse_i64(require_value("--cols"), "--cols");
        } else if (arg == "--nnz-row") {
            cfg.nnz_per_row = parse_i64(require_value("--nnz-row"), "--nnz-row");
        } else if (arg == "--stem-dim") {
            cfg.stem_dim = parse_i64(require_value("--stem-dim"), "--stem-dim");
        } else if (arg == "--hidden-dim") {
            cfg.hidden_dim = parse_i64(require_value("--hidden-dim"), "--hidden-dim");
        } else if (arg == "--time-bins") {
            cfg.time_bins = parse_i64(require_value("--time-bins"), "--time-bins");
        } else if (arg == "--warmup") {
            cfg.warmup = parse_i64(require_value("--warmup"), "--warmup");
        } else if (arg == "--iters") {
            cfg.iters = parse_i64(require_value("--iters"), "--iters");
        } else if (arg == "--out-dir") {
            cfg.out_dir = require_value("--out-dir");
        } else if (arg == "-h" || arg == "--help") {
            std::cout
                << "Usage: developmentalTimeABBench --impl baseline|cuda --mode train|infer "
                << "[--scenario-id ID] [--rows N] [--cols N] [--nnz-row N] "
                << "[--stem-dim N] [--hidden-dim N] [--time-bins N] [--warmup N] [--iters N] "
                << "[--out-dir DIR]\n";
            std::exit(0);
        } else {
            throw std::invalid_argument(std::string("unknown argument: ") + arg);
        }
    }
    require(cfg.rows > 0 && cfg.cols > 0 && cfg.nnz_per_row > 0, "rows/cols/nnz_per_row must be positive");
    require(cfg.cols >= cfg.nnz_per_row, "cols must be at least nnz_per_row");
    require(cfg.time_bins > 1, "time_bins must be > 1");
    require(cfg.iters > 0, "iters must be > 0");
    return cfg;
}

double elapsed_ms(const std::chrono::steady_clock::time_point &start, const std::chrono::steady_clock::time_point &stop) {
    return std::chrono::duration<double, std::milli>(stop - start).count();
}

void sync_if_needed(const torch::Device &device) {
    if (device.is_cuda()) cudaDeviceSynchronize();
}

dt::TimeBatch make_batch(const bench_config &cfg) {
    torch::Tensor crow = torch::empty({ cfg.rows + 1 }, torch::TensorOptions().dtype(torch::kInt64));
    torch::Tensor col = torch::empty({ cfg.rows * cfg.nnz_per_row }, torch::TensorOptions().dtype(torch::kInt64));
    torch::Tensor values = torch::empty({ cfg.rows * cfg.nnz_per_row }, torch::TensorOptions().dtype(torch::kFloat16));
    auto *crow_ptr = crow.data_ptr<std::int64_t>();
    auto *col_ptr = col.data_ptr<std::int64_t>();
    auto *value_ptr = values.data_ptr<at::Half>();
    crow_ptr[0] = 0;
    for (std::int64_t row = 0; row < cfg.rows; ++row) {
        const std::int64_t base = row * cfg.nnz_per_row;
        crow_ptr[row + 1] = base + cfg.nnz_per_row;
        const std::int64_t start = (row * 1315423911ll) % cfg.cols;
        for (std::int64_t j = 0; j < cfg.nnz_per_row; ++j) {
            const std::int64_t idx = base + j;
            col_ptr[idx] = (start + j * 97ll) % cfg.cols;
            value_ptr[idx] = static_cast<at::Half>(0.25f + static_cast<float>((idx + 3) % 17) * 0.0625f);
        }
    }

    torch::Tensor labels = torch::empty({ cfg.rows }, torch::TensorOptions().dtype(torch::kFloat32));
    torch::Tensor bins = torch::empty({ cfg.rows }, torch::TensorOptions().dtype(torch::kInt64));
    auto *label_ptr = labels.data_ptr<float>();
    auto *bin_ptr = bins.data_ptr<std::int64_t>();
    for (std::int64_t row = 0; row < cfg.rows; ++row) {
        const float t = cfg.rows == 1 ? 0.0f : static_cast<float>(row) / static_cast<float>(cfg.rows - 1);
        label_ptr[row] = t;
        bin_ptr[row] = static_cast<std::int64_t>(t * static_cast<float>(cfg.time_bins - 1) + 1.0e-6f);
    }
    torch::Tensor cell_indices = torch::arange(0, cfg.rows, torch::TensorOptions().dtype(torch::kInt64));
    torch::Tensor features = torch::sparse_csr_tensor(
        crow,
        col,
        values,
        { cfg.rows, cfg.cols },
        torch::TensorOptions().dtype(torch::kFloat16));
    return dt::TimeBatch{ features, labels, bins, cell_indices };
}

void write_results(
    const bench_config &cfg,
    const std::string &out_dir,
    const std::string &status,
    double load_ms,
    double setup_ms,
    double steady_ms,
    double collect_ms,
    double checksum,
    double aux_metric) {
    if (out_dir.empty()) return;
    std::filesystem::create_directories(out_dir);
    {
        std::ofstream config_out(out_dir + "/run_config.json");
        config_out
            << "{\n"
            << "  \"implementation\": \"" << cfg.impl << "\",\n"
            << "  \"mode\": \"" << cfg.mode << "\",\n"
            << "  \"scenario_id\": \"" << cfg.scenario_id << "\",\n"
            << "  \"rows\": " << cfg.rows << ",\n"
            << "  \"cols\": " << cfg.cols << ",\n"
            << "  \"nnz_per_row\": " << cfg.nnz_per_row << ",\n"
            << "  \"stem_dim\": " << cfg.stem_dim << ",\n"
            << "  \"hidden_dim\": " << cfg.hidden_dim << ",\n"
            << "  \"time_bins\": " << cfg.time_bins << ",\n"
            << "  \"warmup\": " << cfg.warmup << ",\n"
            << "  \"repeats\": " << cfg.iters << "\n"
            << "}\n";
    }
    {
        const double total_ms = load_ms + setup_ms + steady_ms + collect_ms;
        std::ofstream results_out(out_dir + "/results.json");
        results_out << std::fixed << std::setprecision(6)
            << "{\n"
            << "  \"status\": \"" << status << "\",\n"
            << "  \"correctness\": \"unchecked\",\n"
            << "  \"primary_metric_ms\": " << steady_ms << ",\n"
            << "  \"phases\": {\n"
            << "    \"load_or_generate\": " << load_ms << ",\n"
            << "    \"setup_or_prepare\": " << setup_ms << ",\n"
            << "    \"steady_state_compute\": " << steady_ms << ",\n"
            << "    \"collect_or_finalize\": " << collect_ms << ",\n"
            << "    \"end_to_end\": " << total_ms << "\n"
            << "  },\n"
            << "  \"checksum\": " << checksum << ",\n"
            << "  \"aux_metric\": " << aux_metric << "\n"
            << "}\n";
    }
}

int run_baseline(const bench_config &cfg, const std::string &out_dir) {
    require(torch::cuda::is_available(), "developmentalTimeABBench baseline requires CUDA");
    const torch::Device device(torch::kCUDA, 0);

    const auto load_start = std::chrono::steady_clock::now();
    dt::TimeBatch batch = make_batch(cfg);
    const auto load_stop = std::chrono::steady_clock::now();

    const auto setup_start = std::chrono::steady_clock::now();
    batch.features = batch.features.to(device);
    batch.day_labels = batch.day_labels.to(device);
    batch.day_buckets = batch.day_buckets.to(device);
    batch.cell_indices = batch.cell_indices.to(device);

    dt::SparseTimeEncoderConfig encoder_config;
    encoder_config.input_genes = cfg.cols;
    encoder_config.hidden_dim = cfg.stem_dim;
    encoder_config.proj_dim = cfg.hidden_dim;
    dt::DevelopmentalTimeHeadConfig head_config;
    head_config.input_dim = cfg.hidden_dim;
    head_config.hidden_dim = cfg.hidden_dim;
    head_config.num_time_bins = cfg.time_bins;
    dt::DevelopmentalTimeModel model(encoder_config, head_config);
    model->to(device);
    dt::DevelopmentalTimeTrainConfig train_config;
    torch::optim::SGD optimizer = dt::make_developmental_time_optimizer(model, train_config);
    const auto setup_stop = std::chrono::steady_clock::now();

    for (std::int64_t i = 0; i < cfg.warmup; ++i) {
        if (cfg.mode == "train") {
            (void) dt::train_developmental_time_step(model, optimizer, batch);
        } else {
            torch::NoGradGuard no_grad;
            (void) model->predict_time(batch.features);
        }
        sync_if_needed(device);
    }

    dt::DevelopmentalTimeTrainStep last_step{};
    torch::Tensor last_prediction;
    const auto steady_start = std::chrono::steady_clock::now();
    for (std::int64_t i = 0; i < cfg.iters; ++i) {
        if (cfg.mode == "train") {
            last_step = dt::train_developmental_time_step(model, optimizer, batch);
        } else {
            torch::NoGradGuard no_grad;
            last_prediction = model->predict_time(batch.features);
        }
        sync_if_needed(device);
    }
    const auto steady_stop = std::chrono::steady_clock::now();

    const auto collect_start = std::chrono::steady_clock::now();
    const torch::Tensor prediction = cfg.mode == "train"
        ? last_step.output.predicted_time.detach().to(torch::kCPU)
        : last_prediction.detach().to(torch::kCPU);
    const double checksum = prediction.sum().item<double>();
    const double aux_metric = cfg.mode == "train"
        ? last_step.loss.total.detach().to(torch::kCPU).item<double>()
        : 0.0;
    const auto collect_stop = std::chrono::steady_clock::now();

    write_results(
        cfg,
        out_dir,
        "ok",
        elapsed_ms(load_start, load_stop),
        elapsed_ms(setup_start, setup_stop),
        elapsed_ms(steady_start, steady_stop) / static_cast<double>(cfg.iters),
        elapsed_ms(collect_start, collect_stop),
        checksum,
        aux_metric);
    return 0;
}

int run_cuda_impl(const bench_config &cfg, const std::string &out_dir) {
    require(torch::cuda::is_available(), "developmentalTimeABBench cuda implementation requires CUDA");
    const torch::Device device(torch::kCUDA, 0);

    const auto load_start = std::chrono::steady_clock::now();
    dt::TimeBatch batch = make_batch(cfg);
    const auto load_stop = std::chrono::steady_clock::now();

    const auto setup_start = std::chrono::steady_clock::now();
    batch.features = batch.features.to(device);
    batch.day_labels = batch.day_labels.to(device);
    batch.day_buckets = batch.day_buckets.to(device);
    batch.cell_indices = batch.cell_indices.to(device);

    dtc::DevelopmentalTimeCudaConfig model_config;
    model_config.input_genes = cfg.cols;
    model_config.stem_dim = cfg.stem_dim;
    model_config.hidden_dim = cfg.hidden_dim;
    model_config.num_time_bins = cfg.time_bins;
    model_config.device = device;
    dtc::DevelopmentalTimeCudaModel model(model_config);
    const auto setup_stop = std::chrono::steady_clock::now();

    for (std::int64_t i = 0; i < cfg.warmup; ++i) {
        if (cfg.mode == "train") {
            (void) dtc::train_step(model, batch);
        } else {
            (void) dtc::predict_time(model, batch.features);
        }
        sync_if_needed(device);
    }

    dtc::DevelopmentalTimeCudaTrainStep last_step{};
    torch::Tensor last_prediction;
    const auto steady_start = std::chrono::steady_clock::now();
    for (std::int64_t i = 0; i < cfg.iters; ++i) {
        if (cfg.mode == "train") {
            last_step = dtc::train_step(model, batch);
        } else {
            last_prediction = dtc::predict_time(model, batch.features);
        }
        sync_if_needed(device);
    }
    const auto steady_stop = std::chrono::steady_clock::now();

    const auto collect_start = std::chrono::steady_clock::now();
    const torch::Tensor prediction = cfg.mode == "train"
        ? last_step.output.predicted_time.detach().to(torch::kCPU)
        : last_prediction.detach().to(torch::kCPU);
    const double checksum = prediction.sum().item<double>();
    const double aux_metric = cfg.mode == "train"
        ? last_step.loss.total.detach().to(torch::kCPU).item<double>()
        : 0.0;
    const auto collect_stop = std::chrono::steady_clock::now();

    write_results(
        cfg,
        out_dir,
        "ok",
        elapsed_ms(load_start, load_stop),
        elapsed_ms(setup_start, setup_stop),
        elapsed_ms(steady_start, steady_stop) / static_cast<double>(cfg.iters),
        elapsed_ms(collect_start, collect_stop),
        checksum,
        aux_metric);
    return 0;
}

} // namespace

int main(int argc, char **argv) {
    cellerator::bench::benchmark_mutex_guard benchmark_mutex("developmentalTimeABBench");
    std::cout << "cuda_mode=" << cellerator::build::cuda_mode_name << '\n';
    const bench_config cfg = parse_args(argc, argv);
    if (cfg.impl == "baseline") return run_baseline(cfg, cfg.out_dir);
    if (cfg.impl == "cuda") return run_cuda_impl(cfg, cfg.out_dir);
    throw std::invalid_argument("unknown implementation; expected baseline or cuda");
}
