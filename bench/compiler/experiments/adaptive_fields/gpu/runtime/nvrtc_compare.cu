#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using clock_type = std::chrono::steady_clock;

void check_cuda(cudaError_t status, const char* where) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(where) + ": " + cudaGetErrorString(status));
}
void check_driver(CUresult status, const char* where) {
  if (status == CUDA_SUCCESS) return;
  const char* name = nullptr;
  const char* message = nullptr;
  cuGetErrorName(status, &name);
  cuGetErrorString(status, &message);
  throw std::runtime_error(std::string(where) + ": " +
      (name ? name : "CUDA_ERROR") + " " + (message ? message : ""));
}
void check_nvrtc(nvrtcResult status, const char* where) {
  if (status != NVRTC_SUCCESS)
    throw std::runtime_error(std::string(where) + ": " + nvrtcGetErrorString(status));
}
std::string read_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot read " + path);
  return std::string(std::istreambuf_iterator<char>(in), {});
}
double elapsed_ms(clock_type::time_point start) {
  return std::chrono::duration<double, std::milli>(clock_type::now() - start).count();
}

struct module_artifact {
  CUmodule module = nullptr;
  CUfunction function = nullptr;
  double load_ms = 0.0;
};

module_artifact load_module(const char* image, const char* function_name) {
  module_artifact result;
  const auto started = clock_type::now();
  check_driver(cuModuleLoadData(&result.module, image), "cuModuleLoadData");
  check_driver(cuModuleGetFunction(&result.function, result.module, function_name),
               "cuModuleGetFunction");
  result.load_ms = elapsed_ms(started);
  return result;
}

std::vector<float> reference(const long long* offsets, const long long* columns,
                             const float* values, const float* x, int rows,
                             const long long* segments, int segment_count) {
  std::vector<double> row_values(static_cast<std::size_t>(rows));
  std::vector<float> output(static_cast<std::size_t>(rows), 0.0f);
  for (int row = 0; row < rows; ++row) {
    double sum = 0.0;
    for (long long edge = offsets[row]; edge < offsets[row + 1]; ++edge)
      sum += values[edge] * x[columns[edge]];
    row_values[row] = sum;
  }
  for (int segment = 0; segment < segment_count; ++segment) {
    const auto begin = segments[segment];
    const auto end = segments[segment + 1];
    if (begin == end) continue;
    double maximum = row_values[begin];
    for (long long row = begin + 1; row < end; ++row)
      maximum = std::max(maximum, row_values[row]);
    double denominator = 0.0;
    for (long long row = begin; row < end; ++row)
      denominator += std::exp(row_values[row] - maximum);
    for (long long row = begin; row < end; ++row)
      output[row] = static_cast<float>(std::exp(row_values[row] - maximum) / denominator);
  }
  return output;
}

void invoke(CUfunction function, CUstream stream,
            CUdeviceptr offsets, CUdeviceptr columns, CUdeviceptr values,
            CUdeviceptr x, CUdeviceptr scratch, CUdeviceptr output,
            int rows, CUdeviceptr segment_offsets, int segments) {
  void* arguments[] = {&offsets, &columns, &values, &x, &scratch, &output,
                       &rows, &segment_offsets, &segments};
  check_driver(cuLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, stream,
                             arguments, nullptr), "cuLaunchKernel");
}

std::vector<double> paired_run(CUfunction nvcc, CUfunction nvrtc, CUstream stream,
                               CUdeviceptr offsets, CUdeviceptr columns,
                               CUdeviceptr values, CUdeviceptr x, CUdeviceptr scratch,
                               CUdeviceptr output, int rows,
                               CUdeviceptr segment_offsets, int segments,
                               int warmups, int pairs, int rounds,
                               std::vector<std::vector<std::vector<double>>>& samples,
                               std::vector<std::vector<bool>>& order) {
  for (int i = 0; i < warmups; ++i) {
    invoke(nvcc, stream, offsets, columns, values, x, scratch, output,
           rows, segment_offsets, segments);
    invoke(nvrtc, stream, offsets, columns, values, x, scratch, output,
           rows, segment_offsets, segments);
  }
  check_cuda(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)),
             "warmup sync");
  cudaEvent_t start{}, stop{};
  check_cuda(cudaEventCreate(&start), "cudaEventCreate start");
  check_cuda(cudaEventCreate(&stop), "cudaEventCreate stop");
  std::vector<double> ratios;
  for (int round = 0; round < rounds; ++round) {
    for (int pair = 0; pair < pairs; ++pair) {
      const bool nvcc_first = ((round + pair) % 2 == 0);
      order[round][pair] = nvcc_first;
      double elapsed[2]{};
      CUfunction functions[2] = {nvcc_first ? nvcc : nvrtc,
                                 nvcc_first ? nvrtc : nvcc};
      for (int i = 0; i < 2; ++i) {
        check_cuda(cudaEventRecord(start, reinterpret_cast<cudaStream_t>(stream)),
                   "cudaEventRecord start");
        const auto wall_start = clock_type::now();
        invoke(functions[i], stream, offsets, columns, values, x, scratch,
               output, rows, segment_offsets, segments);
        check_cuda(cudaEventRecord(stop, reinterpret_cast<cudaStream_t>(stream)),
                   "cudaEventRecord stop");
        check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize");
        float device_ms = 0.0f;
        check_cuda(cudaEventElapsedTime(&device_ms, start, stop),
                   "cudaEventElapsedTime");
        const double wall_ms = elapsed_ms(wall_start);
        elapsed[nvcc_first ? i : 1 - i] = wall_ms;
        samples[round][pair].push_back(static_cast<double>(device_ms));
        samples[round][pair].push_back(wall_ms);
      }
      ratios.push_back(elapsed[0] / elapsed[1]);
    }
  }
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return ratios;
}

template <class T>
CUdeviceptr upload(const T* data, std::size_t count, CUstream stream) {
  CUdeviceptr pointer = 0;
  check_driver(cuMemAlloc(&pointer, count * sizeof(T)), "cuMemAlloc");
  check_driver(cuMemcpyHtoDAsync(pointer, data, count * sizeof(T), stream),
               "cuMemcpyHtoDAsync");
  return pointer;
}

} // namespace

int main(int argc, char** argv) {
  try {
    if (std::getenv("CELLERATOR_SPARSE_SUPPORT_BENCH_ADMITTED") == nullptr ||
        std::string(std::getenv("CELLERATOR_SPARSE_SUPPORT_BENCH_ADMITTED")) != "1")
      throw std::runtime_error("E7 requires the parent experiment marker (accidental-run guard only)");
    const char* visible_device = std::getenv("CUDA_VISIBLE_DEVICES");
    if (!visible_device ||
        (std::strncmp(visible_device, "GPU-", 4) != 0 &&
         std::strncmp(visible_device, "MIG-", 4) != 0) ||
        std::strchr(visible_device, ',') != nullptr)
      throw std::runtime_error("one UUID-routed CUDA_VISIBLE_DEVICES value is required");
    const char* run_key = std::getenv("ADAPTIVE_FIELDS_RUN_KEY");
    if (!run_key || !*run_key)
      throw std::runtime_error("parent JobSpec correlation key is required");
    if (argc != 5) {
      std::cerr << "usage: nvrtc_compare region.cu nvcc.fatbin output.json arch\n";
      return 2;
    }
    const std::string source_path = argv[1];
    const std::string fatbin_path = argv[2];
    const std::string output_path = argv[3];
    const std::string arch = argv[4];
    // E7's fixed timing uses 5 warmups, 21 pairs, and four rounds. Keeping the
    // counts fixed here avoids accidental deviation in scheduler runs.
    constexpr int kWarmups = 5;
    constexpr int kPairs = 21;
    constexpr int kRounds = 4;
    const auto source = read_file(source_path);
    const auto fatbin = read_file(fatbin_path);

    check_driver(cuInit(0), "cuInit");
    check_cuda(cudaSetDevice(0), "cudaSetDevice");
    check_cuda(cudaFree(nullptr), "initialize CUDA runtime context");
    CUcontext context = nullptr;
    check_driver(cuCtxGetCurrent(&context), "cuCtxGetCurrent");
    if (!context) throw std::runtime_error("CUDA runtime did not establish a context");
    int driver_version = 0;
    int runtime_version = 0;
    int nvrtc_major = 0;
    int nvrtc_minor = 0;
    check_driver(cuDriverGetVersion(&driver_version), "cuDriverGetVersion");
    check_cuda(cudaRuntimeGetVersion(&runtime_version), "cudaRuntimeGetVersion");
    check_nvrtc(nvrtcVersion(&nvrtc_major, &nvrtc_minor), "nvrtcVersion");

    const auto rtc_compile_start = clock_type::now();
    nvrtcProgram program{};
    check_nvrtc(nvrtcCreateProgram(&program, source.c_str(), "region.cu",
                                   0, nullptr, nullptr), "nvrtcCreateProgram");
    const std::string option = "--gpu-architecture=" + arch;
    const char* options[] = {"--std=c++17", option.c_str()};
    const nvrtcResult compile_status = nvrtcCompileProgram(program, 2, options);
    std::size_t log_size = 0;
    check_nvrtc(nvrtcGetProgramLogSize(program, &log_size), "nvrtcGetProgramLogSize");
    std::string compile_log(log_size, '\0');
    if (log_size) check_nvrtc(nvrtcGetProgramLog(program, compile_log.data()),
                              "nvrtcGetProgramLog");
    if (compile_status != NVRTC_SUCCESS) {
      throw std::runtime_error("NVRTC compilation failed: " + compile_log);
    }
    std::size_t ptx_size = 0;
    check_nvrtc(nvrtcGetPTXSize(program, &ptx_size), "nvrtcGetPTXSize");
    std::string ptx(ptx_size, '\0');
    check_nvrtc(nvrtcGetPTX(program, ptx.data()), "nvrtcGetPTX");
    const double nvrtc_compile_ms = elapsed_ms(rtc_compile_start);

    module_artifact nvcc_module = load_module(fatbin.c_str(), "cellerator_region");
    module_artifact nvrtc_module = load_module(ptx.c_str(), "cellerator_region");

    // Five relation rows include an empty CSR row, signed weights, an empty
    // output segment, and singleton segments. The same pointers and scalar
    // bindings are passed to each module.
    long long row_offsets_host[] = {0, 2, 3, 3, 5, 6};
    long long columns_host[] = {0, 2, 1, 0, 2, 1};
    float values_host[] = {-0.5f, 0.25f, 1.0f, 0.75f, -0.5f, 2.0f};
    float x_host[] = {2.0f, 3.0f, 4.0f};
    long long segment_offsets_host[] = {0, 1, 1, 3, 4, 5};
    constexpr int rows = 5;
    constexpr int segments = 5;
    float scratch_host[rows]{};
    float output_host[rows]{};

    cudaStream_t runtime_stream{};
    check_cuda(cudaStreamCreateWithFlags(&runtime_stream, cudaStreamNonBlocking),
               "cudaStreamCreateWithFlags");
    CUstream stream = reinterpret_cast<CUstream>(runtime_stream);
    CUdeviceptr row_offsets = upload(row_offsets_host, 6, stream);
    CUdeviceptr columns = upload(columns_host, 6, stream);
    CUdeviceptr values = upload(values_host, 6, stream);
    CUdeviceptr x = upload(x_host, 3, stream);
    CUdeviceptr scratch = upload(scratch_host, rows, stream);
    CUdeviceptr output = upload(output_host, rows, stream);
    CUdeviceptr segment_offsets = upload(segment_offsets_host, segments + 1, stream);
    check_cuda(cudaStreamSynchronize(runtime_stream), "input upload sync");

    std::vector<std::vector<std::vector<double>>> samples(
        kRounds, std::vector<std::vector<double>>(kPairs));
    std::vector<std::vector<bool>> order(kRounds, std::vector<bool>(kPairs));
    const auto ratios = paired_run(nvcc_module.function, nvrtc_module.function,
        stream, row_offsets, columns, values, x, scratch, output, rows,
        segment_offsets, segments, kWarmups, kPairs, kRounds, samples, order);

    auto result_for = [&](CUfunction fn) {
      invoke(fn, stream, row_offsets, columns, values, x, scratch, output,
             rows, segment_offsets, segments);
      check_cuda(cudaStreamSynchronize(runtime_stream), "result sync");
      std::vector<float> result(rows);
      check_driver(cuMemcpyDtoH(result.data(), output, rows * sizeof(float)),
                   "cuMemcpyDtoH result");
      return result;
    };
    const auto expected = reference(row_offsets_host, columns_host, values_host,
                                    x_host, rows, segment_offsets_host, segments);
    auto nvcc_output = result_for(nvcc_module.function);
    auto nvrtc_output = result_for(nvrtc_module.function);
    auto max_error = [&](const std::vector<float>& actual) {
      float error = 0.0f;
      for (int i = 0; i < rows; ++i)
        error = std::max(error, std::abs(actual[i] - expected[i]));
      return error;
    };
    const float nvcc_error = max_error(nvcc_output);
    const float nvrtc_error = max_error(nvrtc_output);
    if (nvcc_error > 2e-6f || nvrtc_error > 2e-6f)
      throw std::runtime_error("compiled device region disagrees with FP64-independent host reference");

    // Rebind only numerical relation weights. Both binaries must observe the
    // same new values without recompilation or module replacement.
    values_host[0] = 1.5f;
    check_driver(cuMemcpyHtoDAsync(values, values_host, sizeof(values_host), stream),
                 "rebind numerical values");
    check_cuda(cudaStreamSynchronize(runtime_stream), "rebind sync");
    nvcc_output = result_for(nvcc_module.function);
    nvrtc_output = result_for(nvrtc_module.function);
    const auto rebound_expected = reference(row_offsets_host, columns_host,
        values_host, x_host, rows, segment_offsets_host, segments);
    auto rebound_error = [&](const std::vector<float>& actual) {
      float error = 0.0f;
      for (int i = 0; i < rows; ++i)
        error = std::max(error, std::abs(actual[i] - rebound_expected[i]));
      return error;
    };
    if (rebound_error(nvcc_output) > 2e-6f ||
        rebound_error(nvrtc_output) > 2e-6f)
      throw std::runtime_error("compiled region did not honor dynamic value rebinding");

    std::ostringstream json;
    json << std::setprecision(10)
         << "{\n  \"schema\": \"cellerator.adaptive_fields.e7_nvrtc_compare.v1\",\n"
         << "  \"gpu_execution_performed\": true,\n"
         << "  \"same_region_file\": \"" << source_path << "\",\n"
         << "  \"same_bindings\": true,\n"
         << "  \"cold_nvrtc_compile_ms\": " << nvrtc_compile_ms << ",\n"
         << "  \"nvcc_precompiled_module_load_ms\": " << nvcc_module.load_ms << ",\n"
         << "  \"nvrtc_module_load_ms\": " << nvrtc_module.load_ms << ",\n"
         << "  \"driver_version\": " << driver_version << ",\n"
         << "  \"runtime_version\": " << runtime_version << ",\n"
         << "  \"nvrtc_version\": \"" << nvrtc_major << "." << nvrtc_minor << "\",\n"
         << "  \"arch\": \"" << arch << "\",\n"
         << "  \"warmups\": " << kWarmups << ", \"pairs_per_round\": " << kPairs
         << ", \"rounds\": " << kRounds << ",\n"
         << "  \"correctness\": {\"nvcc_max_abs\": " << nvcc_error
         << ", \"nvrtc_max_abs\": " << nvrtc_error
         << ", \"rebound_nvcc_max_abs\": " << rebound_error(nvcc_output)
         << ", \"rebound_nvrtc_max_abs\": " << rebound_error(nvrtc_output) << "},\n"
         << "  \"fixture\": {\"row_offsets\": [0,2,3,3,5,6],"
         << " \"columns\": [0,2,1,0,2,1],"
         << " \"values_before_rebind\": [-0.5,0.25,1.0,0.75,-0.5,2.0],"
         << " \"values_after_rebind\": [1.5,0.25,1.0,0.75,-0.5,2.0],"
         << " \"x\": [2.0,3.0,4.0],"
         << " \"segment_offsets\": [0,1,1,3,4,5],"
         << " \"rows\": 5, \"segments\": 5},\n"
         << "  \"outputs\": {\"nvcc\": [";
    for (int i = 0; i < rows; ++i) {
      if (i) json << ", ";
      json << nvcc_output[i];
    }
    json << "], \"nvrtc\": [";
    for (int i = 0; i < rows; ++i) {
      if (i) json << ", ";
      json << nvrtc_output[i];
    }
    json << "], \"fp64_reference_after_rebind\": [";
    for (int i = 0; i < rows; ++i) {
      if (i) json << ", ";
      json << rebound_expected[i];
    }
    json << "]},\n"
         << "  \"paired_wall_ratio_nvcc_over_nvrtc\": [";
    for (std::size_t i = 0; i < ratios.size(); ++i) {
      if (i) json << ", ";
      json << ratios[i];
    }
    json << "],\n  \"samples\": [";
    for (int round = 0; round < kRounds; ++round) {
      if (round) json << ", ";
      json << "[";
      for (int pair = 0; pair < kPairs; ++pair) {
        if (pair) json << ", ";
        const auto& row = samples[round][pair];
        json << "{\"round\": " << round << ", \"pair\": " << pair
             << ", \"nvcc_first\": " << (order[round][pair] ? "true" : "false")
             << ", \"first_device_ms\": " << row[0]
             << ", \"first_wall_ms\": " << row[1]
             << ", \"second_device_ms\": " << row[2]
             << ", \"second_wall_ms\": " << row[3] << "}";
      }
      json << "]";
    }
    json << "]\n}\n";
    std::ofstream out(output_path);
    if (!out) throw std::runtime_error("cannot write result JSON");
    out << json.str();
    out.close();

    cuModuleUnload(nvcc_module.module);
    cuModuleUnload(nvrtc_module.module);
    nvrtcDestroyProgram(&program);
    for (CUdeviceptr pointer : {row_offsets, columns, values, x, scratch,
                                output, segment_offsets})
      cuMemFree(pointer);
    cudaStreamDestroy(runtime_stream);
    std::cout << "result=" << output_path << " nvrtc_compile_ms="
              << nvrtc_compile_ms << "\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\n";
    return 1;
  }
}
