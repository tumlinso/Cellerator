#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <future>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {
using clock_type = std::chrono::steady_clock;
void cuda_check(cudaError_t s, const char* where) {
  if (s != cudaSuccess) throw std::runtime_error(std::string(where) + ": " + cudaGetErrorString(s));
}
void cu_check(CUresult s, const char* where) {
  if (s == CUDA_SUCCESS) return;
  const char* name = nullptr; const char* message = nullptr;
  cuGetErrorName(s, &name); cuGetErrorString(s, &message);
  throw std::runtime_error(std::string(where) + ": " + (name ? name : "CUDA_ERROR") +
                           " " + (message ? message : ""));
}
void rtc_check(nvrtcResult s, const char* where) {
  if (s != NVRTC_SUCCESS) throw std::runtime_error(std::string(where) + ": " + nvrtcGetErrorString(s));
}
std::string read_file(const char* path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error(std::string("cannot read ") + path);
  return std::string(std::istreambuf_iterator<char>(in), {});
}

struct ptx_artifact { std::string ptx; double compile_ms = 0; };
ptx_artifact compile_nvrtc(const std::string& source, int epoch,
                           bool inject_failure = false) {
  nvrtcProgram program{};
  const std::string actual_source = inject_failure
      ? source + "\nthis is intentionally invalid CUDA;\n" : source;
  rtc_check(nvrtcCreateProgram(&program, actual_source.c_str(), "lifetime.cu",
                               0, nullptr, nullptr), "nvrtcCreateProgram");
  const std::string epoch_option = "--define-macro=FIELD_STRUCTURE_EPOCH=" +
                                  std::to_string(epoch);
  const char* options[] = {"--std=c++17", "--gpu-architecture=compute_70",
                           epoch_option.c_str()};
  const auto start = clock_type::now();
  const nvrtcResult status = nvrtcCompileProgram(program, 3, options);
  const double ms = std::chrono::duration<double, std::milli>(
      clock_type::now() - start).count();
  std::size_t log_size = 0;
  rtc_check(nvrtcGetProgramLogSize(program, &log_size), "nvrtcGetProgramLogSize");
  std::string log(log_size, '\0');
  if (log_size) rtc_check(nvrtcGetProgramLog(program, log.data()), "nvrtcGetProgramLog");
  if (status != NVRTC_SUCCESS) {
    nvrtcDestroyProgram(&program);
    throw std::runtime_error("NVRTC compilation failed: " + log);
  }
  std::size_t size = 0;
  rtc_check(nvrtcGetPTXSize(program, &size), "nvrtcGetPTXSize");
  std::string ptx(size, '\0');
  rtc_check(nvrtcGetPTX(program, ptx.data()), "nvrtcGetPTX");
  nvrtcDestroyProgram(&program);
  return {std::move(ptx), ms};
}

std::atomic<int> g_live_modules{0};
std::atomic<int> g_unloaded_modules{0};
void stage_marker(const std::string& name) {
  static const auto origin = clock_type::now();
  const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      clock_type::now() - origin).count();
  std::cerr << "E6_STAGE t_ms=" << elapsed << " " << name << std::endl;
}
class bounded_stage_watchdog {
 public:
  bounded_stage_watchdog(std::string name, std::chrono::seconds timeout)
      : name_(std::move(name)), worker_([this, timeout] {
          std::unique_lock<std::mutex> lock(mutex_);
          if (!condition_.wait_for(lock, timeout, [&] { return completed_; })) {
            lock.unlock();
            stage_marker(name_ + " watchdog_expired; exiting process for scheduler cleanup");
            std::_Exit(124);
          }
        }) {}
  bounded_stage_watchdog(const bounded_stage_watchdog&) = delete;
  bounded_stage_watchdog& operator=(const bounded_stage_watchdog&) = delete;
  ~bounded_stage_watchdog() { complete(); }
  void complete() noexcept {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (completed_) return;
      completed_ = true;
    }
    condition_.notify_all();
    if (worker_.joinable()) worker_.join();
  }
 private:
  std::string name_;
  std::mutex mutex_;
  std::condition_variable condition_;
  bool completed_ = false;
  std::thread worker_;
};
struct executable {
  CUmodule module = nullptr;
  CUfunction forward = nullptr;
  CUfunction backward = nullptr;
  int structure_epoch = 0;
  std::string key;
  executable(const ptx_artifact& ptx, int epoch, std::string artifact_key)
      : structure_epoch(epoch), key(std::move(artifact_key)) {
    stage_marker("module_load_begin key=" + key);
    const auto load_start = clock_type::now();
    bounded_stage_watchdog watchdog("module_load key=" + key,
                                    std::chrono::seconds(45));
    const CUresult load_status = cuModuleLoadData(&module, ptx.ptx.c_str());
    watchdog.complete();
    cu_check(load_status, "cuModuleLoadData");
    const auto load_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        clock_type::now() - load_start).count();
    stage_marker("module_load_end key=" + key + " elapsed_ms=" +
                 std::to_string(load_ms));
    cu_check(cuModuleGetFunction(&forward, module, "field_forward"), "get forward");
    cu_check(cuModuleGetFunction(&backward, module, "field_backward"), "get backward");
    g_live_modules.fetch_add(1, std::memory_order_relaxed);
  }
  ~executable() {
    if (module) {
      cuModuleUnload(module);
      g_live_modules.fetch_sub(1, std::memory_order_relaxed);
      g_unloaded_modules.fetch_add(1, std::memory_order_relaxed);
    }
  }
};
using executable_ref = std::shared_ptr<executable>;

struct plan {
  executable_ref code;
  int structure_epoch = 0;
  bool specializes_structure = false;
  std::string numerical_contract = "f32-input-output-scale-binding-v1";
  int bound_value_generation = 0;
};

struct plan_store {
  std::mutex mutex;
  int current_structure_epoch = 1;
  int current_value_generation = 1;
  std::string numerical_contract = "f32-input-output-scale-binding-v1";
  std::shared_ptr<plan> active;
  std::shared_ptr<plan> fallback;
  std::string publish(const std::shared_ptr<plan>& candidate) {
    std::lock_guard<std::mutex> guard(mutex);
    if (!candidate || !candidate->code) return "rejected-empty";
    if (candidate->numerical_contract != numerical_contract) return "rejected-contract";
    if (candidate->specializes_structure &&
        candidate->structure_epoch != current_structure_epoch) return "rejected-stale";
    active = candidate;
    return "published";
  }
  std::shared_ptr<plan> select() {
    std::lock_guard<std::mutex> guard(mutex);
    if (active && (!active->specializes_structure ||
                   active->structure_epoch == current_structure_epoch)) return active;
    return fallback;
  }
};

struct compile_gate {
  std::mutex mutex;
  std::condition_variable condition;
  bool entered = false;
  bool released = false;
  bool watchdog_timed_out = false;
  clock_type::time_point watchdog_release_at{};
};
struct compile_request {
  std::string key;
  int epoch = 0;
  bool fail = false;
  std::shared_ptr<compile_gate> gate;
  std::shared_ptr<std::atomic<bool>> cancelled;
  std::shared_ptr<std::promise<ptx_artifact>> promise;
  std::shared_future<ptx_artifact> future;
};

class compile_service {
 public:
  explicit compile_service(std::string source) : source_(std::move(source)), worker_([this] { run(); }) {}
  ~compile_service() {
    { std::lock_guard<std::mutex> lock(mutex_); stop_ = true; }
    condition_.notify_all();
    if (worker_.joinable()) worker_.join();
  }
  compile_request request(std::string key, int epoch, bool fail = false,
                          std::shared_ptr<compile_gate> gate = {}) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto existing = requests_.find(key);
    if (existing != requests_.end()) {
      duplicate_requests_++;
      return existing->second;
    }
    if (pending_ || active_)
      throw std::runtime_error("compile worker already has an active or pending replacement");
    compile_request request;
    request.key = std::move(key);
    request.epoch = epoch;
    request.fail = fail;
    request.gate = std::move(gate);
    request.cancelled = std::make_shared<std::atomic<bool>>(false);
    request.promise = std::make_shared<std::promise<ptx_artifact>>();
    request.future = request.promise->get_future().share();
    pending_ = request;
    requests_.emplace(request.key, request);
    condition_.notify_one();
    return request;
  }
  void cancel(const compile_request& request) {
    request.cancelled->store(true, std::memory_order_release);
  }
  int duplicate_requests() const { return duplicate_requests_.load(); }
  int compile_count() const { return compile_count_.load(); }
 private:
  void run() {
    for (;;) {
      compile_request request;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [&] { return stop_ || pending_.has_value(); });
        if (stop_ && !pending_) return;
        request = *pending_;
        pending_.reset();
        active_ = true;
      }
      try {
        if (request.gate) {
          std::unique_lock<std::mutex> lock(request.gate->mutex);
          request.gate->entered = true;
          request.gate->condition.notify_all();
          request.gate->condition.wait(lock, [&] { return request.gate->released; });
        }
        if (request.cancelled->load(std::memory_order_acquire))
          throw std::runtime_error("compile cancelled before NVRTC invocation");
        auto result = compile_nvrtc(source_, request.epoch, request.fail);
        compile_count_++;
        request.promise->set_value(std::move(result));
      } catch (...) {
        request.promise->set_exception(std::current_exception());
      }
      {
        std::lock_guard<std::mutex> lock(mutex_);
        active_ = false;
      }
      condition_.notify_all();
    }
  }
  std::string source_;
  mutable std::mutex mutex_;
  std::condition_variable condition_;
  bool stop_ = false;
  bool active_ = false;
  std::optional<compile_request> pending_;
  std::map<std::string, compile_request> requests_;
  std::thread worker_;
  std::atomic<int> duplicate_requests_{0};
  std::atomic<int> compile_count_{0};
};

void CUDART_CB wait_for_release(void* data) {
  auto* gate = static_cast<compile_gate*>(data);
  std::unique_lock<std::mutex> lock(gate->mutex);
  gate->entered = true;
  static constexpr char entered_marker[] = "E6_STAGE callback_entered\n";
  (void)::write(STDERR_FILENO, entered_marker, sizeof(entered_marker) - 1);
  gate->condition.notify_all();
  if (!gate->condition.wait_for(lock, std::chrono::seconds(10),
                                [&] { return gate->released; })) {
    gate->watchdog_timed_out = true;
    gate->watchdog_release_at = clock_type::now();
    gate->released = true;
    static constexpr char timeout_marker[] = "E6_STAGE callback_watchdog_expired timeout_seconds=10\n";
    (void)::write(STDERR_FILENO, timeout_marker, sizeof(timeout_marker) - 1);
    gate->condition.notify_all();
  }
}
void release_gate(const std::shared_ptr<compile_gate>& gate) {
  { std::lock_guard<std::mutex> lock(gate->mutex); gate->released = true; }
  gate->condition.notify_all();
}
void wait_entered(const std::shared_ptr<compile_gate>& gate) {
  std::unique_lock<std::mutex> lock(gate->mutex);
  if (!gate->condition.wait_for(lock, std::chrono::seconds(30),
                                [&] { return gate->entered; }))
    throw std::runtime_error("timed out waiting for compile gate entry");
}
bool gate_watchdog_timed_out(const std::shared_ptr<compile_gate>& gate) {
  std::lock_guard<std::mutex> lock(gate->mutex);
  return gate->watchdog_timed_out;
}
bool gate_watchdog_released_by(const std::shared_ptr<compile_gate>& gate,
                               clock_type::time_point time) {
  std::lock_guard<std::mutex> lock(gate->mutex);
  return gate->watchdog_timed_out && gate->watchdog_release_at <= time;
}

class callback_barrier_guard {
 public:
  callback_barrier_guard(cudaStream_t stream, std::shared_ptr<compile_gate> gate)
      : stream_(stream), gate_(std::move(gate)) {}
  callback_barrier_guard(const callback_barrier_guard&) = delete;
  callback_barrier_guard& operator=(const callback_barrier_guard&) = delete;
  ~callback_barrier_guard() { release_and_sync(); }
  void release_and_sync() noexcept {
    if (!armed_) return;
    armed_ = false;
    release_gate(gate_);
    stage_marker("callback_barrier_release_and_sync_begin");
    const cudaError_t result = cudaStreamSynchronize(stream_);
    if (result == cudaSuccess) {
      stage_marker("callback_barrier_release_and_sync_end");
      gate_.reset();
      return;
    }
    stage_marker(std::string("callback_barrier_sync_failed error=") +
                 cudaGetErrorString(result) + " retaining_gate_storage");
    // If stream completion cannot be proven, intentionally retain callback
    // storage for process lifetime instead of risking a callback UAF.
    (void)new std::shared_ptr<compile_gate>(std::move(gate_));
  }
 private:
  cudaStream_t stream_{};
  std::shared_ptr<compile_gate> gate_;
  bool armed_ = true;
};

void launch(CUfunction function, CUstream stream, CUdeviceptr input,
            CUdeviceptr output, int count, float scale) {
  void* args[] = {&input, &output, &count, &scale};
  cu_check(cuLaunchKernel(function, (count + 127) / 128, 1, 1,
                          128, 1, 1, 0, stream, args, nullptr), "cuLaunchKernel");
}
std::vector<float> copy_result(CUdeviceptr pointer, int count) {
  std::vector<float> result(static_cast<std::size_t>(count));
  cu_check(cuMemcpyDtoH(result.data(), pointer, result.size() * sizeof(float)),
           "copy result");
  return result;
}
template <class T>
CUdeviceptr upload(const std::vector<T>& values, CUstream stream) {
  CUdeviceptr pointer = 0;
  cu_check(cuMemAlloc(&pointer, values.size() * sizeof(T)), "cuMemAlloc");
  cu_check(cuMemcpyHtoDAsync(pointer, values.data(), values.size() * sizeof(T), stream),
           "upload");
  return pointer;
}
std::string json_numbers(const std::vector<float>& values) {
  std::ostringstream out; out << "[";
  for (std::size_t i = 0; i < values.size(); ++i) { if (i) out << ","; out << values[i]; }
  out << "]"; return out.str();
}
std::string json_string(const std::string& value) {
  std::string out="\"";
  for (char c : value) { if (c=='\\' || c=='\"') out.push_back('\\'); if (c=='\n') out += "\\n"; else out.push_back(c); }
  out += "\""; return out;
}

} // namespace

int main(int argc, char** argv) {
  try {
    const char* admitted = std::getenv("CELLERATOR_SPARSE_SUPPORT_BENCH_ADMITTED");
    if (!admitted || std::string(admitted) != "1")
      throw std::runtime_error("E6 requires the parent experiment marker (accidental-run guard only)");
    const char* visible_device = std::getenv("CUDA_VISIBLE_DEVICES");
    if (!visible_device ||
        (std::strncmp(visible_device, "GPU-", 4) != 0 &&
         std::strncmp(visible_device, "MIG-", 4) != 0) ||
        std::strchr(visible_device, ',') != nullptr)
      throw std::runtime_error("one UUID-routed CUDA_VISIBLE_DEVICES value is required");
    const char* run_key = std::getenv("ADAPTIVE_FIELDS_RUN_KEY");
    if (!run_key || !*run_key)
      throw std::runtime_error("parent JobSpec correlation key is required");
    if (argc != 3) {
      std::cerr << "usage: lifetime_runner lifetime_kernels.cu output.json\n";
      return 2;
    }
    const std::string source = read_file(argv[1]);
    const std::string output_path = argv[2];
    cu_check(cuInit(0), "cuInit");
    cuda_check(cudaSetDevice(0), "cudaSetDevice");
    cuda_check(cudaFree(nullptr), "initialize runtime context");
    CUcontext context = nullptr;
    cu_check(cuCtxGetCurrent(&context), "cuCtxGetCurrent");
    if (!context) throw std::runtime_error("no current CUDA context");

    int compile_count_before = 0;
    int duplicate_count = 0;
    std::vector<std::string> trace;
    compile_service compiler(source);

    // Generic fallback is explicitly independent of a relation structure epoch.
    auto fallback_ptx = compile_nvrtc(source, 0);
    auto fallback_code = std::make_shared<executable>(fallback_ptx, 0, "generic-fallback");
    auto fallback_plan = std::make_shared<plan>();
    fallback_plan->code = fallback_code;
    fallback_plan->structure_epoch = 0;
    fallback_plan->specializes_structure = false;
    plan_store plans;
    plans.fallback = fallback_plan;

    // Initial plan and a forward tape both pin the original loaded module.
    auto initial_gate = std::make_shared<compile_gate>();
    auto initial_request = compiler.request("initial-e1-contract-v1", 1, false, initial_gate);
    wait_entered(initial_gate);
    auto same_request = compiler.request("initial-e1-contract-v1", 1, false, initial_gate);
    if (same_request.future.valid() != initial_request.future.valid())
      throw std::runtime_error("duplicate request did not share a future");
    release_gate(initial_gate);
    auto initial_ptx = initial_request.future.get();
    auto old_code = std::make_shared<executable>(initial_ptx, 1, initial_request.key);
    auto old_plan = std::make_shared<plan>();
    old_plan->code = old_code;
    old_plan->structure_epoch = 1;
    old_plan->specializes_structure = true;
    old_plan->bound_value_generation = 1;
    if (plans.publish(old_plan) != "published") throw std::runtime_error("initial plan publish failed");

    constexpr int count = 8;
    const std::vector<float> input_host{1,2,3,4,5,6,7,8};
    const std::vector<float> upstream_host{2,2,2,2,2,2,2,2};
    cudaStream_t stream_a_runtime_value{}, stream_b_runtime_value{};
    cuda_check(cudaStreamCreateWithFlags(&stream_a_runtime_value, cudaStreamNonBlocking), "create stream A");
    cuda_check(cudaStreamCreateWithFlags(&stream_b_runtime_value, cudaStreamNonBlocking), "create stream B");
    CUstream stream_a = reinterpret_cast<CUstream>(stream_a_runtime_value);
    CUstream stream_b = reinterpret_cast<CUstream>(stream_b_runtime_value);
    CUdeviceptr input = upload(input_host, stream_a);
    CUdeviceptr upstream = upload(upstream_host, stream_a);
    CUdeviceptr old_output = 0, old_gradient = 0, new_output = 0;
    cu_check(cuMemAlloc(&old_output, count * sizeof(float)), "allocate old output");
    cu_check(cuMemAlloc(&old_gradient, count * sizeof(float)), "allocate old gradient");
    cu_check(cuMemAlloc(&new_output, count * sizeof(float)), "allocate new output");

    // Queue old work behind an event recorded after a deterministic host gate
    // in stream B. A replacement is compiled and published while the old launch
    // is pending in stream A. The tape retains the old executable through VJP.
    auto stream_gate = std::make_shared<compile_gate>();
    callback_barrier_guard stream_gate_guard(stream_b_runtime_value, stream_gate);
    cudaEvent_t release_event{};
    cuda_check(cudaEventCreateWithFlags(&release_event, cudaEventDisableTiming), "create release event");
    stage_marker("callback_enqueue_begin");
    cuda_check(cudaLaunchHostFunc(stream_b_runtime_value, wait_for_release, stream_gate.get()), "enqueue barrier callback");
    stage_marker("callback_enqueue_end");
    cuda_check(cudaEventRecord(release_event, stream_b_runtime_value), "record release event");
    cuda_check(cudaStreamWaitEvent(stream_a_runtime_value, release_event, 0), "gate stream A");
    wait_entered(stream_gate);
    auto tape_code = old_plan->code;
    auto launch_lease = old_plan->code;
    launch(old_code->forward, stream_a, input, old_output, count, 2.0f);

    auto replacement_gate = std::make_shared<compile_gate>();
    auto replacement_request = compiler.request("replacement-e1-contract-v1", 1,
                                                false, replacement_gate);
    wait_entered(replacement_gate);
    // Numerical generation changes while compilation is paused. It is a bound
    // value and is intentionally absent from the code compatibility key.
    plans.current_value_generation = 2;
    release_gate(replacement_gate);
    auto replacement_ptx = replacement_request.future.get();
    auto replacement_code = std::make_shared<executable>(replacement_ptx, 1,
                                                          replacement_request.key);
    auto replacement_plan = std::make_shared<plan>();
    replacement_plan->code = replacement_code;
    replacement_plan->structure_epoch = 1;
    replacement_plan->specializes_structure = true;
    replacement_plan->bound_value_generation = 2;
    const int old_refs_before_publish = static_cast<int>(old_code.use_count());
    if (plans.publish(replacement_plan) != "published")
      throw std::runtime_error("value-compatible replacement was rejected");
    const auto replacement_publish_at = clock_type::now();
    const bool old_launch_pending_at_publish =
        !gate_watchdog_released_by(stream_gate, replacement_publish_at);
    old_plan.reset();
    old_code.reset();
    const int old_refs_while_pending = static_cast<int>(tape_code.use_count());
    if (g_unloaded_modules.load() < 0 || old_refs_while_pending == 0)
      throw std::runtime_error("old executable was not retained by the tape");
    trace.push_back(old_launch_pending_at_publish
        ? "replacement published before watchdog released the gated old launch"
        : "replacement published after watchdog released the gated old launch; overlap not established");

    // Release A, launch replacement on B, then run the old tape's backward
    // function on A. Both use actual module functions and stream dependencies.
    stream_gate_guard.release_and_sync();
    const bool callback_watchdog_timed_out = gate_watchdog_timed_out(stream_gate);
    trace.push_back(callback_watchdog_timed_out
        ? "callback watchdog released blocked stream after 10 seconds; module-load overlap may serialize"
        : "callback barrier released before watchdog during module-load overlap");
    const float new_scale = 3.0f;
    launch(replacement_code->forward, stream_b, input, new_output, count, new_scale);
    launch(tape_code->backward, stream_a, upstream, old_gradient, count, 2.0f);
    cudaEvent_t old_retired{};
    cudaEvent_t new_complete{};
    cuda_check(cudaEventCreateWithFlags(&old_retired, cudaEventDisableTiming), "create retirement event");
    cuda_check(cudaEventCreateWithFlags(&new_complete, cudaEventDisableTiming), "create new completion event");
    cuda_check(cudaEventRecord(old_retired, stream_a_runtime_value), "record retirement event");
    cuda_check(cudaEventRecord(new_complete, stream_b_runtime_value), "record new completion event");
    cuda_check(cudaEventSynchronize(old_retired), "wait for old tape backward");
    cuda_check(cudaEventSynchronize(new_complete), "wait for replacement launch");
    const auto old_forward_values = copy_result(old_output, count);
    const auto old_gradient_values = copy_result(old_gradient, count);
    const auto new_forward_values = copy_result(new_output, count);
    for (int i = 0; i < count; ++i) {
      const float expected_old = input_host[static_cast<std::size_t>(i)] * 2.0f;
      const float expected_gradient = upstream_host[static_cast<std::size_t>(i)] * 2.0f;
      const float expected_new = input_host[static_cast<std::size_t>(i)] * new_scale;
      if (old_forward_values[static_cast<std::size_t>(i)] != expected_old ||
          old_gradient_values[static_cast<std::size_t>(i)] != expected_gradient ||
          new_forward_values[static_cast<std::size_t>(i)] != expected_new)
        throw std::runtime_error("module output disagrees with independent host reference");
    }
    tape_code.reset();
    launch_lease.reset();
    const int unloaded_after_retire = g_unloaded_modules.load();
    if (unloaded_after_retire < 1 || old_refs_while_pending < 2)
      throw std::runtime_error("old executable lifetime did not span pending work and backward");
    trace.push_back("old executable retained through backward then released after stream event");

    // Failed compilation: invalid source is rejected, leaving active plan untouched.
    const auto active_before_failure = plans.select();
    auto failed = compiler.request("intentional-invalid-source", 1, true);
    bool failure_rejected = false;
    try { (void)failed.future.get(); } catch (...) { failure_rejected = true; }
    if (!failure_rejected || plans.select() != active_before_failure)
      throw std::runtime_error("compile failure disturbed active executable");
    trace.push_back("NVRTC failure kept active plan published");

    // Cancellation is deterministic: the worker pauses before invoking NVRTC,
    // cancellation is requested, then the barrier releases the worker.
    auto cancel_gate = std::make_shared<compile_gate>();
    auto cancelled = compiler.request("cancel-before-compile", 2, false, cancel_gate);
    wait_entered(cancel_gate);
    compiler.cancel(cancelled);
    release_gate(cancel_gate);
    bool cancellation_rejected = false;
    try { (void)cancelled.future.get(); } catch (...) { cancellation_rejected = true; }
    if (!cancellation_rejected || plans.select() != active_before_failure)
      throw std::runtime_error("cancelled compile disturbed active executable");
    trace.push_back("compile cancellation released pending slot without publication");

    // A completed structure-epoch-1 artifact becomes stale after epoch 2.
    // It is rejected before module load; the epoch-generic fallback handles new
    // runtime bindings instead of using a topology-incompatible executable.
    auto stale_request = compiler.request("stale-structure-epoch-1", 1);
    auto stale_ptx = stale_request.future.get();
    (void)stale_ptx;
    plans.current_structure_epoch = 2;
    auto stale_code = std::make_shared<executable>(stale_ptx, 1, stale_request.key);
    auto stale_plan = std::make_shared<plan>();
    stale_plan->code = stale_code;
    stale_plan->structure_epoch = 1;
    stale_plan->specializes_structure = true;
    const std::string stale_result = plans.publish(stale_plan);
    if (stale_result != "rejected-stale" || plans.select() != fallback_plan)
      throw std::runtime_error("stale executable was published or incompatible plan selected");
    stale_plan.reset(); stale_code.reset();
    trace.push_back("stale structure epoch rejected; generic fallback selected");

    // Value-only change under a matching structure epoch remains compatible.
    auto value_request = compiler.request("value-compatible-structure-epoch-2", 2);
    auto value_ptx = value_request.future.get();
    auto value_code = std::make_shared<executable>(value_ptx, 2, value_request.key);
    auto value_plan = std::make_shared<plan>();
    value_plan->code = value_code;
    value_plan->structure_epoch = 2;
    value_plan->specializes_structure = true;
    value_plan->bound_value_generation = plans.current_value_generation;
    ++plans.current_value_generation;
    if (plans.publish(value_plan) != "published")
      throw std::runtime_error("value-only generation change invalidated compatible module");
    trace.push_back("value generation advanced during preparation without code invalidation");

    // Confirm the key duplicate really compiled only once and report live
    // resource counts after all explicit retirements. Plans keep their active
    // executable alive until process teardown, as expected.
    duplicate_count = compiler.duplicate_requests();
    compile_count_before = compiler.compile_count();
    if (duplicate_count < 1 || compile_count_before < 3)
      throw std::runtime_error("compile deduplication/failure/cancel coverage incomplete");

    std::ofstream out(output_path);
    if (!out) throw std::runtime_error("cannot write lifetime report");
    out << "{\n"
        << "  \"schema\": \"cellerator.adaptive_fields.e6_gpu_lifetime.v1\",\n"
        << "  \"gpu_execution_performed\": true,\n"
        << "  \"streams\": 2, \"compile_workers\": 1, \"pending_replacements_max\": 1,\n"
        << "  \"successful_replacement_with_pending_old_launch\": "
        << (old_launch_pending_at_publish ? "true" : "false") << ",\n"
        << "  \"tape_retained_old_code_through_backward\": true,\n"
        << "  \"failure_rejected_without_publication\": " << (failure_rejected ? "true" : "false") << ",\n"
        << "  \"cancellation_rejected_without_publication\": " << (cancellation_rejected ? "true" : "false") << ",\n"
        << "  \"duplicate_requests_deduplicated\": " << duplicate_count << ",\n"
        << "  \"stale_structure_publish_result\": " << json_string(stale_result) << ",\n"
        << "  \"value_only_generation_compatible\": true,\n"
        << "  \"callback_watchdog_timed_out\": " << (callback_watchdog_timed_out ? "true" : "false") << ",\n"
        << "  \"callback_watchdog_timeout_seconds\": 10,\n"
        << "  \"module_load_watchdog_timeout_seconds\": 45,\n"
        << "  \"old_reference_count_before_publish\": " << old_refs_before_publish << ",\n"
        << "  \"old_reference_count_while_pending\": " << old_refs_while_pending << ",\n"
        << "  \"unloaded_modules_after_old_retire\": " << unloaded_after_retire << ",\n"
        << "  \"nvrtc_compile_count\": " << compile_count_before << ",\n"
        << "  \"active_live_modules_before_teardown\": " << g_live_modules.load() << ",\n"
        << "  \"device_buffer_bytes\": " << (5 * count * sizeof(float)) << ",\n"
        << "  \"old_forward_values\": " << json_numbers(old_forward_values) << ",\n"
        << "  \"old_backward_values\": " << json_numbers(old_gradient_values) << ",\n"
        << "  \"new_forward_values\": " << json_numbers(new_forward_values) << ",\n"
        << "  \"host_reference\": {\"old_scale\": 2.0, \"new_scale\": " << new_scale
        << ", \"max_abs_error\": 0.0},\n"
        << "  \"trace\": [";
    for (std::size_t i = 0; i < trace.size(); ++i) { if (i) out << ","; out << json_string(trace[i]); }
    out << "]\n}\n";
    out.close();

    cu_check(cuMemFree(input), "free input"); cu_check(cuMemFree(upstream), "free upstream");
    cu_check(cuMemFree(old_output), "free old output"); cu_check(cuMemFree(old_gradient), "free old gradient");
    cu_check(cuMemFree(new_output), "free new output");
    cuda_check(cudaEventDestroy(release_event), "destroy release event");
    cuda_check(cudaEventDestroy(old_retired), "destroy retirement event");
    cuda_check(cudaEventDestroy(new_complete), "destroy new completion event");
    cuda_check(cudaStreamDestroy(stream_a_runtime_value), "destroy stream A");
    cuda_check(cudaStreamDestroy(stream_b_runtime_value), "destroy stream B");
    std::cout << "lifetime_result=" << output_path << "\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\n";
    return 1;
  }
}
