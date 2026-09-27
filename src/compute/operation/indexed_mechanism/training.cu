#include <Cellerator/compute/operation/indexed_mechanism/training.hh>
#include <Cellerator/memory/session_memory.cuh>
#include <cuda.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>

namespace cellerator::compute::operation::indexed {
namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void check(cudaError_t result) {
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}
struct device_scope {
    int previous = -1;
    explicit device_scope(int device) { check(cudaGetDevice(&previous)); check(cudaSetDevice(device)); }
    ~device_scope() { if (previous >= 0) cudaSetDevice(previous); }
};
struct session_storage {
    runtime::execution_session session{};
    explicit session_storage(int device) {
        runtime::execution_session_options options; options.device = device;
        require(runtime::init_session(&session, options) == runtime::session_status::success,
                "mechanism session initialization failed");
    }
    ~session_storage() { runtime::clear_session(&session); }
    void* reserve(std::size_t bytes) {
        memory::allocation_request request;
        request.bytes = std::max<std::size_t>(bytes, 64);
        request.where = {memory::domain::device, static_cast<std::int16_t>(session.device), -1, 0};
        memory::allocation allocation;
        require(memory::reserve_session_allocation(&session, runtime::persistent_lifetime::graph_stable,
                    request, 1, &allocation) == memory::status::success, "mechanism capacity allocation failed");
        return allocation.base;
    }
};
std::size_t checked_product(std::size_t a, std::size_t b) {
    require(b == 0 || a <= std::numeric_limits<std::size_t>::max() / b, "mechanism capacity overflow");
    return a * b;
}
std::size_t checked_sum(std::size_t a, std::size_t b) {
    require(a <= std::numeric_limits<std::size_t>::max() - b, "mechanism capacity overflow");
    return a + b;
}
std::size_t checked_align(std::size_t n) {
    return checked_sum(n, 63u) & ~std::size_t(63u);
}
std::uint64_t checked_elements(std::uint64_t a, std::uint64_t b) {
    require(b == 0 || a <= std::numeric_limits<std::uint64_t>::max() / b,
            "mechanism element count overflow");
    return a * b;
}
bool overlaps(const void* a, std::size_t a_bytes, const void* b, std::size_t b_bytes) {
    if (!a || !b || !a_bytes || !b_bytes) return false;
    const auto x = reinterpret_cast<std::uintptr_t>(a);
    const auto y = reinterpret_cast<std::uintptr_t>(b);
    const auto maximum = std::numeric_limits<std::uintptr_t>::max();
    if (a_bytes > maximum - x || b_bytes > maximum - y) return true;
    return x < y + b_bytes && y < x + a_bytes;
}
void validate_stream_device(void* stream, int expected) {
    CUcontext current = nullptr;
    require(cuCtxGetCurrent(&current) == CUDA_SUCCESS && current != nullptr,
            "current CUDA context is unavailable");
    CUdevice current_device = -1;
    require(cuCtxGetDevice(&current_device) == CUDA_SUCCESS,
            "current CUDA context device query failed");
    require(static_cast<int>(current_device) == expected,
            "current CUDA context does not match mechanism owner");
    CUcontext stream_context = nullptr;
    require(cuStreamGetCtx(reinterpret_cast<CUstream>(stream), &stream_context) == CUDA_SUCCESS,
            "CUDA stream context query failed");
    require(stream_context == current, "CUDA stream context does not match mechanism owner");
}
struct event {
    cudaEvent_t value = nullptr;
    event() { check(cudaEventCreateWithFlags(&value, cudaEventDisableTiming)); }
    ~event() { if (value) cudaEventDestroy(value); }
    event(const event&) = delete;
    event& operator=(const event&) = delete;
};
__global__ void refresh_half(const float* input, __half* output, std::uint64_t n) {
    for (auto i = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += std::uint64_t(blockDim.x) * gridDim.x)
        output[i] = __float2half_rn(input[i]);
}
unsigned blocks(std::uint64_t count) {
    if (count == 0) return 0;
    return static_cast<unsigned>(std::min<std::uint64_t>(count / 256 + (count % 256 != 0), 65535));
}
struct device_plan {
    std::uint64_t n, o, k, m, a;
    const std::uint64_t *arg_offsets, *args, *coefficient;
    const std::uint64_t *out_offsets, *out_mechanism;
    const float* out_scale;
    const std::uint64_t *input_offsets, *input_slots, *param_offsets, *param_mechanisms;
    const std::uint64_t *mechanism_out_offsets, *mechanism_outputs;
    const float* mechanism_scales;
};
struct slot_storage {
    void* x = nullptr;
    float *products = nullptr, *argument_gradients = nullptr, *mechanism_gradients = nullptr;
    std::unique_ptr<event> completion;
    bool busy = false, recorded = false;
};
__device__ float read_x(const void* x, bool half, std::uint64_t i) {
    return half ? __half2float(static_cast<const __half*>(x)[i]) : static_cast<const float*>(x)[i];
}
__global__ void save_input(const void* input, bool input_half, void* saved, bool mixed, std::uint64_t count) {
    for (auto i = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += std::uint64_t(blockDim.x) * gridDim.x) {
        float value = read_x(input, input_half, i);
        if (mixed) static_cast<__half*>(saved)[i] = __float2half_rn(value);
        else static_cast<float*>(saved)[i] = value;
    }
}
__global__ void products(device_plan p, const void* x, bool mixed, std::uint64_t batch, float* result) {
    for (auto i = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < batch*p.m; i += std::uint64_t(blockDim.x)*gridDim.x) {
        auto m = i % p.m, b = i / p.m;
        float product = 1;
        for (auto a = p.arg_offsets[m]; a < p.arg_offsets[m+1]; ++a) product *= read_x(x, mixed, b*p.n+p.args[a]);
        result[i] = product;
    }
}
__global__ void assemble(device_plan p, const float* products, const void* coefficients, bool mixed,
                         std::uint64_t batch, float* output) {
    for (auto i = std::uint64_t(blockIdx.x)*blockDim.x+threadIdx.x; i < batch*p.o; i += std::uint64_t(blockDim.x)*gridDim.x) {
        auto o = i % p.o, b = i / p.o;
        float sum = 0;
        for (auto j = p.out_offsets[o]; j < p.out_offsets[o+1]; ++j) {
            auto m = p.out_mechanism[j];
            sum += p.out_scale[j] * read_x(coefficients, mixed, p.coefficient[m]) * products[b*p.m+m];
        }
        output[i] = sum;
    }
}
__global__ void local_vjp(device_plan p, const void* x, const void* coefficients, bool mixed,
                          const float* dy, std::uint64_t batch, float* da, float* dm) {
    for (auto i = std::uint64_t(blockIdx.x)*blockDim.x+threadIdx.x; i < batch*p.m; i += std::uint64_t(blockDim.x)*gridDim.x) {
        auto m = i % p.m, b = i / p.m;
        float upstream = 0;
        for (auto j = p.mechanism_out_offsets[m]; j < p.mechanism_out_offsets[m+1]; ++j)
            upstream += p.mechanism_scales[j] * dy[b*p.o+p.mechanism_outputs[j]];
        dm[i] = upstream;
        float prefix = 1;
        for (auto a = p.arg_offsets[m]; a < p.arg_offsets[m+1]; ++a) {
            da[b*p.a+a] = prefix;
            prefix *= read_x(x, mixed, b*p.n+p.args[a]);
        }
        float suffix = 1;
        const float weighted = upstream * read_x(coefficients, mixed, p.coefficient[m]);
        for (auto a = p.arg_offsets[m+1]; a > p.arg_offsets[m];) {
            --a;
            da[b*p.a+a] *= suffix * weighted;
            suffix *= read_x(x, mixed, b*p.n+p.args[a]);
        }
    }
}
__global__ void input_vjp(device_plan p, const float* da, std::uint64_t batch, float* dx) {
    for (auto i = std::uint64_t(blockIdx.x)*blockDim.x+threadIdx.x; i < batch*p.n; i += std::uint64_t(blockDim.x)*gridDim.x) {
        auto n = i % p.n, b = i / p.n;
        float sum = 0;
        for (auto j = p.input_offsets[n]; j < p.input_offsets[n+1]; ++j) sum += da[b*p.a+p.input_slots[j]];
        dx[i] = sum;
    }
}
__global__ void coefficient_vjp(device_plan p, const float* products, const float* dm, std::uint64_t batch, float* dk) {
    for (auto k = std::uint64_t(blockIdx.x)*blockDim.x+threadIdx.x; k < p.k; k += std::uint64_t(blockDim.x)*gridDim.x) {
        float sum = 0;
        for (std::uint64_t b = 0; b < batch; ++b)
            for (auto j = p.param_offsets[k]; j < p.param_offsets[k+1]; ++j) {
                auto m = p.param_mechanisms[j]; sum += dm[b*p.m+m] * products[b*p.m+m];
            }
        dk[k] = sum;
    }
}
struct execution_binding {
    device_plan plan;
    slot_storage* slot;
    const void* input;
    const void* coefficients;
    float* output;
    float* dx;
    float* dk;
    std::uint64_t batch;
    bool input_half, mixed, backward;
};
execution::program::program_status launch_training(const void*, const execution::program::launch_binding_v2& binding, void* stream_ptr) noexcept {
    const auto& b = *static_cast<const execution_binding*>(binding.values);
    auto stream = static_cast<cudaStream_t>(stream_ptr);
    if (!b.backward) {
        save_input<<<blocks(b.batch*b.plan.n),256,0,stream>>>(b.input,b.input_half,b.slot->x,b.mixed,b.batch*b.plan.n);
        products<<<blocks(b.batch*b.plan.m),256,0,stream>>>(b.plan,b.slot->x,b.mixed,b.batch,b.slot->products);
        assemble<<<blocks(b.batch*b.plan.o),256,0,stream>>>(b.plan,b.slot->products,b.coefficients,b.mixed,b.batch,b.output);
    } else {
        local_vjp<<<blocks(b.batch*b.plan.m),256,0,stream>>>(b.plan,b.slot->x,b.coefficients,b.mixed,static_cast<const float*>(b.input),b.batch,b.slot->argument_gradients,b.slot->mechanism_gradients);
        input_vjp<<<blocks(b.batch*b.plan.n),256,0,stream>>>(b.plan,b.slot->argument_gradients,b.batch,b.dx);
        coefficient_vjp<<<blocks(b.plan.k),256,0,stream>>>(b.plan,b.slot->products,b.slot->mechanism_gradients,b.batch,b.dk);
    }
    return cudaGetLastError() == cudaSuccess ? execution::program::program_status::success : execution::program::program_status::launch_failed;
}
} // namespace

struct mechanism_parameter_owner::implementation {
    indexed_axis axis;
    std::vector<identity> ids;
    session_storage storage;
    float* values;
    std::uint16_t* half;
    struct reader { std::unique_ptr<event> done; bool active = false, recorded = false; };
    std::vector<reader> readers;
    event ready;
    event failed_writer_done;
    std::uint64_t generation = 1;
    bool writing = false, poisoned = false;
    cudaStream_t writer_stream = nullptr;
    bool failed_writer_pending = false, failed_writer_recorded = false;
    mutable std::mutex mutex;
    implementation(indexed_axis a, std::vector<identity> i, int device) : axis(a), ids(std::move(i)), storage(device) {}
};
void mechanism_parameter_owner::poison_locked() noexcept {
    if (impl_->writing) {
        int previous = -1;
        const bool have_previous = cudaGetDevice(&previous) == cudaSuccess;
        bool device_ready = have_previous && previous == device();
        if (have_previous && previous != device())
            device_ready = cudaSetDevice(device()) == cudaSuccess;
        impl_->failed_writer_pending = true;
        impl_->failed_writer_recorded = device_ready &&
            cudaEventRecord(impl_->failed_writer_done.value, impl_->writer_stream) == cudaSuccess;
        if (have_previous && previous != device()) (void)cudaSetDevice(previous);
    }
    impl_->poisoned = true;
    impl_->writing = false;
    impl_->writer_stream = nullptr;
}
mechanism_parameter_owner::mechanism_parameter_owner(indexed_axis axis, std::vector<identity> ids,
        std::span<const float> initial, int device, std::uint32_t capacity) {
    device_scope scope(device);
    require(execution::validate_persistent_axis_identity(axis.identity) == execution::biological_validation_code::ok,
            "invalid coefficient axis");
    require(axis.extent && ids.size() == axis.extent && initial.size() == axis.extent && capacity, "invalid coefficient capacity");
    for (std::size_t i = 0; i < ids.size(); ++i) {
        require(v2::valid_stable_id(ids[i]) && std::isfinite(initial[i]), "invalid coefficient identity/value");
        for (std::size_t j = 0; j < i; ++j) require(!v2::same_stable_id(ids[i],ids[j]), "duplicate coefficient identity");
    }
    impl_ = std::make_unique<implementation>(axis,std::move(ids),device);
    auto bytes = checked_product(initial.size(),sizeof(float));
    impl_->values = static_cast<float*>(impl_->storage.reserve(bytes + checked_product(initial.size(),sizeof(std::uint16_t))));
    impl_->half = reinterpret_cast<std::uint16_t*>(impl_->values + initial.size());
    check(cudaMemcpy(impl_->values,initial.data(),bytes,cudaMemcpyHostToDevice));
    refresh_half<<<blocks(initial.size()),256>>>(impl_->values,reinterpret_cast<__half*>(impl_->half),initial.size());
    check(cudaGetLastError()); check(cudaEventRecord(impl_->ready.value)); check(cudaEventSynchronize(impl_->ready.value));
    impl_->readers.resize(capacity);
    for (auto& r : impl_->readers) r.done = std::make_unique<event>();
}
mechanism_parameter_owner::~mechanism_parameter_owner() = default;
float* mechanism_parameter_owner::data() const { return impl_->values; }
const std::uint16_t* mechanism_parameter_owner::half_data() const { return impl_->half; }
std::uint64_t mechanism_parameter_owner::generation() const {
    std::lock_guard lock(impl_->mutex);
    return impl_->generation;
}
std::uint64_t mechanism_parameter_owner::size() const { return impl_->axis.extent; }
int mechanism_parameter_owner::device() const { return impl_->storage.session.device; }
const indexed_axis& mechanism_parameter_owner::axis() const { return impl_->axis; }
const std::vector<identity>& mechanism_parameter_owner::logical_ids() const { return impl_->ids; }
bool mechanism_parameter_owner::poisoned() const {
    std::lock_guard lock(impl_->mutex);
    return impl_->poisoned;
}
void mechanism_parameter_owner::preflight_write() const {
    std::lock_guard lock(impl_->mutex);
    require(!impl_->poisoned && !impl_->writing, "coefficient owner poisoned or writer active");
    require(impl_->generation != std::numeric_limits<std::uint64_t>::max(), "coefficient generation exhausted");
    for (const auto& r : impl_->readers) require(!r.active, "unresolved mechanism forward prevents update");
}
void mechanism_parameter_owner::begin_write(void* stream_ptr) {
    std::lock_guard lock(impl_->mutex); device_scope scope(device());
    validate_stream_device(stream_ptr, device());
    require(!impl_->poisoned && !impl_->writing, "coefficient owner poisoned or writer active");
    require(impl_->generation != std::numeric_limits<std::uint64_t>::max(), "coefficient generation exhausted");
    for (const auto& r : impl_->readers) require(!r.active, "unresolved mechanism forward prevents update");
    auto stream = static_cast<cudaStream_t>(stream_ptr);
    check(cudaStreamWaitEvent(stream,impl_->ready.value));
    for (const auto& r : impl_->readers) if (r.recorded) check(cudaStreamWaitEvent(stream,r.done->value));
    impl_->writer_stream = stream;
    impl_->writing = true;
}
void mechanism_parameter_owner::publish_write(void* stream_ptr) {
    std::lock_guard lock(impl_->mutex); device_scope scope(device());
    require(impl_->writing && !impl_->poisoned, "no sanctioned coefficient writer");
    try {
        require(impl_->generation != std::numeric_limits<std::uint64_t>::max(), "coefficient generation exhausted");
        validate_stream_device(stream_ptr, device());
        auto stream = static_cast<cudaStream_t>(stream_ptr);
        require(stream == impl_->writer_stream, "parameter update must publish on its admission stream");
        refresh_half<<<blocks(size()),256,0,stream>>>(data(),reinterpret_cast<__half*>(impl_->half),size());
        check(cudaGetLastError()); check(cudaEventRecord(impl_->ready.value,stream));
        ++impl_->generation; impl_->writing = false; impl_->writer_stream = nullptr;
        impl_->failed_writer_pending = false; impl_->failed_writer_recorded = false;
    } catch (...) {
        const bool stream_mismatch = static_cast<cudaStream_t>(stream_ptr) != impl_->writer_stream;
        poison_locked();
        // The caller may have issued an unsupported update on the wrong
        // stream before publish detected it. Recovery then synchronizes the
        // whole device because that stream cannot be inferred safely.
        if (stream_mismatch) impl_->failed_writer_recorded = false;
        throw;
    }
}
void mechanism_parameter_owner::poison() noexcept {
    std::lock_guard lock(impl_->mutex);
    poison_locked();
}
void mechanism_parameter_owner::restore(std::span<const float> values, void* stream_ptr) {
    require(values.size() == size(), "checkpoint coefficient count mismatch");
    device_scope scope(device());
    validate_stream_device(stream_ptr, device());
    for (float value : values) require(std::isfinite(value), "nonfinite checkpoint coefficient");
    {
        std::lock_guard lock(impl_->mutex);
        for (const auto& r : impl_->readers)
            require(!r.active, "checkpoint restoration has live tapes");
        require(impl_->generation != std::numeric_limits<std::uint64_t>::max(),
                "coefficient generation exhausted");
        auto stream = static_cast<cudaStream_t>(stream_ptr);
        if (impl_->failed_writer_pending) {
            if (impl_->failed_writer_recorded)
                check(cudaStreamWaitEvent(stream, impl_->failed_writer_done.value));
            else
                check(cudaDeviceSynchronize()); // Cold recovery when event recording itself failed.
        }
        check(cudaStreamWaitEvent(stream, impl_->ready.value));
        for (const auto& r : impl_->readers)
            if (r.recorded) check(cudaStreamWaitEvent(stream, r.done->value));
        impl_->writing = true;
        impl_->writer_stream = stream;
        impl_->poisoned = false;
        impl_->failed_writer_pending = false;
        impl_->failed_writer_recorded = false;
    }
    try {
        check(cudaMemcpyAsync(data(),values.data(),values.size_bytes(),cudaMemcpyHostToDevice,static_cast<cudaStream_t>(stream_ptr)));
        publish_write(stream_ptr);
        check(cudaEventSynchronize(impl_->ready.value)); // cold host checkpoint lifetime
    } catch (...) { poison(); throw; }
}
std::vector<float> mechanism_parameter_owner::snapshot(void* stream_ptr) const {
    std::lock_guard lock(impl_->mutex); device_scope scope(device());
    validate_stream_device(stream_ptr, device());
    require(!impl_->poisoned && !impl_->writing, "cannot checkpoint poisoned or updating owner");
    auto stream = static_cast<cudaStream_t>(stream_ptr);
    check(cudaStreamWaitEvent(stream,impl_->ready.value));
    std::vector<float> values(size());
    check(cudaMemcpyAsync(values.data(),data(),values.size()*sizeof(float),cudaMemcpyDeviceToHost,stream));
    check(cudaStreamSynchronize(stream)); return values;
}
std::uint32_t mechanism_parameter_owner::acquire_reader(void* stream_ptr) {
    std::lock_guard lock(impl_->mutex); device_scope scope(device());
    validate_stream_device(stream_ptr, device());
    require(!impl_->poisoned && !impl_->writing, "coefficient owner unavailable");
    auto stream = static_cast<cudaStream_t>(stream_ptr);
    for (std::uint32_t i = 0; i < impl_->readers.size(); ++i) {
        auto& r = impl_->readers[i]; if (r.active) continue;
        check(cudaStreamWaitEvent(stream,impl_->ready.value));
        if (r.recorded) check(cudaStreamWaitEvent(stream,r.done->value));
        r.active = true; return i;
    }
    throw std::runtime_error("coefficient reader capacity exceeded");
}
void mechanism_parameter_owner::complete_reader(std::uint32_t ticket, void* stream_ptr) noexcept {
    std::lock_guard lock(impl_->mutex);
    if (ticket >= impl_->readers.size() || !impl_->readers[ticket].active) {
        impl_->poisoned = true;
        return;
    }
    auto& r = impl_->readers[ticket];
    if (cudaEventRecord(r.done->value,static_cast<cudaStream_t>(stream_ptr)) != cudaSuccess) impl_->poisoned = true;
    r.recorded = true; r.active = false;
}

struct prepared_mechanism_program::implementation {
    mechanism_declaration declaration;
    std::shared_ptr<mechanism_parameter_owner> parameters;
    session_storage storage;
    device_plan plan{};
    std::vector<slot_storage> slots;
    execution::program::prepared_stage_v2 stage{};
    execution::program::prepared_program_v2 program{};
    std::size_t bytes = 0;
    std::mutex mutex;
    implementation(mechanism_declaration d, std::shared_ptr<mechanism_parameter_owner> p)
        : declaration(std::move(d)), parameters(std::move(p)), storage(parameters->device()) {}
    void release(std::uint32_t slot, std::uint32_t reader, void* stream) noexcept {
        std::lock_guard lock(mutex);
        auto& s = slots[slot];
        if (cudaEventRecord(s.completion->value,static_cast<cudaStream_t>(stream)) != cudaSuccess) parameters->poison();
        s.recorded = true; s.busy = false;
        parameters->complete_reader(reader,stream);
    }
};
struct mechanism_tape::implementation {
    std::shared_ptr<prepared_mechanism_program> program;
    std::uint32_t slot, reader;
    std::uint64_t batch, generation;
    void* stream;
    std::atomic<bool> consumed{false};
};
mechanism_tape::mechanism_tape(std::unique_ptr<implementation> p) : impl_(std::move(p)) {}
mechanism_tape::~mechanism_tape() {
    if (!impl_->consumed.load(std::memory_order_acquire)) {
        try { device_scope scope(impl_->program->parameters()->device());
              impl_->program->impl_->release(impl_->slot,impl_->reader,impl_->stream); }
        catch (...) { impl_->program->parameters()->poison(); }
    }
}
std::uint64_t mechanism_tape::batch() const { return impl_->batch; }

prepared_mechanism_program::prepared_mechanism_program(mechanism_declaration d, std::shared_ptr<mechanism_parameter_owner> p) {
    require(bool(p), "missing coefficient owner"); device_scope scope(p->device());
    require(d.max_batch && d.max_live_forwards && d.max_live_forwards <= 32 && d.input.extent && d.output.extent && !d.mechanisms.empty(), "invalid mechanism capacity");
    require(nf1::same_axis(d.coefficients.identity,p->axis().identity) && d.coefficients.extent == p->size(), "coefficient axis mismatch");
    require(d.coefficient_ids.size() == p->logical_ids().size(), "coefficient identity count mismatch");
    for (std::size_t i = 0; i < d.coefficient_ids.size(); ++i)
        require(v2::same_stable_id(d.coefficient_ids[i],p->logical_ids()[i]), "coefficient order mismatch");
    std::vector<mechanism_incidence> incidence;
    std::vector<std::vector<output_index>> outputs;
    outputs.reserve(d.mechanisms.size()); incidence.reserve(d.mechanisms.size());
    for (const auto& m : d.mechanisms) {
        require(m.coefficient < p->size() && !m.arguments.empty(), "invalid product binding/arity");
        outputs.emplace_back();
        for (const auto& out : m.outputs) {
            require(std::isfinite(out.scale) && out.destination.effect.update == execution::output_update_kind::accumulate,
                    "product output requires finite declared additive assembly");
            outputs.back().push_back(out.destination);
        }
        incidence.push_back({m.instance,{0x4d4c32,1},m.arguments,outputs.back()});
    }
    argument_incidence validated;
    require(validated.prepare(std::span(&d.input,1),std::span(&d.output,1),incidence) == incidence_status::success, "invalid product incidence/assembly/axis");
    for (std::size_t m = 0; m < d.mechanisms.size(); ++m) {
        d.mechanisms[m].arguments = validated.mechanisms()[m].arguments;
        std::sort(d.mechanisms[m].outputs.begin(), d.mechanisms[m].outputs.end(),
                  [](const auto& a, const auto& b) { return a.destination.slot < b.destination.slot; });
    }
    impl_ = std::make_unique<implementation>(std::move(d),std::move(p));
    auto& decl = impl_->declaration; auto& plan = impl_->plan;
    plan.n = decl.input.extent; plan.o = decl.output.extent; plan.k = decl.coefficients.extent; plan.m = decl.mechanisms.size();
    std::vector<std::uint64_t> arg_offsets{0},args,coeff,mo_offsets{0},mo;
    std::vector<float> ms;
    std::vector<std::vector<std::uint64_t>> input(plan.n),param(plan.k),out(plan.o);
    std::vector<std::vector<float>> scales(plan.o);
    for (std::uint64_t m = 0; m < plan.m; ++m) {
        const auto& mech = decl.mechanisms[m]; coeff.push_back(mech.coefficient); param[mech.coefficient].push_back(m);
        for (const auto& a : mech.arguments) { input[a.index].push_back(args.size()); args.push_back(a.index); }
        arg_offsets.push_back(args.size());
        for (const auto& o : mech.outputs) {
            out[o.destination.index].push_back(m); scales[o.destination.index].push_back(o.scale);
            mo.push_back(o.destination.index); ms.push_back(o.scale);
        }
        mo_offsets.push_back(mo.size());
    }
    plan.a = args.size();
    std::vector<std::uint64_t> io{0},is,po{0},pm,oo{0},om;
    std::vector<float> os;
    for (const auto& row : input) { is.insert(is.end(),row.begin(),row.end()); io.push_back(is.size()); }
    for (const auto& row : param) { pm.insert(pm.end(),row.begin(),row.end()); po.push_back(pm.size()); }
    for (std::size_t i = 0; i < out.size(); ++i) { om.insert(om.end(),out[i].begin(),out[i].end()); os.insert(os.end(),scales[i].begin(),scales[i].end()); oo.push_back(om.size()); }
    // One packed immutable allocation, independent of numerical value generation.
    std::size_t metadata = 0;
    auto count = [&](const auto& v) {
        metadata = checked_sum(metadata, checked_align(checked_product(v.size(),sizeof(v[0]))));
    };
    count(arg_offsets); count(args); count(coeff); count(io); count(is); count(po); count(pm); count(oo); count(om); count(os); count(mo_offsets); count(mo); count(ms);
    char* base = static_cast<char*>(impl_->storage.reserve(metadata)); std::size_t offset = 0;
    auto upload = [&](const auto& v, auto& destination) {
        using element = typename std::decay_t<decltype(v)>::value_type;
        destination = reinterpret_cast<const element*>(base+offset);
        if (!v.empty()) check(cudaMemcpy(base+offset,v.data(),v.size()*sizeof(element),cudaMemcpyHostToDevice));
        offset += checked_align(checked_product(v.size(),sizeof(element)));
    };
    upload(arg_offsets,plan.arg_offsets); upload(args,plan.args); upload(coeff,plan.coefficient);
    upload(io,plan.input_offsets); upload(is,plan.input_slots); upload(po,plan.param_offsets); upload(pm,plan.param_mechanisms);
    upload(oo,plan.out_offsets); upload(om,plan.out_mechanism); upload(os,plan.out_scale);
    upload(mo_offsets,plan.mechanism_out_offsets); upload(mo,plan.mechanism_outputs); upload(ms,plan.mechanism_scales);
    auto batch = decl.max_batch;
    const auto max_input_count = checked_elements(batch, plan.n);
    const auto max_output_count = checked_elements(batch, plan.o);
    const auto max_mechanism_count = checked_elements(batch, plan.m);
    const auto max_argument_count = checked_elements(batch, plan.a);
    (void)max_output_count; // validates the runtime output address arithmetic.
    auto xbytes = checked_align(checked_product(static_cast<std::size_t>(max_input_count),decl.precision == training_precision::mixed_f16 ? 2 : 4));
    auto pbytes = checked_align(checked_product(static_cast<std::size_t>(max_mechanism_count),4));
    auto abytes = checked_align(checked_product(static_cast<std::size_t>(max_argument_count),4));
    auto slotbytes = checked_sum(checked_sum(xbytes, checked_product(2, pbytes)), abytes);
    char* workspace = static_cast<char*>(impl_->storage.reserve(checked_product(slotbytes,decl.max_live_forwards)));
    impl_->bytes = checked_sum(metadata, checked_product(slotbytes,decl.max_live_forwards));
    impl_->slots.resize(decl.max_live_forwards);
    for (auto& s : impl_->slots) {
        s.x = workspace; s.products = reinterpret_cast<float*>(workspace+xbytes);
        s.argument_gradients = reinterpret_cast<float*>(workspace+xbytes+pbytes);
        s.mechanism_gradients = reinterpret_cast<float*>(workspace+xbytes+pbytes+abytes);
        s.completion = std::make_unique<event>(); workspace += slotbytes;
    }
    impl_->stage.stable_stage_id = 0x4d4c32; impl_->stage.candidate_id = 1; impl_->stage.launch = launch_training;
    impl_->stage.prepared_state = impl_.get();
    impl_->program.stages = &impl_->stage; impl_->program.stage_count = 1;
    require(execution::program::validate_prepared_program_v2(impl_->program) == execution::program::program_status::success, "invalid prepared mechanism program");
}
prepared_mechanism_program::~prepared_mechanism_program() = default;
const mechanism_declaration& prepared_mechanism_program::declaration() const { return impl_->declaration; }
const std::shared_ptr<mechanism_parameter_owner>& prepared_mechanism_program::parameters() const { return impl_->parameters; }
std::size_t prepared_mechanism_program::reserved_bytes() const { return impl_->bytes; }
std::shared_ptr<mechanism_tape> prepared_mechanism_program::forward(const void* input, bool input_half,
        std::uint64_t batch, const execution::persistent_axis_identity& axis, float* output, void* stream) {
    device_scope scope(parameters()->device()); std::lock_guard lock(impl_->mutex);
    validate_stream_device(stream, parameters()->device());
    require(input && output && batch && batch <= declaration().max_batch, "invalid mechanism launch/capacity");
    require(execution::validate_persistent_axis_identity(axis) == execution::biological_validation_code::ok && nf1::same_axis(axis,declaration().input.identity), "input biological axis mismatch");
    require(declaration().precision == training_precision::mixed_f16 || !input_half, "FP32 program requires FP32 input");
    const auto output_bytes = checked_product(
        static_cast<std::size_t>(checked_elements(batch, declaration().output.extent)), sizeof(float));
    const auto parameter_bytes = checked_product(static_cast<std::size_t>(parameters()->size()), sizeof(float));
    const auto half_bytes = checked_product(static_cast<std::size_t>(parameters()->size()), sizeof(std::uint16_t));
    require(!overlaps(output, output_bytes, parameters()->data(), parameter_bytes) &&
            !overlaps(output, output_bytes, parameters()->half_data(), half_bytes),
            "mechanism output cannot alias canonical or derived coefficients");
    auto self = shared_from_this();
    std::uint32_t index = 0;
    while (index < impl_->slots.size() && impl_->slots[index].busy) ++index;
    require(index < impl_->slots.size(), "mechanism tape capacity exceeded");
    auto& slot = impl_->slots[index];
    if (slot.recorded) check(cudaStreamWaitEvent(static_cast<cudaStream_t>(stream),slot.completion->value));
    auto reader = parameters()->acquire_reader(stream); slot.busy = true;
    try {
        bool mixed = declaration().precision == training_precision::mixed_f16;
        execution_binding call{impl_->plan,&slot,input,mixed ? static_cast<const void*>(parameters()->half_data()) : parameters()->data(),output,nullptr,nullptr,batch,input_half,mixed,false};
        execution::program::launch_binding_v2 binding; binding.values = &call;
        require(execution::program::execute_prepared_program_v2(impl_->program,&binding,1,stream) == execution::program::program_status::success, "mechanism forward launch failed");
        check(cudaEventRecord(slot.completion->value,static_cast<cudaStream_t>(stream))); slot.recorded = true;
        auto tape = std::make_unique<mechanism_tape::implementation>();
        tape->program = std::move(self); tape->slot = index; tape->reader = reader;
        tape->batch = batch; tape->generation = parameters()->generation(); tape->stream = stream;
        return std::shared_ptr<mechanism_tape>(new mechanism_tape(std::move(tape)));
    } catch (...) {
        if (cudaEventRecord(slot.completion->value, static_cast<cudaStream_t>(stream)) != cudaSuccess)
            parameters()->poison();
        else slot.recorded = true;
        parameters()->complete_reader(reader,stream);
        slot.busy = false;
        parameters()->poison();
        throw;
    }
}
void prepared_mechanism_program::backward(mechanism_tape& tape, const float* dy, float* dx, float* dk, void* stream) {
    device_scope scope(parameters()->device());
    validate_stream_device(stream, parameters()->device());
    auto& t = *tape.impl_;
    require(t.program.get() == this, "foreign mechanism tape");
    require(!parameters()->poisoned() && t.generation == parameters()->generation(), "stale mechanism coefficient generation");
    require(dy && dx && dk, "missing VJP buffers");
    const auto dx_bytes = checked_product(static_cast<std::size_t>(checked_elements(t.batch, declaration().input.extent)), sizeof(float));
    const auto dk_bytes = checked_product(static_cast<std::size_t>(parameters()->size()), sizeof(float));
    const auto parameter_bytes = dk_bytes;
    const auto half_bytes = checked_product(static_cast<std::size_t>(parameters()->size()), sizeof(std::uint16_t));
    require(!overlaps(dx, dx_bytes, dk, dk_bytes) &&
            !overlaps(dx, dx_bytes, parameters()->data(), parameter_bytes) &&
            !overlaps(dx, dx_bytes, parameters()->half_data(), half_bytes) &&
            !overlaps(dk, dk_bytes, parameters()->data(), parameter_bytes) &&
            !overlaps(dk, dk_bytes, parameters()->half_data(), half_bytes),
            "input and coefficient gradients must not alias each other or coefficient storage");
    bool expected = false;
    require(t.consumed.compare_exchange_strong(expected, true, std::memory_order_acq_rel),
            "mechanism backward replay");
    auto& slot = impl_->slots[t.slot];
    try {
        check(cudaStreamWaitEvent(static_cast<cudaStream_t>(stream),slot.completion->value));
        bool mixed = declaration().precision == training_precision::mixed_f16;
        execution_binding call{impl_->plan,&slot,dy,mixed ? static_cast<const void*>(parameters()->half_data()) : parameters()->data(),nullptr,dx,dk,t.batch,false,mixed,true};
        execution::program::launch_binding_v2 binding; binding.values = &call;
        require(execution::program::execute_prepared_program_v2(impl_->program,&binding,1,stream) == execution::program::program_status::success, "mechanism VJP launch failed");
        impl_->release(t.slot,t.reader,stream);
    } catch (...) { parameters()->poison(); impl_->release(t.slot,t.reader,stream); throw; }
}
} // namespace cellerator::compute::operation::indexed
