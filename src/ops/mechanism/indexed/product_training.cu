#include <Cellerator/ops/mechanism/indexed/training.hh>
#include "parameter_identity_validation.hh"
#include "indexed_cuda_support.cuh"
#include "indexed_product_cuda.cuh"

#include <cuda_fp16.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <memory>
#include <mutex>
#include <type_traits>
#include <utility>
#include <vector>

namespace cellerator::ops::mechanism::indexed {
using namespace detail;

struct slot_storage {
    void* x = nullptr;
    float *products = nullptr, *argument_gradients = nullptr, *mechanism_gradients = nullptr;
    std::unique_ptr<event> completion;
    bool busy = false, recorded = false;
};

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
        execution_binding call{impl_->plan, slot.x, slot.products, slot.argument_gradients,
            slot.mechanism_gradients, input,
            mixed ? static_cast<const void*>(parameters()->half_data()) : parameters()->data(),
            output, nullptr, nullptr, batch, input_half, mixed, false};
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
        execution_binding call{impl_->plan, slot.x, slot.products, slot.argument_gradients,
            slot.mechanism_gradients, dy,
            mixed ? static_cast<const void*>(parameters()->half_data()) : parameters()->data(),
            nullptr, dx, dk, t.batch, false, mixed, true};
        execution::program::launch_binding_v2 binding; binding.values = &call;
        require(execution::program::execute_prepared_program_v2(impl_->program,&binding,1,stream) == execution::program::program_status::success, "mechanism VJP launch failed");
        impl_->release(t.slot,t.reader,stream);
    } catch (...) { parameters()->poison(); impl_->release(t.slot,t.reader,stream); throw; }
}
} // namespace cellerator::ops::mechanism::indexed
