#include <Cellerator/execution/program/session_v2.hh>
#include <algorithm>
#include <Cellerator/execution/launch_bindings.hh>
#include <limits>

namespace cellerator::execution::program {
namespace {
instance_status readiness(runtime::relation_readiness_status status) noexcept {
    using code=runtime::relation_readiness_status;
    if (status==code::success) return instance_status::success;
    if (status==code::busy) return instance_status::busy;
    if (status==code::device_mismatch) return instance_status::device_mismatch;
    if (status==code::cuda_failure) return instance_status::runtime_failure;
    return instance_status::invalid_state;
}
bool overlaps(const runtime::launch_runtime_binding& a,const runtime::launch_runtime_binding& b) noexcept {
    if (!a.workspace_bytes || !b.workspace_bytes) return false;
    const auto x=reinterpret_cast<std::uintptr_t>(a.workspace),y=reinterpret_cast<std::uintptr_t>(b.workspace);
    const auto max=std::numeric_limits<std::uintptr_t>::max();
    return a.workspace_bytes>max-x || b.workspace_bytes>max-y ||
        (x<y+b.workspace_bytes && y<x+a.workspace_bytes);
}
}
program_session_v2::~program_session_v2() { (void)close(); }
instance_status program_session_v2::context(std::uint32_t index) const noexcept {
    if (!session_ || !program_ || session_->exclusive_attachment!=this || !session_->initialized || !session_->sealed || index>=instance_count_ || session_->stream_count!=instance_count_ ||
        session_->stream_count>bindings_.size()) return instance_status::invalid_state;
    int current=-1;
    if (cudaGetDevice(&current)!=cudaSuccess) return instance_status::runtime_failure;
    if (current!=session_->device || current!=bindings_[index].execution.device) return instance_status::device_mismatch;
    const auto& slot=session_->streams[index];
    if (slot.execution.stream!=bindings_[index].execution.stream ||
        slot.transient.data!=bindings_[index].workspace || slot.transient.bytes<workspace_bytes_)
        return instance_status::invalid_binding;
    return instance_status::success;
}
instance_status program_session_v2::initialize(runtime::execution_session& session,
        const prepared_program_v2& program,execution::structure_id identity,execution::structure_epoch epoch) noexcept {
    guard lock(active_);if (!lock.acquired) return instance_status::busy;
    if (session_ || !session.initialized || !session.sealed || !session.stream_count ||
        session.stream_count>bindings_.size() || !execution::valid_identity(identity) || !epoch.value)
        return instance_status::invalid_state;
    if (validate_prepared_program_v2(program)!=program_status::success) return instance_status::invalid_binding;
    int current=-1;if (cudaGetDevice(&current)!=cudaSuccess) return instance_status::runtime_failure;
    if (current!=session.device) return instance_status::device_mismatch;
    std::uint64_t bytes=0;
    for (std::uint64_t i=0;i<program.stage_count;++i) {
        bytes=std::max(bytes,program.stages[i].required_workspace_bytes);
        if (program.stages[i].binding_contract)
            bytes=std::max(bytes,program.stages[i].binding_contract->workspace.minimum_bytes);
    }
    for (std::uint32_t i=0;i<session.stream_count;++i) {
        bindings_[i]=runtime::bind_launch(&session,i,bytes);
        if (bindings_[i].status!=runtime::session_status::success || (bytes && !bindings_[i].workspace))
            return instance_status::insufficient_workspace;
        for (std::uint32_t j=0;j<i;++j)
            if (bindings_[i].execution.stream==bindings_[j].execution.stream || overlaps(bindings_[i],bindings_[j]))
                return instance_status::invalid_binding;
    }
    if (runtime::attach_session(&session,this)!=runtime::session_status::success)
        return instance_status::invalid_state;
    session_=&session;instance_count_=session.stream_count;
    for (std::uint32_t i=0;i<session.stream_count;++i) {
        auto result=readiness_[i].initialize(identity,epoch,session.device,bindings_[i].execution.stream);
        if (result!=runtime::relation_readiness_status::success) {
            bool drained=true;
            for (std::uint32_t j=0;j<=i;++j)
                if (readiness_[j].close()!=runtime::relation_readiness_status::success) drained=false;
            if (drained) {
                (void)runtime::detach_session(&session,this);
                session_=nullptr;
            }
            // Failed cleanup retains attachment for checked close retry.
            return readiness(result);
        }
    }
    session_=&session;program_=&program;identity_=identity;epoch_=epoch;workspace_bytes_=bytes;instance_count_=session.stream_count;
    return instance_status::success;
}
instance_status program_session_v2::binding(std::uint32_t index,runtime::launch_runtime_binding& output) noexcept {
    guard lock(active_);if (!lock.acquired) return instance_status::busy;
    auto result=context(index);if (result!=instance_status::success) return result;
    output=bindings_[index];return instance_status::success;
}
instance_status program_session_v2::execute(std::uint32_t index,const launch_binding_v2* bindings,
        std::uint64_t count,execution::value_generation next) noexcept {
    guard lock(active_);if (!lock.acquired) return instance_status::busy;
    auto result=context(index);if (result!=instance_status::success) return result;
    const auto& bound=bindings_[index];
    if (count && !bindings) return instance_status::invalid_binding;
    const auto base=reinterpret_cast<std::uintptr_t>(bound.workspace);
    for (std::uint64_t i=0;i<count;++i) {
        const auto address=reinterpret_cast<std::uintptr_t>(bindings[i].workspace);
        // Prepared stage slices may reuse the owning instance arena. Never
        // accept a foreign range, wraparound, or bytes past the reserved end.
        if (address<base || address-base>bound.workspace_bytes ||
            bindings[i].workspace_bytes>bound.workspace_bytes-(address-base))
            return instance_status::invalid_binding;
    }
    auto& ready=readiness_[index];
    result=readiness(ready.validate_write(ready.generation(),next,bound.execution.stream));
    if (result!=instance_status::success) return result;
    if (preflight_prepared_program_v2(*program_,bindings,count,bound.execution.stream)!=program_status::success)
        return instance_status::preflight_rejected;
    const auto launched=execute_prepared_program_v2(*program_,bindings,count,bound.execution.stream);
    result=readiness(ready.publish(next,bound.execution.stream,
        launched==program_status::success?cudaSuccess:cudaErrorUnknown));
    return launched!=program_status::success?instance_status::launch_failed:result;
}
instance_status program_session_v2::begin_read(std::uint32_t index,execution::value_generation generation,
        cudaStream_t consumer,runtime::relation_read_ticket& ticket) noexcept {
    guard lock(active_);if (!lock.acquired) return instance_status::busy;
    auto result=context(index);if (result!=instance_status::success) return result;
    return readiness(readiness_[index].begin_read(identity_,epoch_,generation,session_->device,consumer,&ticket));
}
instance_status program_session_v2::end_read(std::uint32_t index,runtime::relation_read_ticket& ticket,cudaStream_t consumer) noexcept {
    guard lock(active_);if (!lock.acquired) return instance_status::busy;
    auto result=context(index);if (result!=instance_status::success) return result;
    return readiness(readiness_[index].end_read(ticket,consumer));
}
instance_status program_session_v2::close() noexcept {
    guard lock(active_);if (!lock.acquired) return instance_status::busy;
    if (!session_) return instance_status::success;
    for (std::uint32_t i=0;i<instance_count_;++i)
        if (readiness_[i].active_reader()) return instance_status::busy;
    for (std::uint32_t i=0;i<instance_count_;++i) {
        auto result=readiness(readiness_[i].close());
        if (result!=instance_status::success) return result;
    }
    if (runtime::detach_session(session_,this)!=runtime::session_status::success)
        return instance_status::invalid_state;
    session_=nullptr;program_=nullptr;return instance_status::success;
}
} // namespace cellerator::execution::program
