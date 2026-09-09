#include <Cellerator/execution/program/workspace_v2.hh>
#include <algorithm>
#include <Cellerator/execution/launch_bindings.hh>
#include <limits>
namespace cellerator::execution::program {
namespace {
constexpr auto maximum=std::numeric_limits<std::uint64_t>::max();
bool align(std::uint64_t value,std::uint64_t alignment,std::uint64_t& result) {
    if (value>maximum-(alignment-1)) return false;
    result=(value+alignment-1)&~(alignment-1);return true;
}
}
workspace_status prepare_workspace_v2(const scratch_lifetime_v2* requests,std::uint64_t count,
        std::uint64_t stages,workspace_plan_v2& output) noexcept {
    if (!stages || (count&&!requests)) return workspace_status::invalid_argument;
    workspace_plan_v2 candidate;candidate.stage_count=stages;
    try {
        if (count>candidate.slots.max_size()) return workspace_status::overflow;
        candidate.slots.reserve(count);
        for(std::uint64_t i=0;i<count;++i) {
            const auto& r=requests[i];
            if (!r.alignment || (r.alignment&(r.alignment-1)) || r.first>r.last || r.last>=stages)
                return workspace_status::invalid_argument;
            candidate.alignment=std::max(candidate.alignment,r.alignment);
            std::uint64_t offset=0;
            // Deterministic first fit among conflicting inclusive lifetimes.
            bool retry=true;
            while(retry) {
                retry=false;
                if (!align(offset,r.alignment,offset) || r.bytes>maximum-offset) return workspace_status::overflow;
                for(const auto& s:candidate.slots) {
                    if (!r.bytes || !s.lifetime.bytes || r.last<s.lifetime.first || s.lifetime.last<r.first) continue;
                    const auto end=s.offset+s.lifetime.bytes;
                    if (offset<end && s.offset<offset+r.bytes) { offset=end;retry=true;break; }
                }
            }
            candidate.slots.push_back({r,offset});
            candidate.bytes=std::max(candidate.bytes,offset+r.bytes);
        }
    } catch(...) { return workspace_status::allocation_failure; }
    output=std::move(candidate);return workspace_status::success;
}
workspace_status prepare_program_workspace_v2(const prepared_program_v2& program,
        const scratch_lifetime_v2* extra,std::uint64_t count,workspace_plan_v2& output) noexcept {
    if (validate_prepared_program_v2(program)!=program_status::success || !program.stage_count ||
        (count&&!extra) || count>maximum-program.stage_count) return workspace_status::invalid_argument;
    try {
        std::vector<scratch_lifetime_v2> requests;
        if (count+program.stage_count>requests.max_size()) return workspace_status::overflow;
        requests.reserve(count+program.stage_count);
        for(std::uint64_t i=0;i<program.stage_count;++i) {
            const auto& stage=program.stages[i];
            auto bytes=stage.required_workspace_bytes;
            std::uint64_t alignment=1;
            if (stage.binding_contract) {
                bytes=std::max(bytes,stage.binding_contract->workspace.minimum_bytes);
                alignment=stage.binding_contract->workspace.alignment;
            }
            requests.push_back({bytes,alignment,i,i});
        }
        for(std::uint64_t i=0;i<count;++i) requests.push_back(extra[i]);
        return prepare_workspace_v2(requests.data(),requests.size(),program.stage_count,output);
    } catch(...) { return workspace_status::allocation_failure; }
}
workspace_status workspace_slice_v2(const workspace_plan_v2& plan,std::uint64_t index,
        void* storage,std::uint64_t capacity,void*& output,std::uint64_t& bytes) noexcept {
    if (index>=plan.slots.size() || !plan.alignment || (plan.alignment&(plan.alignment-1))) return workspace_status::invalid_argument;
    const auto base=reinterpret_cast<std::uintptr_t>(storage);
    const auto& slot=plan.slots[index];
    if ((plan.bytes&&!storage) || base%plan.alignment || plan.bytes>capacity ||
        capacity>std::numeric_limits<std::uintptr_t>::max()-base || slot.offset>plan.bytes ||
        slot.lifetime.bytes>plan.bytes-slot.offset) return workspace_status::insufficient_capacity;
    output=slot.lifetime.bytes?reinterpret_cast<void*>(base+slot.offset):nullptr;
    bytes=slot.lifetime.bytes;return workspace_status::success;
}
workspace_status bind_state_ping_pong_v2(void* first,std::uint64_t first_capacity,void* second,
        std::uint64_t second_capacity,std::uint64_t required,state_ping_pong_v2& output) noexcept {
    const auto a=reinterpret_cast<std::uintptr_t>(first), b=reinterpret_cast<std::uintptr_t>(second);
    constexpr auto limit=std::numeric_limits<std::uintptr_t>::max();
    if (!required || !first || !second || required>first_capacity || required>second_capacity ||
        first_capacity>limit-a || second_capacity>limit-b) return workspace_status::insufficient_capacity;
    if(a<b+second_capacity && b<a+first_capacity) return workspace_status::invalid_argument;
    output={{first,second},required,0};return workspace_status::success;
}
} // namespace cellerator::execution::program
