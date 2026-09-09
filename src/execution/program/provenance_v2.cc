#include <Cellerator/execution/program/provenance_v2.hh>
namespace cellerator::execution::program {
namespace {
bool equal(compute::operation::v2::stable_id a,compute::operation::v2::stable_id b) {
    return a.low==b.low && a.high==b.high;
}
bool same(const operation_origin_v2& a,const operation_origin_v2& b) {
    return equal(a.operation,b.operation) && equal(a.source,b.source) &&
        a.source_epoch.value==b.source_epoch.value && a.values.value==b.values.value;
}
bool subset(const program_origins_v2& a,const program_origins_v2& b) {
    for(std::uint64_t i=0;i<a.count;++i) for(std::uint64_t j=0;j<a.stages[i].count;++j) {
        bool found=false;
        for(std::uint64_t k=0;k<b.count && !found;++k)
            for(std::uint64_t l=0;l<b.stages[k].count && !found;++l)
                found=same(a.stages[i].origins[j],b.stages[k].origins[l]);
        if(!found)return false;
    }
    return true;
}
}
bool valid_origins_v2(const program_origins_v2& map) noexcept {
    if(!map.program || validate_prepared_program_v2(*map.program)!=program_status::success ||
        map.count!=map.program->stage_count || (map.count&&!map.stages))return false;
    for(std::uint64_t i=0;i<map.count;++i) {
        const auto& entry=map.stages[i];const auto& stage=map.program->stages[i];
        if(entry.stage_id!=stage.stable_stage_id || entry.candidate_id!=stage.candidate_id ||
            !entry.count || !entry.origins)return false;
        for(std::uint64_t j=0;j<entry.count;++j) {
            const auto& origin=entry.origins[j];
            if(equal(origin.operation,{}) ||
                equal(origin.source,{}) ||
                !origin.source_epoch.value || !origin.values.value)return false;
        }
    }
    return true;
}
bool same_origins_v2(const program_origins_v2& a,const program_origins_v2& b) noexcept {
    return valid_origins_v2(a) && valid_origins_v2(b) && subset(a,b) && subset(b,a);
}
} // namespace cellerator::execution::program
