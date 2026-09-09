#include <Cellerator/execution/program/provenance_v2.hh>
#include <array>
#include <iostream>
#include <stdexcept>
#include <limits>
namespace pg=cellerator::execution::program;
void check(bool b,const char* s){if(!b)throw std::runtime_error(s);}
pg::program_status add(const void*,const pg::launch_binding_v2& b,void*)noexcept {
    *static_cast<float*>(b.output)=*static_cast<const float*>(b.input)+1;return pg::program_status::success;
}
pg::program_status multiply(const void*,const pg::launch_binding_v2& b,void*)noexcept {
    *static_cast<float*>(b.output)=*static_cast<const float*>(b.input)*2;return pg::program_status::success;
}
pg::program_status fused(const void*,const pg::launch_binding_v2& b,void*)noexcept {
    *static_cast<float*>(b.output)=(*static_cast<const float*>(b.input)+1)*2;return pg::program_status::success;
}
int main()try {
    pg::prepared_stage_v2 separate[2]{{10,100,nullptr,add},{11,101,nullptr,multiply,0,1,1}};
    const std::uint64_t deps[]{0};pg::prepared_program_v2 unfused{2,0,separate,2,deps,1};
    pg::prepared_stage_v2 combined{20,200,nullptr,fused};pg::prepared_program_v2 joint{2,0,&combined,1};
    pg::operation_origin_v2 origins[2]{{{1,1},{9,1},{3},{7}},{{2,1},{9,1},{3},{7}}};
    pg::stage_origins_v2 split_map[2]{{10,100,&origins[0],1},{11,101,&origins[1],1}},fused_map{20,200,origins,2};
    pg::program_origins_v2 a{&unfused,split_map,2},b{&joint,&fused_map,1};
    check(pg::same_origins_v2(a,b),"same originating operations despite fusion and candidate difference");
    float input=3,middle=-1,left=-1,right=-1;
    pg::launch_binding_v2 bindings[2]{{&input,&middle},{&middle,&left}},binding{&input,&right};
    pg::stage_submission_counters_v2 counts[2]{},joint_count{};pg::submission_report_v2 report;
    for(unsigned i=0;i<10;++i) {
        check(pg::execute_prepared_program_report_v2(unfused,bindings,2,nullptr,report,counts,2)==pg::program_status::success,"actual unfused run");
        check(pg::execute_prepared_program_report_v2(joint,&binding,1,nullptr,report,&joint_count,1)==pg::program_status::success,"actual fused run");
    }
    check(left==8 && right==8,"independent numeric oracle");
    check(counts[0].accepted==10 && counts[1].accepted==10 && joint_count.accepted==10,"per-stage counters retain fusion difference");
    check(pg::same_origins_v2(a,b),"executed programs retain same operation origins");
    left=-99;
    check(pg::execute_prepared_program_report_v2(unfused,bindings,2,nullptr,report,counts,1)==pg::program_status::invalid_argument && left==-99 && counts[0].attempted==10,"short counters reject before effects");
    counts[0].attempted=counts[0].accepted=std::numeric_limits<std::uint64_t>::max();
    check(pg::execute_prepared_program_report_v2(unfused,bindings,2,nullptr,report,counts,2)==pg::program_status::success && counts[0].saturated && counts[0].accepted==std::numeric_limits<std::uint64_t>::max(),"counter saturation never wraps");
    auto altered=origins[1];altered.values.value=8;split_map[1].origins=&altered;
    check(!pg::same_origins_v2(a,b),"different source values distinguished");
    altered=origins[1];altered.source_epoch.value=4;check(!pg::same_origins_v2(a,b),"different source structure distinguished");
    split_map[1].origins=&origins[1];split_map[1].candidate_id=999;
    check(!pg::valid_origins_v2(a),"stale selected candidate rejected");
    split_map[1].candidate_id=101;split_map[1].count=0;
    check(!pg::valid_origins_v2(a),"missing stage origins rejected");
    // Decomposition may repeat an origin; provenance equality is an origin set.
    split_map[1].count=2;split_map[1].origins=origins;check(pg::same_origins_v2(a,b),"many-to-many origin retention");
    std::cout<<"P06 real fused/unfused execution, semantic origins/generations and saturated stage counters passed\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
