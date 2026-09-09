#include <Cellerator/execution/program/program_v2.h>
#include <array>
#include <iostream>
#include <stdexcept>
namespace pg=cellerator::execution::program;
void check(bool b,const char* s){if(!b)throw std::runtime_error(s);}
struct op {unsigned index;bool fail=false,reject=false;};
pg::program_status preflight(const void* p,const pg::launch_binding_v2&,void*)noexcept {
    return static_cast<const op*>(p)->reject?pg::program_status::invalid_argument:pg::program_status::success;
}
pg::program_status launch(const void* p,const pg::launch_binding_v2& b,void*)noexcept {
    const auto& o=*static_cast<const op*>(p);static_cast<int*>(b.output)[o.index]=int(o.index)+10;
    return o.fail?pg::program_status::launch_failed:pg::program_status::success;
}
int main()try {
    op ops[3]{{0},{1,true},{2}};pg::prepared_stage_v2 stages[3];
    for(unsigned i=0;i<3;++i){stages[i]={i+1,i+1,&ops[i],launch};stages[i].preflight=preflight;}
    pg::prepared_program_v2 program{2,0,stages,3};std::array<int,3> output{-1,-1,-1};
    pg::launch_binding_v2 binding{};binding.output=output.data();pg::submission_report_v2 report;
    check(pg::execute_prepared_program_report_v2(program,&binding,1,nullptr,report)==pg::program_status::launch_failed,"late failure");
    check(report.attempted_stages==2 && report.accepted_stages==1,"accepted vs attempted");
    check(output==std::array<int,3>{10,11,-1},"partial effects remain and later stage not submitted");
    output.fill(-1);ops[2].reject=true;
    check(pg::execute_prepared_program_report_v2(program,&binding,1,nullptr,report)==pg::program_status::invalid_dynamic_binding,"whole preflight rejection");
    check(report.attempted_stages==0 && report.accepted_stages==0 && output==std::array<int,3>{-1,-1,-1},"preflight no writes");
    ops[2].reject=false;ops[1].fail=false;
    check(pg::execute_prepared_program_report_v2(program,&binding,1,nullptr,report)==pg::program_status::success && report.accepted_stages==3,"all submissions accepted");
    std::cout<<"P05 reports rejected preflight versus partial submission without rollback claims\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
