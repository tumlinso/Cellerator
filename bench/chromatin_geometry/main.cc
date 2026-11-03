#include "body_runner.hh"
#include "campaign_runner.hh"
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--help") {
        std::puts("cgs_study --mode body-check|campaign --case CASE --backend scalar|bitmap|direct|masked-direct-two-tile|cohort-direct|cohort-direct-two-tile|cohort-shared|library|library-batched --gate supplied|ste|teacher --body product|linear --precision fp16|fp32 [--tile 32|64|128] [--stages 1|2] [--replay CGATAC_PATH] [--condition-update none|rebuild|incremental] [--warmups N] [--repeats N] [--output PATH]\nbody-check reports diagnostic phase timings; campaign reports its specified cost boundary and validation status.");
        return 0;
    }
    try {
        cgs::BodyRunOptions o; std::string mode,path,backend="scalar"; unsigned tile=32,stages=1;
        for(int i=1;i<argc;i+=2){
            if(i+1>=argc) throw std::runtime_error("missing option value");
            std::string key=argv[i], value=argv[i+1];
            if(key=="--mode")mode=value;
            else if(key=="--case")o.case_id=value;
            else if(key=="--replay")o.replay_path=value;
            else if(key=="--condition-update")o.condition_update=value;
            else if(key=="--output")path=value;
            else if(key=="--warmups")o.warmups=std::stoul(value);
            else if(key=="--repeats")o.repeats=std::stoul(value);
            else if(key=="--tile")tile=std::stoul(value);
            else if(key=="--stages")stages=std::stoul(value);
            else if(key=="--backend"){
                backend=value;
                if(value=="scalar")o.backend=cgs::Backend::scalar;else if(value=="bitmap")o.backend=cgs::Backend::bitmap;
                else if(value=="direct")o.backend=cgs::Backend::direct;else if(value=="cohort-direct")o.backend=cgs::Backend::cohort_direct;
                else if(value=="cohort-direct-two-tile")o.backend=cgs::Backend::cohort_direct_two_tile;
                else if(value=="masked-direct-two-tile")o.backend=cgs::Backend::masked_direct_two_tile;
                else if(value=="cohort-shared")o.backend=cgs::Backend::cohort_shared;else if(value!="library"&&value!="library-batched")throw std::runtime_error("unknown backend");
            } else if(key=="--gate"){
                if(value=="supplied")o.gate=cgs::GateProfile::supplied;else if(value=="ste")o.gate=cgs::GateProfile::ste;
                else if(value=="teacher")o.gate=cgs::GateProfile::teacher;else throw std::runtime_error("unknown gate");
            } else if(key=="--body"){
                if(value=="product")o.body=cgs::BodyKind::ordered_product;else if(value=="linear")o.body=cgs::BodyKind::grouped_linear;else throw std::runtime_error("unknown body");
            } else if(key=="--precision"){
                if(value=="fp16")o.dtype=cgs::DType::fp16;else if(value=="fp32")o.dtype=cgs::DType::fp32;else throw std::runtime_error("unknown precision");
            } else throw std::runtime_error("unknown option: "+key);
        }
        if(o.case_id.empty()) throw std::runtime_error("--case is required");
        if(mode=="campaign"){
            cgs::CampaignOptions c; c.case_id=o.case_id;c.replay_path=o.replay_path;c.backend=backend;c.condition_update=o.condition_update;
            c.gate=o.gate;c.body=o.body;c.dtype=o.dtype;c.cell_tile=tile;c.stage_count=stages;c.warmups=o.warmups;c.repeats=o.repeats;
            return cgs::run_campaign(c,path);
        }
        if(mode!="body-check") throw std::runtime_error("unknown mode");
        if(backend=="library"||backend=="library-batched"||tile!=32||stages!=1)throw std::runtime_error("unsupported body-check option");
        return cgs::run_body_check(o,path);
    } catch(const std::exception& e){std::fprintf(stderr,"cgs_study: %s\n",e.what());return 2;}
}
