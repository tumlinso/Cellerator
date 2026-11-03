#include "body_runner.hh"
#include "device_fixture.hh"
#include "reference/reference.hh"
#include "learning/ste_chain.hh"
#include "learning/model_update.hh"
#include "backends/conditions.hh"
#include "teacher_fixture.hh"
#include "data/replay.hh"
#include "workloads.hh"
#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>

namespace cgs {
namespace {
struct Stream { cudaStream_t value{}; Stream(){cuda_check(cudaStreamCreate(&value),"stream");} ~Stream(){cudaStreamDestroy(value);} };
struct Event { cudaEvent_t value{}; Event(){cuda_check(cudaEventCreate(&value),"event");} ~Event(){cudaEventDestroy(value);} };
float max_error(const std::vector<float>& a,const std::vector<float>& b) {
 if(a.size()!=b.size()) throw std::runtime_error("result/reference shape mismatch");
 float e=0; for(std::size_t i=0;i<a.size();++i) {if(!std::isfinite(a[i])||!std::isfinite(b[i])) throw std::runtime_error("nonfinite result/reference"); e=std::max(e,std::fabs(a[i]-b[i]));} return e;
}
void check_product(const std::vector<float>& a,const std::vector<float>& b,const char* label){
 if(a.size()!=b.size())throw std::runtime_error(std::string(label)+" shape mismatch");
 for(std::size_t i=0;i<a.size();++i)if(!std::isfinite(a[i])||std::fabs(a[i]-b[i])>reference::product_abs_tolerance*(1.f+std::fabs(b[i])))throw std::runtime_error(std::string(label)+" outside product tolerance");
}
void check_bounds(const std::vector<float>& a,const std::vector<float>& b,const std::vector<float>& limits,const char* label){
 if(a.size()!=b.size()||a.size()!=limits.size())throw std::runtime_error(std::string(label)+" bound shape mismatch");
 for(std::size_t i=0;i<a.size();++i)if(!reference::within_absolute_bound(a[i],b[i],limits[i]))throw std::runtime_error(std::string(label)+" outside frozen compound bound");
}
const char* backend_name(Backend b){switch(b){case Backend::scalar:return "scalar";case Backend::bitmap:return "bitmap";case Backend::direct:return "direct";case Backend::cohort_direct:return "cohort-direct";case Backend::cohort_shared:return "cohort-shared";case Backend::cohort_direct_two_tile:return "cohort-direct-two-tile";case Backend::masked_direct_two_tile:return "masked-direct-two-tile";}return "invalid";}
}
int run_body_check(const BodyRunOptions& o,const std::string& path) {
 try {
  const auto& specs=workload_manifest();
  const auto it=std::find_if(specs.begin(),specs.end(),[&](const auto& s){return s.id==o.case_id;});
  if(it==specs.end()) throw std::runtime_error("unknown workload case");
  if(o.repeats==0 || o.repeats>10000 || o.warmups>10000) throw std::runtime_error("invalid repeat count");
  if(o.condition_update!="none" && o.condition_update!="rebuild" && o.condition_update!="incremental")throw std::runtime_error("unknown condition update mode");
  if(o.gate==GateProfile::ste && o.condition_update!="none")throw std::runtime_error("condition updates do not generate STE scores");
  using Clock=std::chrono::steady_clock;
  const auto fixture_start=Clock::now();
  auto workload=materialize_workload(*it);
  if(!o.replay_path.empty())apply_atac_replay(workload,o.replay_path);
  auto expected_workload=workload;
  if(o.condition_update!="none")expected_workload.predicates_before=workload.predicates_after;
  auto host=make_consumer_fixture(expected_workload,o.body,o.dtype,o.gate);
  const auto fixture_end=Clock::now();
  auto hp=host.problem(); auto hg=host.gates();
  std::vector<float> ey,edx,edtheta,edweights,edq; reference::LinearValidationBounds linear_bounds;
  if(o.body==BodyKind::ordered_product){
    auto f=reference::product_forward(hp,hg); auto b=reference::product_backward(hp,hg,host.dy.data());
    ey=std::move(f.y); edx=std::move(b.dx); edtheta=std::move(b.dtheta); edq=std::move(b.dq);
  } else {
    auto f=reference::grouped_linear_forward(hp,hg); auto b=reference::grouped_linear_backward(hp,hg,host.dy.data(),f);
    linear_bounds=reference::grouped_linear_validation_bounds(hp,hg,host.dy.data(),f,b);
    ey=std::move(f.y); edx=std::move(b.dx); edweights=std::move(b.dweights); edq=std::move(b.dq);
  }
  const auto reference_end=Clock::now();
  DeviceFixture d(host); Stream stream; BackendOptions opt{}; std::size_t bytes=0;
  std::unique_ptr<ConditionDeviceFixture> conditions;
  std::unique_ptr<TeacherFixture> teacher;
  if(o.gate!=GateProfile::ste)conditions=std::make_unique<ConditionDeviceFixture>(workload,stream.value);
  if(o.gate==GateProfile::teacher)teacher=std::make_unique<TeacherFixture>(d.p.batch,d.p.topology.conditions,o.backend==Backend::scalar);
  const auto producer=o.backend==Backend::scalar?ConditionProducer::root_mapped:ConditionProducer::bitplanes;
  const bool produce_condition_bytes=o.backend==Backend::scalar && o.gate==GateProfile::supplied;
  const bool request_condition_planes=o.backend==Backend::scalar && o.gate==GateProfile::teacher;
  std::vector<float> initial_scores(o.gate==GateProfile::ste?host.admitted.size():0);
  for(std::size_t i=0;i<initial_scores.size();++i) initial_scores[i]=host.admitted[i]?std::log(3.f):-std::log(3.f);
  DeviceBuffer<float> scores(initial_scores),q(initial_scores.size()),ds(initial_scores.size());
  SteChain chain{};
  SteChainBuffers sb{};
  sb.gates={scores.ptr,nullptr,d.probs.ptr,q.ptr,d.admitted.ptr,d.primal.ptr,d.derivative.ptr,d.status.ptr};
  sb.dq=d.dq.ptr; sb.ds=ds.ptr; sb.parameters=scores.ptr;
  if(!workspace_bytes(o.backend,o.body,d.p,opt,&bytes)) throw std::runtime_error("unsupported workspace/shape");
  DeviceBuffer<std::uint8_t> memory(bytes); WorkspaceView ws{memory.ptr,bytes,d.status.ptr};
  const auto transfer_end=Clock::now();
  d.clear(stream.value);
  if(o.gate==GateProfile::supplied || o.gate==GateProfile::teacher){
    cuda_check(conditions->rebuild(producer,false,o.gate,produce_condition_bytes,stream.value,request_condition_planes),"condition rebuild");
    if(o.condition_update=="rebuild")cuda_check(conditions->rebuild(producer,true,o.gate,produce_condition_bytes,stream.value,request_condition_planes),"condition full rebuild");
    if(o.condition_update=="incremental")cuda_check(conditions->update(producer,o.gate,produce_condition_bytes,stream.value,request_condition_planes),"condition incremental update");
    if(conditions->status(stream.value))throw std::runtime_error("condition producer failed");
    d.g=conditions->gates(o.gate);
    if(o.gate==GateProfile::teacher)d.g=teacher->enqueue(d.g,d.status.ptr,d.g.revisions.parameter_revision,it->parameter_pattern==ParameterPattern::update_every_step,stream.value,{},o.backend==Backend::scalar);
  }
  if(o.gate==GateProfile::ste){
    cuda_check(ste_chain_forward(d.p.batch,d.p.topology.conditions,1.f,0.5f,d.g.revisions,sb,&chain,{},stream.value),"STE forward");
    d.g=chain.gates;
  }
  cuda_check(prepare(o.backend,o.body,d.p,d.g,opt,ws,stream.value),"prepare");
  cuda_check(forward(o.backend,o.body,d.p,d.g,opt,d.result,ws,stream.value),"forward");
  cuda_check(backward(o.backend,o.body,d.p,d.g,d.dy.ptr,opt,d.result,ws,stream.value),"backward");
  const bool update_model=it->parameter_pattern==ParameterPattern::update_every_step;
  const float score_rate=update_model?0.01f:0.f;
  if(o.gate==GateProfile::ste) cuda_check(ste_chain_backward_update(&chain,d.g.revisions,d.result,ws,score_rate,{},stream.value),"STE update");
  std::uint64_t parameter_revision=d.g.revisions.parameter_revision;
  if(update_model){
    auto pointer=o.body==BodyKind::ordered_product?static_cast<void*>(d.theta.ptr):const_cast<void*>(d.p.weights);
    auto gradient=o.body==BodyKind::ordered_product?d.dtheta.ptr:d.dweights.ptr;
    auto count=o.body==BodyKind::ordered_product?d.p.topology.parameters:d.p.weight_count;
    auto dtype=o.body==BodyKind::ordered_product?DType::fp32:o.dtype;
    cuda_check(model_sgd(pointer,gradient,count,dtype,0.01f,d.status.ptr,stream.value),"model update");
    ++parameter_revision;
  }
  cuda_check(cudaStreamSynchronize(stream.value),"synchronize");
  auto flags=d.status.download(); if(flags[0]) throw std::runtime_error("device status="+std::to_string(flags[0]));
  float teacher_uniform_soft_hard=0;
  if(o.gate==GateProfile::teacher){auto changes=teacher->changes.download();for(float v:changes)teacher_uniform_soft_hard+=v;if(!changes.empty())teacher_uniform_soft_hard/=changes.size();}
  std::vector<std::uint32_t> actual_p(host.primal.size()),actual_d(host.derivative.size());
  std::vector<std::uint8_t> actual_admission(host.admitted.size());
  const bool has_planes=d.g.primal_words&&d.g.derivative_words;
  if(has_planes){cuda_check(cudaMemcpy(actual_p.data(),d.g.primal_words,actual_p.size()*sizeof(std::uint32_t),cudaMemcpyDeviceToHost),"copy primal gates");
  cuda_check(cudaMemcpy(actual_d.data(),d.g.derivative_words,actual_d.size()*sizeof(std::uint32_t),cudaMemcpyDeviceToHost),"copy derivative gates");}
  const bool has_admission=d.g.admitted!=nullptr;
  if(has_admission)cuda_check(cudaMemcpy(actual_admission.data(),d.g.admitted,actual_admission.size(),cudaMemcpyDeviceToHost),"copy admission");
  if((has_planes && (actual_p!=host.primal || actual_d!=host.derivative)) || (has_admission && actual_admission!=host.admitted) || (!has_planes&&!has_admission))
    throw std::runtime_error("gate membership differs from independent host fixture");
  const auto actual_y=d.y.download(),actual_dx=d.dx.download(),actual_dq=d.dq.download();
  const float yerr=max_error(actual_y,ey), dxerr=max_error(actual_dx,edx), dqerr=max_error(actual_dq,edq);
  float dserr=0, scoreerr=0;
  if(o.gate==GateProfile::ste){
    std::vector<float> expected_ds(edq.size()),expected_scores(edq.size());
    for(std::size_t i=0;i<edq.size();++i){expected_ds[i]=edq[i]*host.probabilities[i]*(1.f-host.probabilities[i]); expected_scores[i]=initial_scores[i]-score_rate*expected_ds[i];}
    auto actual_ds=ds.download(),actual_scores=scores.download();
    dserr=max_error(actual_ds,expected_ds);scoreerr=max_error(actual_scores,expected_scores);
    for(std::size_t i=0;i<edq.size();++i){
      float upstream=o.body==BodyKind::ordered_product?reference::product_abs_tolerance*(1.f+std::fabs(edq[i])):linear_bounds.dq[i];
      const float dbound=0.25f*upstream+1e-6f*(1.f+std::fabs(expected_ds[i]));
      if(std::fabs(actual_ds[i]-expected_ds[i])>dbound || std::fabs(actual_scores[i]-expected_scores[i])>score_rate*dbound+1e-6f)
        throw std::runtime_error("STE gradient/update outside propagated bound");
    }
  }
  float paramerr=0;
  if(o.body==BodyKind::ordered_product){auto got=d.dtheta.download();paramerr=max_error(got,edtheta);check_product(actual_y,ey,"y");check_product(actual_dx,edx,"dx");check_product(actual_dq,edq,"dq");check_product(got,edtheta,"dtheta");}
  else {auto got=d.dweights.download(); got.resize(edweights.size()); paramerr=max_error(got,edweights);check_bounds(actual_y,ey,linear_bounds.y,"y");check_bounds(actual_dx,edx,linear_bounds.dx,"dx");check_bounds(actual_dq,edq,linear_bounds.dq,"dq");check_bounds(got,edweights,linear_bounds.dweights,"dweights");}
  float model_update_err=0;
  if(update_model){
    const auto& grad=o.body==BodyKind::ordered_product?edtheta:edweights;
    std::vector<float> actual,expected(grad.size());
    if(o.body==BodyKind::ordered_product)actual=d.theta.download();
    else if(o.dtype==DType::fp32)actual=d.w32.download();
    else {auto bits=d.w16.download();actual.resize(bits.size());for(std::size_t i=0;i<bits.size();++i)actual[i]=reference::half_bits_to_float(bits[i]);}
    for(std::size_t i=0;i<grad.size();++i){
      const float before=o.body==BodyKind::ordered_product?host.theta[i]:(o.dtype==DType::fp16?reference::half_bits_to_float(host.weights16[i]):host.weights32[i]);
      expected[i]=reference::round_value(before-0.01f*grad[i],o.body==BodyKind::ordered_product?DType::fp32:o.dtype);
    }
    model_update_err=max_error(actual,expected);
    for(std::size_t i=0;i<grad.size();++i){
      float gb=o.body==BodyKind::ordered_product?reference::product_abs_tolerance*(1.f+std::fabs(grad[i])):linear_bounds.dweights[i];
      float rounding=1e-6f*(1.f+std::fabs(expected[i]));
      if(o.body==BodyKind::grouped_linear && o.dtype==DType::fp16){
        const float magnitude=std::fabs(expected[i]);
        rounding+=magnitude<std::ldexp(1.f,-14)?std::ldexp(1.f,-24):std::ldexp(1.f,std::ilogb(magnitude)-10);
      }
      if(std::fabs(actual[i]-expected[i])>0.01f*gb+rounding)throw std::runtime_error("model update outside propagated bound");
    }
  }
  // This is deliberately only a local diagnostic, not the approved compound
  // acceptance bound; an independent validation packet adjudicates accuracy.
  if(!std::isfinite(yerr+dxerr+dqerr+paramerr)) throw std::runtime_error("invalid numerical differences");
  Event start,stop,teacher_fit_begin,teacher_fit_end,teacher_hard_begin,teacher_hard_end,teacher_circuit_begin,teacher_circuit_end,ste_forward_begin,ste_forward_end,ste_update_begin,ste_update_end,model_begin,model_end;
  teacher::ChainEvents teacher_events{};
  teacher_events.training={teacher_fit_begin.value,teacher_fit_end.value};
  teacher_events.hardening={teacher_hard_begin.value,teacher_hard_end.value};
  teacher_events.circuit={teacher_circuit_begin.value,teacher_circuit_end.value};
  std::vector<float> prep,fw,bw,condition_times,teacher_fit,teacher_hard,teacher_circuit,ste_gate,ste_update,model_update;
  for(unsigned i=0;i<o.warmups+o.repeats;++i){
    d.clear(stream.value);
    float condition_ms=0;
    if(o.gate==GateProfile::supplied || o.gate==GateProfile::teacher){
      cuda_check(cudaEventRecord(start.value,stream.value),"event");
      cuda_check(conditions->rebuild(producer,false,o.gate,produce_condition_bytes,stream.value,request_condition_planes),"condition rebuild");
      if(o.condition_update=="rebuild")cuda_check(conditions->rebuild(producer,true,o.gate,produce_condition_bytes,stream.value,request_condition_planes),"condition full rebuild");
      if(o.condition_update=="incremental")cuda_check(conditions->update(producer,o.gate,produce_condition_bytes,stream.value,request_condition_planes),"condition incremental update");
      cuda_check(cudaEventRecord(stop.value,stream.value),"event");cuda_check(cudaEventSynchronize(stop.value),"event wait");
      cuda_check(cudaEventElapsedTime(&condition_ms,start.value,stop.value),"elapsed");
      if(conditions->status(stream.value))throw std::runtime_error("condition producer failed");
      d.g=conditions->gates(o.gate);
      if(o.gate==GateProfile::teacher)d.g=teacher->enqueue(d.g,d.status.ptr,parameter_revision,update_model,stream.value,teacher_events,o.backend==Backend::scalar);
    }
    d.g.revisions.parameter_revision=parameter_revision;
    if(o.gate==GateProfile::ste){
      if(update_model)++d.g.revisions.score_generation;
      cuda_check(ste_chain_forward(d.p.batch,d.p.topology.conditions,1.f,0.5f,d.g.revisions,sb,&chain,{ste_forward_begin.value,ste_forward_end.value},stream.value),"STE forward");d.g=chain.gates;
    }
    cuda_check(cudaEventRecord(start.value,stream.value),"event");
    cuda_check(prepare(o.backend,o.body,d.p,d.g,opt,ws,stream.value),"prepare");
    cuda_check(cudaEventRecord(stop.value,stream.value),"event"); cuda_check(cudaEventSynchronize(stop.value),"event wait");
    float a; cuda_check(cudaEventElapsedTime(&a,start.value,stop.value),"elapsed");
    cuda_check(cudaEventRecord(start.value,stream.value),"event"); cuda_check(forward(o.backend,o.body,d.p,d.g,opt,d.result,ws,stream.value),"forward");
    cuda_check(cudaEventRecord(stop.value,stream.value),"event"); cuda_check(cudaEventSynchronize(stop.value),"event wait");
    float b; cuda_check(cudaEventElapsedTime(&b,start.value,stop.value),"elapsed");
    cuda_check(cudaEventRecord(start.value,stream.value),"event"); cuda_check(backward(o.backend,o.body,d.p,d.g,d.dy.ptr,opt,d.result,ws,stream.value),"backward");
    cuda_check(cudaEventRecord(stop.value,stream.value),"event"); cuda_check(cudaEventSynchronize(stop.value),"event wait");
    float c; cuda_check(cudaEventElapsedTime(&c,start.value,stop.value),"elapsed");
    if(o.gate==GateProfile::ste)cuda_check(ste_chain_backward_update(&chain,d.g.revisions,d.result,ws,score_rate,{ste_update_begin.value,ste_update_end.value},stream.value),"STE update");
    if(update_model){
      auto pointer=o.body==BodyKind::ordered_product?static_cast<void*>(d.theta.ptr):const_cast<void*>(d.p.weights);
      auto gradient=o.body==BodyKind::ordered_product?d.dtheta.ptr:d.dweights.ptr;
      auto count=o.body==BodyKind::ordered_product?d.p.topology.parameters:d.p.weight_count;
      auto dtype=o.body==BodyKind::ordered_product?DType::fp32:o.dtype;
      cuda_check(cudaEventRecord(model_begin.value,stream.value),"model update begin");
      cuda_check(model_sgd(pointer,gradient,count,dtype,0.01f,d.status.ptr,stream.value),"model update");
      cuda_check(cudaEventRecord(model_end.value,stream.value),"model update end");
      ++parameter_revision;
    }
    cuda_check(cudaStreamSynchronize(stream.value),"complete step");
    auto elapsed=[](const Event& a,const Event& b){float ms=0;cuda_check(cudaEventElapsedTime(&ms,a.value,b.value),"phase elapsed");return ms;};
    if(i>=o.warmups){condition_times.push_back(condition_ms);prep.push_back(a);fw.push_back(b);bw.push_back(c);
      teacher_fit.push_back(o.gate==GateProfile::teacher?elapsed(teacher_fit_begin,teacher_fit_end):0.f);
      teacher_hard.push_back(o.gate==GateProfile::teacher?elapsed(teacher_hard_begin,teacher_hard_end):0.f);
      teacher_circuit.push_back(o.gate==GateProfile::teacher?elapsed(teacher_circuit_begin,teacher_circuit_end):0.f);
      ste_gate.push_back(o.gate==GateProfile::ste?elapsed(ste_forward_begin,ste_forward_end):0.f);
      ste_update.push_back(o.gate==GateProfile::ste?elapsed(ste_update_begin,ste_update_end):0.f);
      model_update.push_back(update_model?elapsed(model_begin,model_end):0.f);
    }
  }
  flags=d.status.download(); if(flags[0]) throw std::runtime_error("timing device status="+std::to_string(flags[0]));
  auto host_ms=[](auto a,auto b){return std::chrono::duration<double,std::milli>(b-a).count();};
  std::ostringstream out; out<<std::setprecision(9)<<"{\"schema\":\"cgs.body-diagnostic.v1\",\"boundary\":\"body-diagnostic-not-complete-cost\",\"accuracy_accepted\":false,\"case\":\""<<o.case_id<<"\",\"backend\":\""<<backend_name(o.backend)<<"\",\"gate\":\""<<(o.gate==GateProfile::ste?"ste":o.gate==GateProfile::teacher?"teacher":"supplied")<<"\",\"body\":\""<<(o.body==BodyKind::ordered_product?"product":"linear")<<"\",\"precision\":\""<<(o.dtype==DType::fp16?"fp16":"fp32")<<"\",\"condition_update\":\""<<o.condition_update<<"\",\"fixture_host_ms\":"<<host_ms(fixture_start,fixture_end)<<",\"reference_host_ms\":"<<host_ms(fixture_end,reference_end)<<",\"allocation_and_transfer_host_ms\":"<<host_ms(reference_end,transfer_end)<<",\"workspace_bytes\":"<<bytes<<",\"max_abs_y\":"<<yerr<<",\"max_abs_dx\":"<<dxerr<<",\"max_abs_dq\":"<<dqerr<<",\"max_abs_parameter\":"<<paramerr<<",\"max_abs_ds\":"<<dserr<<",\"max_abs_score_update\":"<<scoreerr<<",\"max_abs_model_update\":"<<model_update_err;
  out<<",\"workload_role\":\""<<(it->role==WorkloadRole::complete_consumer?"complete-consumer":"producer-only-diagnostic")<<"\",\"manifest_seed\":"<<it->seed<<",\"cells\":"<<it->cells<<",\"features\":"<<it->features<<",\"groups\":"<<it->groups<<",\"condition_edges\":"<<it->condition_edges<<",\"demanded_roots\":"<<workload.demanded_roots.size()<<",\"demanded_nodes\":"<<workload.demanded_nodes.size()<<",\"accessibility_source\":\""<<(o.replay_path.empty()?"synthetic":"10x-pbmc-cgatac01-sha256-45de0da9")<<"\",\"teacher_uniform_table_soft_hard_mean\":"<<teacher_uniform_soft_hard;
  auto array=[&](const char* name,const auto& v){out<<",\""<<name<<"\":[";for(std::size_t i=0;i<v.size();++i){if(i)out<<',';out<<v[i];}out<<']';};
  array("condition_ms",condition_times);array("teacher_fit_ms",teacher_fit);array("teacher_hardening_ms",teacher_hard);array("teacher_circuit_ms",teacher_circuit);array("ste_gate_ms",ste_gate);array("ste_update_ms",ste_update);array("model_update_ms",model_update);array("prepare_ms",prep);array("forward_ms",fw);array("backward_ms",bw);out<<"}\n";
  if(path.empty()) std::fputs(out.str().c_str(),stdout); else {std::ofstream f(path);if(!f)throw std::runtime_error("cannot open output");f<<out.str();if(!f)throw std::runtime_error("cannot write output");}
  return 0;
 } catch(const std::exception& e){std::fprintf(stderr,"cgs body diagnostic: %s\n",e.what());return 1;}
}
}
