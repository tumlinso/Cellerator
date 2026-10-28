#include <Cellerator/compiler/frontend/parser/parse_anonymous_execution_fields_v1.hh>
#include <Cellerator/compiler/frontend/parser/parse_named_execution_fields_and_references_v1.hh>
#include <Cellerator/compiler/ir/semantic/deliver_source_to_semantic_ir_vertical_slice_v1.hh>
#include <Cellerator/compiler/backend/implement_generic_cpu_relation_apply_v1.hh>
#include <Cellerator/compiler/backend/implement_cpu_segment_gate_update_bundle_and_chain_paths_v1.hh>
#include <Cellerator/compute/operation/relation_algebra.hh>
#include <cassert>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

namespace parser = Cellerator::compiler::frontend::parser;
namespace sem = Cellerator::compiler::ir::semantic;
namespace cb = cellerator::compiler::backend::v1;
namespace ex = cellerator::execution;
namespace co = cellerator::compute::operation;
namespace cp = cellerator::compute::projection_family;

ex::persistent_axis_identity axis(std::uint64_t s) {
    return {{ex::biological_abi_version,
             ex::serialized_record_kind::persistent_axis_identity,
             sizeof(ex::persistent_axis_identity)},
            {s, 1}, {s+1, 1}, {s+2, 1}, {s+3, 1}};
}

std::vector<double> reference(const std::vector<std::uint64_t>& off,
                        const std::vector<std::uint64_t>& col,
                        const std::vector<double>& val,
                        const std::vector<double>& x,
                        const std::vector<std::uint64_t>& seg) {
    std::vector<double> y(off.size()-1), p(y.size());
    for (std::size_t r=0; r<y.size(); ++r)
        for (auto e=off[r]; e<off[r+1]; ++e) y[r]+=val[e]*x[col[e]];
    for (std::size_t s=0; s+1<seg.size(); ++s) {
        const auto b=seg[s], e=seg[s+1];
        if (b==e) continue;
        double m=y[b]; for (auto r=b+1;r<e;++r) m=std::max(m,y[r]);
        double d=0; for (auto r=b;r<e;++r) d+=std::exp(y[r]-m);
        for (auto r=b;r<e;++r) p[r]=std::exp(y[r]-m)/d;
    }
    return p;
}

std::vector<float> native(const std::vector<std::uint64_t>& off,
                          const std::vector<std::uint64_t>& col,
                          const std::vector<double>& vd,
                          const std::vector<double>& xd,
                          const std::vector<std::uint64_t>& seg) {
    const auto n=off.size()-1, m=xd.size(), E=col.size();
    std::vector<float> val(E), x(m), y(n), p(n);
    for(std::size_t i=0;i<E;++i) val[i]=static_cast<float>(vd[i]);
    for(std::size_t i=0;i<m;++i) x[i]=static_cast<float>(xd[i]);
    std::vector<std::uint64_t> logical(E); for(std::size_t i=0;i<E;++i) logical[i]=i;
    cb::cpu_relation_apply_request_v1 q{};
    q.problem.operation_identity={7,11}; q.problem.relation.structure={17,19};
    q.problem.relation.epoch={1}; q.problem.relation.source_axis=axis(30);
    q.problem.relation.destination_axis=axis(40); q.problem.relation.logical_edge_count=E;
    q.problem.numeric={ex::numeric_type::f32,ex::numeric_type::f32,
      ex::numeric_type::f32,ex::numeric_type::f32,ex::numeric_type::f32,
      ex::numeric_type::f32,co::core::rounding_policy::nearest_even,
      co::core::saturation_policy::none,co::nan_policy_v1::reject,{}};
    q.problem.semantic_flags=co::alpha_applied_once|co::beta_applied_once;
    q.problem.dense_width=1;
    q.projection.destination_offsets=off.data();q.projection.source_indices=col.data();
    q.projection.logical_edge_ids=logical.data();q.projection.source_count=m;
    q.projection.destination_count=n;q.projection.logical_edge_count=E;
    q.relation_values=val.data();q.input=x.data();q.output=y.data();
    assert(cb::apply_cpu_relation_v1(q)==cb::cpu_relation_apply_status_v1::success);
    assert(cb::run_cpu_segment_v1({cb::cpu_segment_kind_v1::softmax,seg.data(),
      static_cast<std::uint64_t>(seg.size()-1),y.data(),p.data()})==cb::cpu_fallback_status_v1::success);
    return p;
}

struct node { std::string output, operation; std::vector<std::string> inputs; };

std::vector<std::string> statements(const parser::anonymous_execution_field_v1& field) {
  std::vector<std::string> out;
  for (const auto& child:field.nested_fields) {
    auto nested=statements(child);out.insert(out.end(),nested.begin(),nested.end());
  }
  std::vector<std::string> pieces;
  for(const auto& captured:field.captured_cxx_statements) {
    std::size_t begin=0;
    for(std::size_t i=0;i<captured.size();++i)if(captured[i]==';') {
      pieces.push_back(captured.substr(begin,i-begin+1));begin=i+1;
    }
    if(begin<captured.size())pieces.push_back(captured.substr(begin));
  }
  for (auto statement:pieces) {
    if(const auto separator=statement.rfind("::");separator!=std::string::npos) statement.erase(0,separator+2);
    const auto open=statement.find('(');
    if(open==std::string::npos) continue;
    const auto equal=statement.find('=');
    const auto close=statement.find(')',open);
    if(equal==std::string::npos||close==std::string::npos||equal>open) continue;
    node n; n.output=statement.substr(0,equal); n.operation=statement.substr(equal+1,open-equal-1);
    std::size_t begin=open+1;
    for(std::size_t i=begin;i<=close;++i) if(i==close||statement[i]==','){
      n.inputs.push_back(statement.substr(begin,i-begin));begin=i+1;
    }
    auto trim=[](std::string&v){auto a=v.find_first_not_of(" \t\r\n");auto b=v.find_last_not_of(" \t\r\n;");v=a==std::string::npos?"":v.substr(a,b-a+1);};
    trim(n.output);trim(n.operation);for(auto&v:n.inputs)trim(v);
    if(n.operation=="relation_apply"&&n.inputs.size()==2)out.push_back(n.output+"=relation:"+n.inputs[0]+":"+n.inputs[1]);
    else if(n.operation=="segment_softmax"&&n.inputs.size()==2)out.push_back(n.output+"=softmax:"+n.inputs[0]+":"+n.inputs[1]);
    else if(n.operation!="") throw std::runtime_error("unsupported semantic operation in local adapter: "+n.operation);
  }
  return out;
}

std::vector<std::string> graph_from_source(const std::string& source,int shape) {
  auto trim=[](std::string&v){auto a=v.find_first_not_of(" \t\r\n");auto b=v.find_last_not_of(" \t\r\n;");v=a==std::string::npos?"":v.substr(a,b-a+1);};
  std::vector<std::string> ops;
  if(shape==1){auto p=parser::parse_named_execution_fields_v1(source);assert(p.accepted()&&p.fields.size()==1);ops=statements(p.fields[0].body);}
  else {auto p=parser::parse_anonymous_execution_fields_v1(source);assert(p.accepted()&&p.fields.size()==1);ops=statements(p.fields[0]);}
  std::vector<std::string> unique;
  for(const auto&op:ops)if(std::find(unique.begin(),unique.end(),op)==unique.end())unique.push_back(op);
  return unique;
}

std::vector<double> evaluate_graph(const std::vector<std::string>&graph,
 const std::vector<std::uint64_t>&off,const std::vector<std::uint64_t>&col,
 const std::vector<double>&values,const std::vector<double>&x,
 const std::vector<std::uint64_t>&seg,std::vector<double>*intermediate) {
  std::unordered_map<std::string,std::vector<double>> arrays{{"A_vals",values},{"x",x}};
  for(const auto&row:graph){auto eq=row.find('=');auto a=row.find(':',eq);auto b=row.find(':',a+1);
    const auto out=row.substr(0,eq),op=row.substr(eq+1,a-eq-1),i0=row.substr(a+1,b-a-1),i1=row.substr(b+1);
    if(op=="relation") {assert(i0=="A"||i0=="A_vals");assert(arrays.contains(i1));std::vector<double> y(off.size()-1);
      const auto&iv=arrays.at(i0=="A"?"A_vals":i0);const auto&xv=arrays.at(i1);
      for(std::size_t r=0;r<y.size();++r)for(auto e=off[r];e<off[r+1];++e)y[r]+=iv[e]*xv[col[e]];arrays[out]=y;*intermediate=y;
    } else {assert(op=="softmax"&&arrays.contains(i0)&&i1=="segment_offsets");const auto&v=arrays.at(i0);std::vector<double>y(v.size());
      for(std::size_t s=0;s+1<seg.size();++s){auto b=seg[s],e=seg[s+1];if(b==e)continue;double m=v[b];for(auto r=b+1;r<e;++r)m=std::max(m,v[r]);double d=0;for(auto r=b;r<e;++r)d+=std::exp(v[r]-m);for(auto r=b;r<e;++r)y[r]=std::exp(v[r]-m)/d;}arrays[out]=y;
    }
  }
  return arrays.at("p");
}

std::vector<float> execute_native_graph(const std::vector<std::string>& graph,
 const std::vector<std::uint64_t>&off,const std::vector<std::uint64_t>&col,
 const std::vector<double>&values,const std::unordered_map<std::string,std::vector<double>>&bindings,
 const std::vector<std::uint64_t>&seg) {
  assert(graph.size()==2);
  auto parse=[](const std::string& row){auto eq=row.find('=');auto a=row.find(':',eq);auto b=row.find(':',a+1);
    return std::vector<std::string>{row.substr(0,eq),row.substr(eq+1,a-eq-1),row.substr(a+1,b-a-1),row.substr(b+1)};};
  const auto relation=parse(graph[0]), softmax=parse(graph[1]);
  assert(relation[1]=="relation"&&relation[2]=="A_vals");
  assert(softmax[1]=="softmax"&&softmax[2]==relation[0]&&softmax[3]=="segment_offsets");
  const auto& xd=bindings.at(relation[3]);
  const auto n=off.size()-1,m=xd.size(),E=col.size();
  std::vector<float> val(E),x(m),y(n),p(n);
  for(std::size_t i=0;i<E;++i)val[i]=static_cast<float>(values[i]);
  for(std::size_t i=0;i<m;++i)x[i]=static_cast<float>(xd[i]);
  std::vector<std::uint64_t> logical(E);for(std::size_t i=0;i<E;++i)logical[i]=i;
  cb::cpu_relation_apply_request_v1 q{};
  q.problem.operation_identity={7,11};q.problem.relation.structure={17,19};q.problem.relation.epoch={1};
  q.problem.relation.source_axis=axis(30);q.problem.relation.destination_axis=axis(40);
  q.problem.relation.logical_edge_count=E;
  q.problem.numeric={ex::numeric_type::f32,ex::numeric_type::f32,ex::numeric_type::f32,
    ex::numeric_type::f32,ex::numeric_type::f32,ex::numeric_type::f32,
    co::core::rounding_policy::nearest_even,co::core::saturation_policy::none,co::nan_policy_v1::reject,{}};
  q.problem.semantic_flags=co::alpha_applied_once|co::beta_applied_once;q.problem.dense_width=1;
  q.projection.destination_offsets=off.data();q.projection.source_indices=col.data();
  q.projection.logical_edge_ids=logical.data();q.projection.source_count=m;q.projection.destination_count=n;
  q.projection.logical_edge_count=E;q.relation_values=val.data();q.input=x.data();q.output=y.data();
  assert(cb::apply_cpu_relation_v1(q)==cb::cpu_relation_apply_status_v1::success);
  assert(cb::run_cpu_segment_v1({cb::cpu_segment_kind_v1::softmax,seg.data(),
    static_cast<std::uint64_t>(seg.size()-1),y.data(),p.data()})==cb::cpu_fallback_status_v1::success);
  return p;
 }

int main() {
  const std::string body="y = relation_apply(A_vals, x);\np = segment_softmax(y, segment_offsets);";
  const std::string anonymous="<[ ::\n"+body+"\n]>";
  const std::string named_source="field void score() <[ ::\n"+body+"\n]>";
  const std::string nested_source="<[ ::\n field void inner() <[ ::\n y = relation_apply(A_vals, x);\n]>\n inner();\n p = segment_softmax(y, segment_offsets);\n]>";
  const auto anon=parser::parse_anonymous_execution_fields_v1(anonymous);
  const auto named=parser::parse_named_execution_fields_v1(named_source);
  const auto nested=parser::parse_anonymous_execution_fields_v1(nested_source);
  assert(anon.accepted()&&anon.fields.size()==1&&anon.fields[0].captured_cxx_statements.size()==2);
  assert(named.accepted()&&named.fields.size()==1&&named.fields[0].body.captured_cxx_statements.size()==2);
  assert(nested.accepted()&&nested.fields.size()==1&&nested.fields[0].nested_fields.size()==1);
  const auto ga=graph_from_source(anonymous,0),gn=graph_from_source(named_source,1),gt=graph_from_source(nested_source,0);
  assert(ga==gn&&ga==gt&&ga.size()==2);
  const std::string slice=R"cell(#pragma cellerator
field void score() <[
 given ce::profile(cpu_probe);
::
 edge = ce::contract_on(support, expression, context);
 result = ce::segment_normalize(edge, scale);
]>
)cell";
  auto ir=sem::lower_cell_source_to_semantic_ir_v1(slice);
  assert(ir&&ir->operations.size()==2);
  const auto canonical=sem::write_semantic_ir_v1(*ir); assert(canonical);
  auto round=sem::read_semantic_ir_v1(*canonical);assert(round&&round->operations.size()==2);
  auto changed=slice;changed.replace(changed.find("expression"),10,"different");
  auto changed_ir=sem::lower_cell_source_to_semantic_ir_v1(changed);
  assert(changed_ir&&sem::write_semantic_ir_v1(*changed_ir)!=canonical);

  // Three frozen deterministic fixtures include signed weights, an empty
  // relation row, and singleton softmax segments.
  const std::vector<std::uint64_t> off{0,2,2,5,6}, col{0,2,1,0,2,1}, seg{0,1,4};
  const std::vector<double> x{2,3,4};
  for(int seed=0;seed<3;++seed) {
    std::vector<double> val{.5,-.25,1,-.75,.5,2};
    for(std::size_t i=0;i<val.size();++i) val[i]+=0.125*seed*static_cast<double>(static_cast<int>(i%3)-1);
    const auto expected=reference(off,col,val,x,seg);
    const auto actual=execute_native_graph(ga,off,col,val,{{"x",x}},seg);
    for(std::size_t i=0;i<actual.size();++i) assert(std::abs(actual[i]-expected[i])<2e-7);
    std::vector<double> intermediate;
    const auto graph_output=evaluate_graph(ga,off,col,val,x,seg,&intermediate);
    for(std::size_t i=0;i<actual.size();++i) assert(std::abs(graph_output[i]-expected[i])<1e-12);
  }
  std::vector<double> intermediate;
  const std::vector<double> val{.5,-.25,1,-.75,.5,2};
  const auto graph_output=evaluate_graph(ga,off,col,val,x,seg,&intermediate);
  auto dependency_source=anonymous;auto at=dependency_source.find("A_vals, x");assert(at!=std::string::npos);
  dependency_source.replace(at,9,"A_vals, z");
  const auto gd=graph_from_source(dependency_source,0);assert(gd!=ga);
  const std::vector<double> z{4,3,2};
  const auto changed_output=execute_native_graph(gd,off,col,val,{{"z",z}},seg);
  assert(std::vector<float>(changed_output.begin(),changed_output.end())!=
         execute_native_graph(ga,off,col,val,{{"x",x}},seg));
  std::cout<<"E1 parsed graph drives existing CPU primitive dispatch, FP64 oracle, altered dependency PASS; typed vector bridge adapter-only\n";
  std::cout<<"semantic_ir="<<*canonical;
}
