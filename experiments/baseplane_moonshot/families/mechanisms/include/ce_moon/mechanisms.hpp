#pragma once
#include <Cellerator/ops/process/condensed_ports.hh>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <Cellerator/ops/iterative/coarse_correction.hh>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ce_moon::mechanisms {
// E43 IDs and posting keys are opaque caller provenance. Equal key = caller-declared compatibility.
struct RoleEntry { std::uint64_t id,key; double value; };
struct FactorTuple { std::uint64_t ids[3]; std::uint64_t key; double score; };
struct JoinResult { std::size_t required,written; bool overflow; };
inline JoinResult join_factors(const std::vector<RoleEntry>& a,const std::vector<RoleEntry>& b,
                              const std::vector<RoleEntry>& c,const std::vector<double>& weights,
                              FactorTuple* output,std::size_t capacity) {
  if(weights.size()!=4 || (capacity && !output)) throw std::invalid_argument("factor output/weights");
  finite(weights);
  std::map<std::uint64_t,std::vector<const RoleEntry*>> pb,pc;
  for(const auto& e:b) { if(!std::isfinite(e.value)) throw std::invalid_argument("role value"); pb[e.key].push_back(&e); }
  for(const auto& e:c) { if(!std::isfinite(e.value)) throw std::invalid_argument("role value"); pc[e.key].push_back(&e); }
  std::size_t count=0;
  for(const auto& x:a) {
    if(!std::isfinite(x.value)) throw std::invalid_argument("role value");
    auto ib=pb.find(x.key),ic=pc.find(x.key);
    if(ib==pb.end() || ic==pc.end()) continue;
    for(const auto* y:ib->second) for(const auto* z:ic->second) {
      if(count==std::numeric_limits<std::size_t>::max()) throw std::overflow_error("join cardinality");
      if(count<capacity) {
        const double score=weights[0]+weights[1]*x.value+weights[2]*y->value+weights[3]*z->value;
        if(!std::isfinite(score)) throw std::overflow_error("factor tuple score");
        output[count]={{x.id,y->id,z->id},x.key,score};
      }
      ++count;
    }
  }
  return {count,std::min(count,capacity),count>capacity};
}
// Linear factor scores admit a factorized sum without materializing the hyperedges.
struct FactorAggregate { std::uint64_t key; std::size_t count; double score_sum; };
inline std::vector<FactorAggregate> aggregate_factors(const std::vector<RoleEntry>& a,
    const std::vector<RoleEntry>& b,const std::vector<RoleEntry>& c,const std::vector<double>& weights) {
  if(weights.size()!=4) throw std::invalid_argument("factor weights");
  finite(weights);
  using Posting=std::pair<std::size_t,double>;
  std::map<std::uint64_t,Posting> pa,pb,pc;
  auto collect=[](const std::vector<RoleEntry>& entries,auto& postings) {
    for(const auto& e:entries) {
      if(!std::isfinite(e.value)) throw std::invalid_argument("role value");
      auto& bucket=postings[e.key];++bucket.first;bucket.second+=e.value;
    }
  };
  collect(a,pa);collect(b,pb);collect(c,pc);
  std::vector<FactorAggregate> output;
  for(const auto& entry:pa) {
    auto ib=pb.find(entry.first),ic=pc.find(entry.first);
    if(ib==pb.end() || ic==pc.end()) continue;
    auto na=entry.second.first,nb=ib->second.first,nc=ic->second.first;
    if(na>std::numeric_limits<std::size_t>::max()/nb || na*nb>std::numeric_limits<std::size_t>::max()/nc)
      throw std::overflow_error("factor cardinality");
    double total=weights[0]*double(na)*double(nb)*double(nc)+
      weights[1]*entry.second.second*double(nb)*double(nc)+
      weights[2]*ib->second.second*double(na)*double(nc)+
      weights[3]*ic->second.second*double(na)*double(nb);
    if(!std::isfinite(total)) throw std::overflow_error("factor score sum");
    output.push_back({entry.first,na*nb*nc,total});
  }
  return output;
}
// E44 topological affine DAG. -1 means absent input; earlier node indices only.
struct Node { int lhs=-1,rhs=-1; double lhs_weight=0,rhs_weight=0,bias=0; };
struct WorldDelta { std::size_t node; double value; };
inline void validate_dag(const std::vector<Node>& dag) {
  for(std::size_t i=0;i<dag.size();++i) {
    const auto& n=dag[i];
    if(n.lhs < -1 || n.rhs < -1 || n.lhs>=static_cast<int>(i) || n.rhs>=static_cast<int>(i))
      throw std::invalid_argument("DAG must be topological");
    if(!std::isfinite(n.lhs_weight)||!std::isfinite(n.rhs_weight)||!std::isfinite(n.bias))
      throw std::invalid_argument("DAG nonfinite parameter");
  }
}
inline double evaluate_node(const Node& n,const std::vector<double>& state) {
  return n.bias+(n.lhs<0?0:n.lhs_weight*state[n.lhs])+(n.rhs<0?0:n.rhs_weight*state[n.rhs]);
}
inline std::vector<double> evaluate_independent(const std::vector<Node>& dag,const std::vector<WorldDelta>& deltas={}) {
  validate_dag(dag); std::map<std::size_t,double> overrides;
  for(const auto& d:deltas) {
    if(d.node>=dag.size() || !std::isfinite(d.value) || !overrides.emplace(d.node,d.value).second)
      throw std::invalid_argument("world delta");
  }
  std::vector<double> s(dag.size());
  for(std::size_t i=0;i<dag.size();++i) s[i]=overrides.count(i)?overrides.at(i):evaluate_node(dag[i],s);
  finite(s); return s;
}
struct WorldBatch { std::vector<std::vector<double>> states; std::size_t evaluations=0; };
inline WorldBatch evaluate_worlds(const std::vector<Node>& dag,const std::vector<std::vector<WorldDelta>>& worlds) {
  auto baseline=evaluate_independent(dag); WorldBatch result; result.evaluations=dag.size();
  for(const auto& deltas:worlds) {
    auto state=baseline; std::vector<bool> changed(dag.size(),false); std::map<std::size_t,double> overrides;
    for(const auto& d:deltas) {
      if(d.node>=dag.size() || !std::isfinite(d.value) || !overrides.emplace(d.node,d.value).second)
        throw std::invalid_argument("world delta");
    }
    for(std::size_t i=0;i<dag.size();++i) {
      const auto& n=dag[i];
      if(overrides.count(i)) { state[i]=overrides.at(i); changed[i]=(state[i]!=baseline[i]); }
      else if((n.lhs>=0 && changed[n.lhs]) || (n.rhs>=0 && changed[n.rhs])) {
        state[i]=evaluate_node(n,state); ++result.evaluations; changed[i]=(state[i]!=baseline[i]);
      }
    }
    finite(state); result.states.push_back(std::move(state));
  }
  return result;
}
// Query equality groups do not discard interior states or authorize reuse under another query.
inline std::vector<std::size_t> exact_query_groups(const std::vector<std::vector<double>>& states,
                                                  const std::vector<std::size_t>& query) {
  std::vector<std::size_t> groups(states.size());
  for(const auto& s:states) { finite(s); for(auto q:query) if(q>=s.size()) throw std::invalid_argument("query node"); }
  for(std::size_t i=0;i<states.size();++i) {
    groups[i]=i;
    for(std::size_t j=0;j<i;++j) {
      bool equal=true; for(auto q:query) if(states[i][q]!=states[j][q]) { equal=false; break; }
      if(equal) { groups[i]=groups[j]; break; }
    }
  }
  return groups;
}
// E45 numeric sample oracle: no texture precision or boundary claims.
inline double bilinear_response(double x,double y,const std::vector<double>& corners) {
  if(corners.size()!=4 || !std::isfinite(x)||!std::isfinite(y)||x<0||x>1||y<0||y>1)
    throw std::invalid_argument("bilinear unit-square domain");
  finite(corners);
  return (1-x)*(1-y)*corners[0]+x*(1-y)*corners[1]+(1-x)*y*corners[2]+x*y*corners[3];
}
} // namespace ce_moon::mechanisms
