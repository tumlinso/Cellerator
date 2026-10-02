#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ce_moon::mechanisms {
// Row-major FP64 host reference. Axes represent numerical variables only.
struct Matrix {
  std::size_t rows=0, cols=0;
  std::vector<double> data;
  Matrix() = default;
  Matrix(std::size_t r, std::size_t c): rows(r),cols(c),data(r*c,0) {}
  Matrix(std::size_t r, std::size_t c,std::vector<double> v):rows(r),cols(c),data(std::move(v)) {
    if(data.size()!=r*c) throw std::invalid_argument("matrix shape");
  }
  double& operator()(std::size_t r,std::size_t c) { return data.at(r*cols+c); }
  double operator()(std::size_t r,std::size_t c) const { return data.at(r*cols+c); }
};
inline void finite(const std::vector<double>& a) {
  for(double v:a) if(!std::isfinite(v)) throw std::invalid_argument("nonfinite numerical input");
}
inline Matrix multiply(const Matrix& a,const Matrix& b) {
  if(a.cols!=b.rows) throw std::invalid_argument("multiply shape");
  finite(a.data); finite(b.data);
  Matrix c(a.rows,b.cols);
  for(std::size_t i=0;i<a.rows;++i) for(std::size_t k=0;k<a.cols;++k)
    for(std::size_t j=0;j<b.cols;++j) c(i,j)+=a(i,k)*b(k,j);
  finite(c.data); return c;
}
inline std::vector<double> matvec(const Matrix& a,const std::vector<double>& x) {
  if(a.cols!=x.size()) throw std::invalid_argument("apply shape");
  finite(a.data); finite(x);
  std::vector<double> y(a.rows,0);
  for(std::size_t i=0;i<a.rows;++i) for(std::size_t j=0;j<a.cols;++j) y[i]+=a(i,j)*x[j];
  finite(y); return y;
}
// Partial-pivot Gaussian solve; tolerance relative to global max entry, no conditioning certificate.
inline Matrix solve(Matrix a,Matrix b,double pivot_tolerance=1e-12) {
  if(a.rows==0 || a.rows!=a.cols || b.rows!=a.rows || !(pivot_tolerance>0))
    throw std::invalid_argument("solve shape/tolerance");
  finite(a.data); finite(b.data);
  double scale=0; for(double v:a.data) scale=std::max(scale,std::abs(v));
  for(std::size_t k=0;k<a.rows;++k) {
    std::size_t pivot=k;
    for(std::size_t i=k+1;i<a.rows;++i) if(std::abs(a(i,k))>std::abs(a(pivot,k))) pivot=i;
    if(std::abs(a(pivot,k))<=pivot_tolerance*scale) throw std::domain_error("singular/near-singular pivot");
    if(pivot!=k) {
      for(std::size_t j=0;j<a.cols;++j) std::swap(a(k,j),a(pivot,j));
      for(std::size_t j=0;j<b.cols;++j) std::swap(b(k,j),b(pivot,j));
    }
    for(std::size_t i=k+1;i<a.rows;++i) {
      double factor=a(i,k)/a(k,k);
      for(std::size_t j=k;j<a.cols;++j) a(i,j)-=factor*a(k,j);
      for(std::size_t j=0;j<b.cols;++j) b(i,j)-=factor*b(k,j);
    }
  }
  Matrix x(a.rows,b.cols);
  for(std::size_t ii=a.rows;ii>0;--ii) {
    std::size_t i=ii-1;
    for(std::size_t j=0;j<b.cols;++j) {
      double v=b(i,j); for(std::size_t k=i+1;k<a.cols;++k) v-=a(i,k)*x(k,j);
      x(i,j)=v/a(i,i);
    }
  }
  finite(x.data); return x;
}
inline std::vector<double> solve(const Matrix& a,const std::vector<double>& b) {
  return solve(a,Matrix(b.size(),1,b)).data;
}
struct PortResponse {
  Matrix condensed; // [port,port], exact Schur model up to FP rounding
  std::vector<double> load; // [port]
  Matrix interior_response; // [interior,port], inv(Aii)*Aip
  std::vector<double> interior_load; // [interior], inv(Aii)*bi
};
inline PortResponse condense_ports(const Matrix& app,const Matrix& api,const Matrix& aip,
                                  const Matrix& aii,const std::vector<double>& bp,
                                  const std::vector<double>& bi) {
  std::size_t p=app.rows,i=aii.rows;
  if(!p || !i || app.cols!=p || api.rows!=p || api.cols!=i || aip.rows!=i || aip.cols!=p ||
     aii.cols!=i || bp.size()!=p || bi.size()!=i) throw std::invalid_argument("port shapes");
  finite(app.data); finite(api.data); finite(aip.data); finite(bp); finite(bi);
  Matrix response=solve(aii,aip),schur=app, product=multiply(api,response);
  auto inner=solve(aii,bi),shift=matvec(api,inner),load=bp;
  for(std::size_t k=0;k<schur.data.size();++k) schur.data[k]-=product.data[k];
  for(std::size_t k=0;k<p;++k) load[k]-=shift[k];
  return {schur,load,response,inner};
}
inline std::vector<double> solve_ports(const PortResponse& ports) { return solve(ports.condensed,ports.load); }
struct PortSystem { Matrix stiffness; std::vector<double> load; };
// Caller declares the same port axis/order for every region. Sum response forces on common ports.
inline PortSystem compose_port_system(const std::vector<PortResponse>& regions) {
  if(regions.empty()) throw std::invalid_argument("empty port composition");
  std::size_t p=regions.front().load.size(); PortSystem system{Matrix(p,p),std::vector<double>(p,0)};
  for(const auto& region:regions) {
    if(region.condensed.rows!=p || region.condensed.cols!=p || region.load.size()!=p)
      throw std::invalid_argument("port composition shapes");
    for(std::size_t k=0;k<p*p;++k) system.stiffness.data[k]+=region.condensed.data[k];
    for(std::size_t k=0;k<p;++k) system.load[k]+=region.load[k];
  }
  finite(system.stiffness.data);finite(system.load);return system;
}
inline std::vector<double> reconstruct_interior(const PortResponse& ports,const std::vector<double>& x) {
  auto y=matvec(ports.interior_response,x);
  for(std::size_t i=0;i<y.size();++i) y[i]=ports.interior_load[i]-y[i];
  return y;
}
inline std::vector<double> residual(const Matrix& a,const std::vector<double>& x,const std::vector<double>& b) {
  auto r=matvec(a,x); if(r.size()!=b.size()) throw std::invalid_argument("residual shape");
  finite(b);
  for(std::size_t i=0;i<r.size();++i) r[i]=b[i]-r[i];
  return r;
}
inline double objective(const Matrix& a,const std::vector<double>& x,const std::vector<double>& b) {
  auto r=residual(a,x,b); double loss=0; for(double v:r) loss+=0.5*v*v; return loss;
}
// E42 one fixed budget correction. No convergence promise; nonsingular R*A*P required.
inline std::vector<double> coarse_direction(const Matrix& a,const Matrix& p,const Matrix& r,
                                           const std::vector<double>& x,const std::vector<double>& b) {
  if(a.rows!=a.cols || p.rows!=a.rows || r.cols!=a.rows || p.cols!=r.rows)
    throw std::invalid_argument("coarse shapes");
  return matvec(p,solve(multiply(multiply(r,a),p),matvec(r,residual(a,x,b))));
}
inline std::vector<double> coarse_correct(const Matrix& a,const Matrix& p,const Matrix& r,
                                         const std::vector<double>& x,const std::vector<double>& b,double alpha=1) {
  if(!std::isfinite(alpha)) throw std::invalid_argument("alpha");
  auto d=coarse_direction(a,p,r,x,b),y=x;
  for(std::size_t i=0;i<y.size();++i) y[i]+=alpha*d[i];
  return y;
}
struct StepFit { double alpha,loss_before,loss_after,gradient_before; };
// Fit scalar coarse response parameter exactly for this quadratic objective/fixture.
inline StepFit fit_step_size(const Matrix& a,const Matrix& p,const Matrix& r,
                             const std::vector<double>& x,const std::vector<double>& b,double initial_alpha=0) {
  auto ad=matvec(a,coarse_direction(a,p,r,x,b)),e=residual(a,x,b);
  double dot=0,norm=0; for(std::size_t i=0;i<e.size();++i) { dot+=e[i]*ad[i]; norm+=ad[i]*ad[i]; }
  if(norm==0) throw std::domain_error("zero coarse direction");
  double alpha=dot/norm;
  return {alpha,objective(a,coarse_correct(a,p,r,x,b,initial_alpha),b),
          objective(a,coarse_correct(a,p,r,x,b,alpha),b),initial_alpha*norm-dot};
}
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
      if(count<capacity) output[count]={{x.id,y->id,z->id},x.key,
        weights[0]+weights[1]*x.value+weights[2]*y->value+weights[3]*z->value};
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
