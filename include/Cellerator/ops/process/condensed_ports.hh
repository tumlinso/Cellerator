#pragma once
#include <Cellerator/ops/matrix/host_reference.hh>

namespace ce_moon::mechanisms {
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
} // namespace ce_moon::mechanisms
