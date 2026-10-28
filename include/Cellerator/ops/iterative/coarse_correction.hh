#pragma once
#include <Cellerator/ops/matrix/host_reference.hh>

namespace ce_moon::mechanisms {
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
} // namespace ce_moon::mechanisms
