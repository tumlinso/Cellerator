#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ce_moon::mechanisms {
// Row-major FP64 host reference. Axes represent numerical variables only.
struct Matrix {
  std::size_t rows=0, cols=0;
  std::vector<double> data;
  static std::size_t checked_size(std::size_t r,std::size_t c) {
    if(c && r>std::numeric_limits<std::size_t>::max()/c)
      throw std::overflow_error("matrix dimensions");
    return r*c;
  }
  Matrix() = default;
  Matrix(std::size_t r, std::size_t c): rows(r),cols(c),data(checked_size(r,c),0) {}
  Matrix(std::size_t r, std::size_t c,std::vector<double> v):rows(r),cols(c),data(std::move(v)) {
    if(data.size()!=checked_size(r,c)) throw std::invalid_argument("matrix shape");
  }
  std::size_t index(std::size_t r,std::size_t c) const {
    if(r>=rows || c>=cols) throw std::out_of_range("matrix coordinate");
    // Dimensions are public in this experimental record; guard their mutation too.
    if(data.size()!=checked_size(rows,cols)) throw std::invalid_argument("matrix shape");
    return r*cols+c;
  }
  double& operator()(std::size_t r,std::size_t c) { return data.at(index(r,c)); }
  double operator()(std::size_t r,std::size_t c) const { return data.at(index(r,c)); }
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
} // namespace ce_moon::mechanisms
