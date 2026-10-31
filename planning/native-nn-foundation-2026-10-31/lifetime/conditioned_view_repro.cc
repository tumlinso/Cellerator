#include <Cellerator/nn/query.hh>
#include <span>
#include <vector>
#include <cstdio>
namespace nn = cellerator::nn;
struct Binding { std::vector<int> ids{1}; const auto& canonical_ids() const { return ids; } };
struct Targets { std::vector<int> ids{1}; std::vector<long> cols{0}; const auto& canonical_ids() const { return ids; } const auto& source_ids() const { return ids; } const auto& model_indices() const { return cols; } const auto& output_columns() const { return cols; } };
struct Result { std::vector<int> data; };
struct Model { Result operator()(int) { return {{42}}; } };
struct Head { template<class T,class C> std::span<const int> query(const std::vector<int>& src, const T&, const C&) { return src; } };
struct Cond { std::span<const int> operator()(const std::vector<int>& x, int) const { return x; } };
struct SpanModel { std::span<const int> operator()(std::span<const int> x) const { return x; } };
struct SpanHead { template<class T,class C> std::span<const int> query(std::span<const int> src, const T&, const C&) { return src; } };
struct Selector { std::span<const int> operator()(std::span<const int> x) const { return x; } };
int main() { Binding b; Targets t; SpanHead h; Selector sel; auto v=nn::readout(h,sel,b); int coord=0; auto request=nn::query(v,t,coord); auto p=nn::predict(nn::conditioned(SpanModel{},std::vector<int>{42},Cond{},0),nn::Run{},request); std::printf("%d\n",p.values[0]); }
