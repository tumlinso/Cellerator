#include <Cellerator/nn/query.hh>
#include <span>
#include <vector>
#include <cstdio>
namespace nn = cellerator::nn;
struct Binding { std::vector<int> ids{1}; const auto& canonical_ids() const { return ids; } };
struct Targets { std::vector<int> ids{1}; std::vector<long> cols{0}; const auto& canonical_ids() const { return ids; } const auto& source_ids() const { return ids; } const auto& model_indices() const { return cols; } const auto& output_columns() const { return cols; } };
struct Result { std::vector<int> data; };
struct Model { Result operator()(int) { return {{42}}; } };
struct Full { std::vector<int> data; template<class T> std::span<const int> gather_feature_columns(const T&) && { return data; } };
struct Head { Full operator()(const std::vector<int>& src) { return {src}; } };
int main() { Binding b; Targets t; Head h; auto v=nn::readout(h,&Result::data,b); int coord=0; auto request=nn::query(v,t,coord); Model model; auto p=nn::predict(model,0,request); std::printf("%d\n", p.values[0]); }
