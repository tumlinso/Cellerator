#pragma once
#include <map>
#include <string>
#include <vector>
namespace Cellerator::compiler::composition {
using canonical_semantics_v1=std::map<std::string,std::string>;
struct intentional_difference_v1{std::string field,reason;};
struct semantic_differential_v1{bool equivalent=false;std::vector<std::string> unexpected;std::vector<intentional_difference_v1> intentional;};
[[nodiscard]] semantic_differential_v1 compare_canonical_semantics_v1(const canonical_semantics_v1&,const canonical_semantics_v1&,const std::vector<intentional_difference_v1>&);
} // namespace Cellerator::compiler::composition
