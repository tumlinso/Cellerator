#pragma once
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
namespace Cellerator::compiler::tooling {
enum class language_feature_v1 : std::uint8_t { completion=1, hover, definition, references, rename };
struct language_feature_item_v1 { std::string symbol, detail, uri; std::uint64_t begin=0,end=0; bool cellerator=false; };
struct language_feature_result_v1 { language_feature_v1 feature=language_feature_v1::completion; std::vector<language_feature_item_v1> items; };
[[nodiscard]] language_feature_result_v1 merge_language_features_v1(language_feature_v1 feature,
 std::vector<language_feature_item_v1> ordinary_cpp,std::vector<language_feature_item_v1> cellerator);
[[nodiscard]] bool rename_is_consistent_v1(const language_feature_result_v1 &result,std::string_view old_name);
} // namespace Cellerator::compiler::tooling
