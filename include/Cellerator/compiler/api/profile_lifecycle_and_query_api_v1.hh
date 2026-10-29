#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>
namespace cellerator::compiler::api::v1 {
struct profile_entry_v1{const char* name;double value;};
struct profile_v1{std::unordered_map<std::string,double> states;std::string environment;};
using profile_transfer_v1=double(*)(double,void*) noexcept;
[[nodiscard]] profile_v1 build_profile_v1(const profile_entry_v1* entries,std::size_t count);
[[nodiscard]] profile_v1 load_profile_text_v1(std::string_view text);
[[nodiscard]] profile_v1 load_profile_binary_v1(const std::uint8_t* bytes,std::size_t count);
[[nodiscard]] const double* find_profile_state_v1(const profile_v1& profile,std::string_view name) noexcept;
[[nodiscard]] std::vector<std::string> diff_profiles_v1(const profile_v1& a,const profile_v1& b);
void transfer_profile_v1(profile_v1& profile,profile_transfer_v1 transfer,void* data);
void bind_profile_environment_v1(profile_v1& profile,std::string environment);
}
