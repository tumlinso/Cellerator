#include <Cellerator/compiler/api/create_installed_sdk_examples_v1.hh>
#include <algorithm>
namespace cellerator::compiler::api::v1 {bool has_installed_sdk_example_v1(std::string_view n)noexcept{return std::find(installed_sdk_examples_v1.begin(),installed_sdk_examples_v1.end(),n)!=installed_sdk_examples_v1.end();}}
