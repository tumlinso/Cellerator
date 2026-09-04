#include <Cellerator/compiler/api/freeze_libcellerator_sdk_acceptance_v1.hh>
namespace cellerator::compiler::api::v1 {bool sdk_public_header_is_dependency_clean_v1(std::string_view s)noexcept{return s.find("clang/")==s.npos&&s.find("llvm/")==s.npos&&s.find("clang::")==s.npos&&s.find("llvm::")==s.npos;}}
