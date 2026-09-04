#include <Cellerator/compiler/backend/nvcc/implement_ptx_cubin_and_fatbinary_intermediates_v1.hh>
#include <set>
namespace cellerator::compiler::backend::nvcc::v1 {
artifact_status validate_artifact_bundle(const artifact_bundle&b) noexcept{bool ptx=false;std::set<std::pair<unsigned,std::string>>seen;for(const auto&a:b.artifacts){if(a.path.empty()||a.toolchain.empty()||!a.architecture||!a.content_hash)return artifact_status::invalid_artifact;if(!seen.insert({static_cast<unsigned>(a.kind),a.path}).second)return artifact_status::duplicate_artifact;if(a.kind==artifact_kind::ptx)ptx=true;}return ptx?artifact_status::ok:artifact_status::missing_ptx;}
const backend_artifact* select_artifact(const artifact_bundle&b,std::uint32_t arch) noexcept{if(validate_artifact_bundle(b)!=artifact_status::ok)return nullptr;for(const auto&a:b.artifacts)if(a.kind==artifact_kind::cubin&&a.architecture==arch)return &a;for(const auto&a:b.artifacts)if(a.kind==artifact_kind::fatbinary&&a.architecture<=arch)return &a;for(const auto&a:b.artifacts)if(a.kind==artifact_kind::ptx&&a.architecture<=arch)return &a;return nullptr;}
}
