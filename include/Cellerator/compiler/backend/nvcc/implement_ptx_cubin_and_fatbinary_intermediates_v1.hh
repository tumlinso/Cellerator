#pragma once
#include <cstdint>
#include <optional>
#include <string>
#include <vector>
namespace cellerator::compiler::backend::nvcc::v1 {
enum class artifact_kind:std::uint8_t{ptx=1,cubin,fatbinary};
struct backend_artifact{artifact_kind kind=artifact_kind::ptx;std::string path,toolchain;std::uint32_t architecture=0;std::uint64_t content_hash=0;bool embedded=false;};
struct artifact_bundle{std::vector<backend_artifact> artifacts;};
enum class artifact_status:std::uint8_t{ok=0,invalid_artifact,duplicate_artifact,missing_ptx,architecture_mismatch};
[[nodiscard]] artifact_status validate_artifact_bundle(const artifact_bundle&) noexcept;
[[nodiscard]] const backend_artifact* select_artifact(const artifact_bundle&,std::uint32_t architecture) noexcept;
}
