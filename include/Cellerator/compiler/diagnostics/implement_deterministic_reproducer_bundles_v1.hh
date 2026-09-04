#pragma once
#include <cstdint>
#include <string>
#include <vector>
namespace cellerator::compiler::diagnostics::v1 {enum class bundle_entry_kind:std::uint8_t{source_subset=0,profile,ceir_checkpoint,toolchain_manifest,pipeline,extension,diagnostic,command};struct bundle_entry{bundle_entry_kind kind=bundle_entry_kind::source_subset;std::string content;};struct reproducer_bundle{std::vector<bundle_entry> entries;std::uint64_t digest=0;bool contains_dataset_payload=false;};[[nodiscard]] reproducer_bundle make_reproducer_bundle(std::vector<bundle_entry>);[[nodiscard]] bool replay_matches(const reproducer_bundle&,std::uint64_t expected_digest) noexcept;}
