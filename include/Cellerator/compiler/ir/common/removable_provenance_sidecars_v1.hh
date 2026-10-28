#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace cellerator::compiler::ir {

struct hot_operation_record {
    std::uint32_t opcode{};
    std::uint32_t operand_begin{};
    std::uint32_t operand_count{};
    std::uint32_t result_begin{};
    std::uint32_t result_count{};
    std::uint32_t flags{};
};
struct source_span_sidecar { std::string file; std::uint32_t begin{}; std::uint32_t end{}; };
struct provenance_sidecar {
    source_span_sidecar source;
    std::vector<std::uint64_t> transform_lineage;
    std::vector<std::string> profile_evidence;
    std::vector<std::string> planner_decisions;
    std::vector<std::string> backend_mappings;
};

class provenance_sidecars {
public:
    void set(std::uint32_t operation, provenance_sidecar sidecar);
    const provenance_sidecar *get(std::uint32_t operation) const noexcept;
    void strip() noexcept;
    std::size_t size() const noexcept { return records_.size(); }
private:
    std::unordered_map<std::uint32_t, provenance_sidecar> records_;
};

std::uint64_t executable_semantic_hash(
    const std::vector<hot_operation_record> &operations) noexcept;

} // namespace cellerator::compiler::ir
