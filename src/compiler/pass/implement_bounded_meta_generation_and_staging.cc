#include <Cellerator/compiler/pass/implement_bounded_meta_generation_and_staging_v1.hh>

#include <algorithm>

namespace cellerator::compiler::pass::v1 {
namespace {
struct queued_transform {
    meta_transform_v1 transform;
    std::uint32_t depth = 0;
    std::vector<std::string> lineage;
};
}

meta_generation_receipt_v1 run_bounded_meta_generation_v1(
    const std::vector<meta_transform_v1>& roots, meta_generation_policy_v1 policy) {
    meta_generation_receipt_v1 receipt;
    std::vector<queued_transform> queue;
    for (const auto& root : roots) queue.push_back({root, 0, {root.name}});
    std::uint32_t generated_count = 0;
    for (std::size_t cursor = 0; cursor < queue.size(); ++cursor) {
        const auto current = queue[cursor];
        if (current.transform.name.empty() || current.transform.run == nullptr) {
            receipt.status = meta_generation_status_v1::invalid_transform;
            receipt.diagnostic = current.transform.name;
            return receipt;
        }
        receipt.execution_order.push_back(current.transform.name);
        std::vector<meta_transform_v1> generated;
        const meta_generation_context_v1 context{
            current.transform.phase, current.depth, current.transform.user_data};
        if (!current.transform.run(context, generated)) {
            receipt.status = meta_generation_status_v1::generation_failed;
            receipt.diagnostic = current.transform.name;
            return receipt;
        }
        for (auto& child : generated) {
            if (static_cast<unsigned>(child.phase)
                <= static_cast<unsigned>(current.transform.phase)) {
                receipt.status = meta_generation_status_v1::phase_violation;
                receipt.diagnostic = current.transform.name + " -> " + child.name;
                return receipt;
            }
            if (std::find(current.lineage.begin(), current.lineage.end(), child.name)
                != current.lineage.end()) {
                receipt.status = meta_generation_status_v1::cycle;
                receipt.diagnostic = current.transform.name + " -> " + child.name;
                return receipt;
            }
            if (current.depth + 1 > policy.maximum_depth) {
                receipt.status = meta_generation_status_v1::depth_limit;
                receipt.diagnostic = child.name;
                return receipt;
            }
            if (++generated_count > policy.maximum_generated_transforms) {
                receipt.status = meta_generation_status_v1::generation_limit;
                receipt.diagnostic = child.name;
                return receipt;
            }
            auto lineage = current.lineage;
            lineage.push_back(child.name);
            queue.push_back({std::move(child), current.depth + 1, std::move(lineage)});
        }
    }
    return receipt;
}

}  // namespace cellerator::compiler::pass::v1
