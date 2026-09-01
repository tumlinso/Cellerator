#include <CellShard/interop/cellerator/execution_payload.hh>

#include <Cellerator/execution/opaque_artifact.hh>

#include <type_traits>

namespace bridge = cellshard::interop::cellerator;
namespace execution = cellerator::execution;

static_assert(std::is_standard_layout_v<bridge::execution_artifact_expected>);
static_assert(std::is_trivially_copyable_v<bridge::execution_artifact_expected>);
static_assert(std::is_standard_layout_v<bridge::validated_execution_artifact>);
static_assert(std::is_trivially_copyable_v<bridge::validated_execution_artifact>);

int main() {
    const cellshard::execution_payload_host empty_host{};
    const bridge::execution_artifact_expected expected{};
    bridge::validated_execution_artifact validated{};

    const execution::opaque_artifact_status status =
        bridge::validate_execution_artifact_host(
            empty_host, expected, &validated);
    if (status.code != execution::opaque_artifact_code::invalid_argument)
        return 1;
    if (validated.transport.dataset_identity != 0u)
        return 2;
    if (validated.image.host_image.image_base != nullptr)
        return 3;
    return 0;
}
