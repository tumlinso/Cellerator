#include <Cellerator/execution/dependency/distributed.hh>

#include <array>
#include <cassert>
#include <cstddef>

namespace dep = cellerator::execution::dependency;

int main() {
    alignas(16) std::array<std::byte, 64> payload{};
    dep::ExtentRequirement required{{1, 0}, {2, 0}, {3, 0}, {4, 0}, 7, 11, 5,
                                    payload.size(), 16, 1};
    dep::DeviceExtentBinding actual{payload.data(), payload.size(),
        {cellerator::memory::domain::device, 1, -1, 0},
        {1, 0}, {2, 0}, {3, 0}, {4, 0}, 7, 11, 5, 11, 1};
    assert(dep::validate_extent(required, actual));
    auto check = [&](auto mutate, dep::DistributedBindingCode code) {
        auto invalid = actual; mutate(invalid);
        assert(dep::validate_extent(required, invalid).code == code);
    };
    check([](auto& x) { x.domain.low = 99; }, dep::DistributedBindingCode::wrong_domain);
    check([](auto& x) { x.logical_order.low = 99; }, dep::DistributedBindingCode::wrong_order);
    check([](auto& x) { x.physical_order.low = 99; }, dep::DistributedBindingCode::wrong_order);
    check([](auto& x) { ++x.structure_epoch; }, dep::DistributedBindingCode::stale_structure);
    check([](auto& x) { ++x.value_generation; }, dep::DistributedBindingCode::stale_values);
    check([](auto& x) { ++x.placement_revision; }, dep::DistributedBindingCode::stale_placement);
    check([](auto& x) { --x.where.device_ordinal; }, dep::DistributedBindingCode::wrong_device);
    check([](auto& x) { --x.ready_device; }, dep::DistributedBindingCode::wrong_device);
    check([](auto& x) { --x.ready_generation; }, dep::DistributedBindingCode::stale_readiness);
    check([](auto& x) { x.bytes = 1; }, dep::DistributedBindingCode::insufficient_capacity);
    check([](auto& x) { x.data = nullptr; }, dep::DistributedBindingCode::invalid_extent);
    check([](auto& x) { x.data = static_cast<const std::byte*>(x.data) + 1; },
          dep::DistributedBindingCode::invalid_alignment);
    auto bad_requirement = required; bad_requirement.alignment = 3;
    assert(dep::validate_extent(bad_requirement, actual).code
           == dep::DistributedBindingCode::invalid_alignment);
    std::array expected{required, required};
    std::array supplied{actual, actual}; supplied[1].ready_generation = 0;
    const auto status = dep::validate_extents(expected, supplied);
    assert(status.code == dep::DistributedBindingCode::stale_readiness && status.index == 1);
    assert(dep::validate_extents({}, {}));
    assert(dep::validate_extents(expected, {}).code == dep::DistributedBindingCode::count_mismatch);

    namespace gd = cellerator::geometry::dependency;
    namespace op = cellerator::ops::mechanism;
    const gd::ReadDependency read{{{20, 0}, {1, 0}, {21, 0}},
                                  {{22, 0}, {1, 0}, gd::SelectionSpace::coordinates}};
    const std::array reads{read};
    const std::array phases{op::PhaseDemand{op::Phase::condition, reads, {}}};
    const op::AccessContract access{{30, 0}, 1, {31, 0}, phases, {}};
    std::array transfers{dep::TransferRequirement{op::Phase::condition, read, 1, 0, 64}};
    assert(dep::validate_transfer_requirements(access, transfers));
    transfers[0].phase = op::Phase::primal;
    assert(!dep::validate_transfer_requirements(access, transfers));
    transfers[0].phase = op::Phase::condition;
    ++transfers[0].source.source.representation.low;
    assert(!dep::validate_transfer_requirements(access, transfers));
}
