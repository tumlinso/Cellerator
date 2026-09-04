#include <Cellerator/compiler/ir/realization/implement_physical_plane_representation_v1.hh>

#include <set>
#include <tuple>

namespace cellerator::compiler::ir::realization::v1 {
namespace {

physical_plane_status_v1 fail(
    physical_plane_status_v1 status, std::string* error, const char* message) noexcept {
    if (error != nullptr) {
        *error = message;
    }
    return status;
}

bool generation_plane(physical_plane_kind_v1 kind) noexcept {
    return kind == physical_plane_kind_v1::values ||
        kind == physical_plane_kind_v1::active_support ||
        kind == physical_plane_kind_v1::gradients ||
        kind == physical_plane_kind_v1::partials;
}

} // namespace

physical_plane_status_v1 validate_physical_plane_set_v1(
    const physical_plane_set_v1& set, std::string* error) noexcept {
    if (!valid(set.identity) || set.planes.empty()) {
        return fail(physical_plane_status_v1::invalid_identity, error,
            "plane-set identity and planes are required");
    }
    std::set<std::tuple<std::uint64_t, std::uint64_t>> identities;
    stable_identity_v1 structure_identity{};
    std::uint64_t structure_epoch = 0u;
    bool saw_structure = false;
    for (const auto& plane : set.planes) {
        if (!valid(plane.identity) || !valid(plane.artifact_identity) ||
            !valid(plane.structure_identity)) {
            return fail(physical_plane_status_v1::invalid_identity, error,
                "plane, artifact, and structure identities are required");
        }
        if (!identities.emplace(plane.identity.high, plane.identity.low).second) {
            return fail(physical_plane_status_v1::duplicate_plane, error,
                "plane identities must be unique");
        }
        if (plane.structure_epoch == 0u) {
            return fail(physical_plane_status_v1::invalid_epoch, error,
                "structure epoch is required");
        }
        if (plane.residency_requirements == 0u) {
            return fail(physical_plane_status_v1::invalid_residency, error,
                "residency requirements are required");
        }
        if (plane.kind == physical_plane_kind_v1::structure) {
            if (plane.mutable_values || plane.value_generation != 0u ||
                plane.lifetime != plane_lifetime_v1::structure_epoch) {
                return fail(physical_plane_status_v1::invalid_lifetime, error,
                    "structure is immutable for a structure epoch");
            }
            structure_identity = plane.structure_identity;
            structure_epoch = plane.structure_epoch;
            saw_structure = true;
        }
        if (generation_plane(plane.kind) &&
            (!plane.mutable_values || plane.value_generation == 0u ||
             plane.lifetime != plane_lifetime_v1::value_generation)) {
            return fail(physical_plane_status_v1::invalid_generation, error,
                "mutable planes require an explicit value generation lifetime");
        }
    }
    if (!saw_structure) {
        return fail(physical_plane_status_v1::invalid_identity, error,
            "one structure plane is required");
    }
    for (const auto& plane : set.planes) {
        if (!(plane.structure_identity == structure_identity) ||
            plane.structure_epoch != structure_epoch) {
            return fail(physical_plane_status_v1::structure_mismatch, error,
                "all planes must reference the same immutable structure epoch");
        }
    }
    if (error != nullptr) {
        error->clear();
    }
    return physical_plane_status_v1::valid;
}

physical_plane_status_v1 advance_value_generation_v1(
    const physical_plane_set_v1& source,
    std::uint64_t generation,
    physical_plane_set_v1* output,
    std::string* error) noexcept {
    const auto status = validate_physical_plane_set_v1(source, error);
    if (status != physical_plane_status_v1::valid) {
        return status;
    }
    if (output == nullptr || generation == 0u) {
        return fail(physical_plane_status_v1::invalid_generation, error,
            "output and nonzero generation are required");
    }
    *output = source;
    for (auto& plane : output->planes) {
        if (plane.mutable_values) {
            plane.value_generation = generation;
        }
    }
    return validate_physical_plane_set_v1(*output, error);
}

} // namespace cellerator::compiler::ir::realization::v1
