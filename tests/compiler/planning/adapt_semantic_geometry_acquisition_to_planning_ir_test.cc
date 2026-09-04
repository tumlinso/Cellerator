#include <Cellerator/compiler/planning/adapt_semantic_geometry_acquisition_to_planning_ir_v1.hh>

#include <array>
#include <cassert>
#include <cstring>

namespace planning = Cellerator::compiler::planning;

int main() {
    const std::array<planning::geometry_acquisition_kind_v1, 4> kinds{
        planning::geometry_acquisition_kind_v1::compile_now,
        planning::geometry_acquisition_kind_v1::precompiled_semantic_geometry,
        planning::geometry_acquisition_kind_v1::external_exact_cover,
        planning::geometry_acquisition_kind_v1::conventional_fallback,
    };

    for (const auto kind : kinds) {
        planning::geometry_acquisition_request_v1 request{};
        request.kind = kind;
        request.request_identity = {1u, static_cast<std::uint64_t>(kind)};
        request.semantic_problem_identity = {2u, 20u};
        request.profile_identity = {3u, 30u};
        request.target_identity = {4u, 40u};
        request.required_compatibility = planning::compatible_semantics_v1 |
            planning::compatible_profile_v1 | planning::compatible_target_v1;
        if (kind == planning::geometry_acquisition_kind_v1::precompiled_semantic_geometry ||
            kind == planning::geometry_acquisition_kind_v1::external_exact_cover) {
            request.supplied_geometry_identity = {5u, 50u};
        }
        if (kind == planning::geometry_acquisition_kind_v1::external_exact_cover) {
            request.required_compatibility |= planning::exact_logical_coverage_v1;
        }
        request.maximum_acquisition_cost = {100u, 200u, 50u, 4096u, 1024u};
        assert(planning::validate_geometry_acquisition_request_v1(request) ==
               planning::geometry_acquisition_validation_code_v1::ok);

        const auto request_bytes = planning::encode_csg1_request_v1(request);
        planning::geometry_acquisition_request_v1 decoded_request{};
        assert(planning::decode_csg1_request_v1(request_bytes.data(), request_bytes.size(),
                   &decoded_request) == planning::geometry_acquisition_validation_code_v1::ok);
        assert(std::memcmp(&request, &decoded_request, sizeof(request)) == 0);

        planning::geometry_acquisition_result_v1 result{};
        result.kind = kind;
        result.status = planning::geometry_acquisition_status_v1::acquired;
        result.request_identity = request.request_identity;
        result.semantic_geometry_identity = {6u, 60u};
        result.provider_identity = {7u, 70u};
        result.satisfied_compatibility = request.required_compatibility;
        result.measured_acquisition_cost = {80u, 150u, 40u, 2048u, 512u};
        assert(planning::validate_geometry_acquisition_result_v1(request, result) ==
               planning::geometry_acquisition_validation_code_v1::ok);

        const auto result_bytes = planning::encode_csg1_result_v1(result);
        planning::geometry_acquisition_result_v1 decoded_result{};
        assert(planning::decode_csg1_result_v1(result_bytes.data(), result_bytes.size(),
                   &decoded_result) == planning::geometry_acquisition_validation_code_v1::ok);
        assert(std::memcmp(&result, &decoded_result, sizeof(result)) == 0);
    }

    planning::geometry_acquisition_request_v1 invalid_exact{};
    invalid_exact.request_identity = {1u, 1u};
    invalid_exact.semantic_problem_identity = {2u, 2u};
    invalid_exact.profile_identity = {3u, 3u};
    invalid_exact.target_identity = {4u, 4u};
    invalid_exact.supplied_geometry_identity = {5u, 5u};
    invalid_exact.kind = planning::geometry_acquisition_kind_v1::external_exact_cover;
    invalid_exact.required_compatibility = planning::compatible_semantics_v1;
    assert(planning::validate_geometry_acquisition_request_v1(invalid_exact) ==
           planning::geometry_acquisition_validation_code_v1::exact_cover_not_required);
}
