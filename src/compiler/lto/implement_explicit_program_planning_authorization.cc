#include <Cellerator/compiler/lto/implement_explicit_program_planning_authorization_v1.hh>

#include <algorithm>

namespace cellerator::compiler::lto::v1 {
namespace {

bool identity_is_present(const artifact_identity_v1& identity) noexcept {
    return identity.high != 0 || identity.low != 0;
}

bool same_identity(
    const artifact_identity_v1& lhs,
    const artifact_identity_v1& rhs) noexcept {
    return lhs.high == rhs.high && lhs.low == rhs.low;
}

bool contains_identity(
    const std::vector<artifact_identity_v1>& identities,
    const artifact_identity_v1& identity) noexcept {
    return std::any_of(
        identities.begin(), identities.end(),
        [&](const artifact_identity_v1& candidate) {
            return same_identity(candidate, identity);
        });
}

bool contains_source(
    const std::vector<std::string>& sources,
    const std::string& source) noexcept {
    return !source.empty() &&
           std::find(sources.begin(), sources.end(), source) != sources.end();
}

}  // namespace

program_planning_authorization_status_v1
authorize_cross_tu_program_planning_v1(
    const cross_tu_planning_request_v1& request,
    const program_planning_authorization_v1& authorization) noexcept {
    if (!request.producer_has_ceir || !request.consumer_has_ceir) {
        return program_planning_authorization_status_v1::semantic_body_unavailable;
    }
    if (!identity_is_present(request.producer_field) ||
        !identity_is_present(request.consumer_field)) {
        return program_planning_authorization_status_v1::field_identity_missing;
    }

    switch (authorization.authority) {
    case program_planning_authority_v1::exported_or_named_fields:
        if (!request.producer_is_exported_or_named ||
            !request.consumer_is_exported_or_named) {
            return program_planning_authorization_status_v1::authorization_missing;
        }
        if (!authorization.authorized_fields.empty() &&
            (!contains_identity(authorization.authorized_fields,
                                request.producer_field) ||
             !contains_identity(authorization.authorized_fields,
                                request.consumer_field))) {
            return program_planning_authorization_status_v1::authorization_missing;
        }
        return program_planning_authorization_status_v1::authorized;

    case program_planning_authority_v1::source_policy:
        if (!contains_source(authorization.authorized_sources,
                             request.producer_source) ||
            !contains_source(authorization.authorized_sources,
                             request.consumer_source)) {
            return program_planning_authorization_status_v1::source_not_authorized;
        }
        return program_planning_authorization_status_v1::authorized;

    case program_planning_authority_v1::driver_lto_flag:
        return authorization.allow_all_ceir_fields
                   ? program_planning_authorization_status_v1::authorized
                   : program_planning_authorization_status_v1::authorization_missing;

    case program_planning_authority_v1::none:
        return program_planning_authorization_status_v1::authorization_missing;
    }
    return program_planning_authorization_status_v1::authorization_missing;
}

}  // namespace cellerator::compiler::lto::v1
