#include <Cellerator/compiler/lto/implement_explicit_program_planning_authorization_v1.hh>

#include <cassert>

using namespace cellerator::compiler::lto::v1;

int main() {
    const artifact_identity_v1 producer{1, 2};
    const artifact_identity_v1 consumer{3, 4};
    cross_tu_planning_request_v1 request{
        producer, consumer, "producer.cc", "consumer.cc", true, true, true, true};

    program_planning_authorization_v1 named{
        program_planning_authority_v1::exported_or_named_fields,
        {producer, consumer}, {}, false};
    assert(authorize_cross_tu_program_planning_v1(request, named) ==
           program_planning_authorization_status_v1::authorized);

    request.consumer_is_exported_or_named = false;
    assert(authorize_cross_tu_program_planning_v1(request, named) ==
           program_planning_authorization_status_v1::authorization_missing);

    program_planning_authorization_v1 source_policy{
        program_planning_authority_v1::source_policy,
        {}, {"producer.cc", "consumer.cc"}, false};
    assert(authorize_cross_tu_program_planning_v1(request, source_policy) ==
           program_planning_authorization_status_v1::authorized);
    request.consumer_source = "ordinary.cc";
    assert(authorize_cross_tu_program_planning_v1(request, source_policy) ==
           program_planning_authorization_status_v1::source_not_authorized);

    program_planning_authorization_v1 driver{
        program_planning_authority_v1::driver_lto_flag, {}, {}, true};
    request.consumer_source = "consumer.cc";
    assert(authorize_cross_tu_program_planning_v1(request, driver) ==
           program_planning_authorization_status_v1::authorized);

    request.consumer_has_ceir = false;
    assert(authorize_cross_tu_program_planning_v1(request, driver) ==
           program_planning_authorization_status_v1::semantic_body_unavailable);

    request.consumer_has_ceir = true;
    assert(authorize_cross_tu_program_planning_v1(
               request, program_planning_authorization_v1{}) ==
           program_planning_authorization_status_v1::authorization_missing);
}
