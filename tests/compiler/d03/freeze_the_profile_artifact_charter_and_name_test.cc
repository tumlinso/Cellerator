#include <Cellerator/compiler/profile/freeze_the_profile_artifact_charter_and_name_v1.hh>

#include <array>
#include <cassert>
#include <cstring>
#include <string_view>

int main() {
    using namespace cellerator::compiler::profile::v1;

    static_assert(sizeof(profile_artifact_header_v1) == 112u);
    static_assert(std::string_view(profile_artifact_suffix_v1) == ".ceprofile");
    constexpr std::array<std::array<unsigned char, 8>, 4> reserved_magics{{
        {{'C', 'E', 'L', 'L', 'P', 'K', '0', '1'}},
        {{'C', 'E', 'L', 'L', 'C', 'S', 'G', '1'}},
        {{'C', 'E', 'L', 'L', 'E', 'X', '0', '2'}},
        {{'C', 'E', 'O', 'R', 'C', 'L', '1', 0}}}};
    for (const auto &reserved : reserved_magics)
        assert(std::memcmp(profile_artifact_magic_v1, reserved.data(),
                           reserved.size()) != 0);

    auto header = make_profile_artifact_header_v1(1u, 2u, 3u, 4u, 5u);
    assert(validate_profile_artifact_charter_v1(header)
           == profile_artifact_charter_status_v1::ok);
    header.schema_version = 2u;
    assert(validate_profile_artifact_charter_v1(header)
           == profile_artifact_charter_status_v1::unsupported_schema);
    header.schema_version = profile_artifact_schema_version_v1;
    header.flags = profile_artifact_flag_none;
    assert(validate_profile_artifact_charter_v1(header)
           == profile_artifact_charter_status_v1::not_data_derived);
}
