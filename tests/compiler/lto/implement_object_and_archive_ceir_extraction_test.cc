#include <Cellerator/compiler/lto/implement_object_and_archive_ceir_extraction_v1.hh>

#include <cassert>

using namespace cellerator::compiler::lto::v1;

namespace {

ceir_companion_artifact_v1 companion(
    artifact_identity_v1 field, const char* symbol) {
    ceir_companion_artifact_v1 result;
    result.format = object_format_v1::archive;
    result.semantic_summary = {10, 1};
    result.planning_summary = {10, 2};
    result.profile_reference = {10, 3};
    result.toolchain = {10, 4};
    result.content_hash[0] = 1;
    result.fields.push_back({field, symbol});
    result.placement = "archive-member-metadata";
    return result;
}

}  // namespace

int main() {
    std::vector<ceir_linker_input_v1> archive;
    archive.reserve(4097);
    for (std::uint64_t member = 1; member <= 4096; ++member) {
        archive.push_back({linker_input_kind_v1::static_archive_member,
                           "large.a", std::to_string(member), 256,
                           companion({0, member}, "field")});
    }
    archive.push_back(archive.front());

    ceir_extraction_index_v1 index;
    assert(extract_ceir_linker_inputs_v1(archive, &index) ==
           ceir_extraction_status_v1::valid);
    assert(index.fields.size() == 4096);
    assert(index.duplicate_members == 1);
    assert(index.bytes_scanned == archive.size() * 256);
    assert(index.peak_index_bytes >=
           index.fields.size() * sizeof(extracted_ceir_record_v1));
    assert(!index.loaded_native_code);

    archive.back().companion.fields.front().symbol = "conflict";
    assert(extract_ceir_linker_inputs_v1(archive, &index) ==
           ceir_extraction_status_v1::conflicting_duplicate);
}
