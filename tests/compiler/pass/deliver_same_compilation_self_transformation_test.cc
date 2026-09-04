#include <Cellerator/compiler/pass/deliver_same_compilation_self_transformation_v1.hh>

#include <cassert>
#include <filesystem>

namespace cp = cellerator::compiler::pass::v1;

int main() {
    const auto directory = std::filesystem::temp_directory_path() / "ce_ccp1_g02_017";
    std::filesystem::remove_all(directory);
    const cp::same_compilation_transform_request_v1 request{
        "#include <cstdint>\nextern \"C\" bool cellerator_self_transform(std::uint64_t* field) noexcept { *field *= 3; return true; }\n",
        "extern \"C\" int ordinary_object() { return 42; }\n", "g++", "pass-api-v1",
        directory.string(), "same.cell", 17, 9, 1, 2, true};
    const auto cold = cp::deliver_same_compilation_transform_v1(request);
    assert(cold.status == cp::same_compilation_transform_status_v1::success);
    assert(!cold.cache_hit && cold.reflected_field_after == 27);
    assert(cold.source_file == "same.cell" && cold.source_line == 17);
    assert(std::filesystem::is_regular_file(cold.ordinary_object));
    const auto warm = cp::deliver_same_compilation_transform_v1(request);
    assert(warm.status == cp::same_compilation_transform_status_v1::success);
    assert(warm.cache_hit && warm.reflected_field_after == 27);
    auto recursive = request;
    recursive.requested_generations = 3;
    assert(cp::deliver_same_compilation_transform_v1(recursive).status
        == cp::same_compilation_transform_status_v1::recursion_limit);
    auto fallback_request = request;
    fallback_request.prelude_transform_source =
        "#include <cstdint>\nextern \"C\" bool cellerator_self_transform(std::uint64_t*) noexcept { return false; }\n";
    const auto fallback = cp::deliver_same_compilation_transform_v1(fallback_request);
    assert(fallback.status == cp::same_compilation_transform_status_v1::success);
    assert(fallback.fallback_used && fallback.reflected_field_after == 9);
    assert(std::filesystem::is_regular_file(fallback.ordinary_object));
    std::filesystem::remove_all(directory);
}
