#include <Cellerator/compiler/backend/implement_backend_code_generation_plans_v1.hh>

#include <algorithm>
#include <sstream>
#include <string_view>

namespace cellerator::compiler::backend::v1 {
namespace {
template <class Range, class Name>
bool ordered_unique(const Range& range, Name name) {
    for (std::size_t i = 1; i < range.size(); ++i)
        if (!(name(range[i - 1]) < name(range[i]))) return false;
    return true;
}
bool has_file(const backend_codegen_plan_v1& plan, const std::string& path) {
    return std::any_of(plan.generated_files.begin(), plan.generated_files.end(),
                       [&](const auto& file) { return file.logical_path == path; });
}
}  // namespace

backend_codegen_plan_status_v1 validate_backend_codegen_plan_v1(
    const backend_codegen_plan_v1& plan) noexcept {
    if (plan.target.triple.data == nullptr || plan.target.triple.size == 0)
        return backend_codegen_plan_status_v1::invalid_target;
    if (plan.generated_files.empty() || plan.compile_jobs.empty() || plan.link_jobs.empty())
        return backend_codegen_plan_status_v1::missing_output;
    if (!ordered_unique(plan.generated_files, [](const auto& v) -> const auto& { return v.logical_path; }) ||
        !ordered_unique(plan.embedded_data, [](const auto& v) -> const auto& { return v.symbol; }) ||
        !ordered_unique(plan.compile_jobs, [](const auto& v) -> const auto& { return v.object_path; }) ||
        !ordered_unique(plan.link_jobs, [](const auto& v) -> const auto& { return v.output_path; }))
        return backend_codegen_plan_status_v1::unordered_or_duplicate;
    for (const auto& job : plan.compile_jobs)
        if (!has_file(plan, job.source_path) || job.object_path.empty())
            return backend_codegen_plan_status_v1::dangling_job_input;
    for (const auto& map : plan.source_maps)
        if (!has_file(plan, map.generated_path) || map.generated_line == 0 ||
            map.source_identity == 0)
            return backend_codegen_plan_status_v1::invalid_source_map;
    return backend_codegen_plan_status_v1::valid;
}

std::string snapshot_backend_codegen_plan_v1(const backend_codegen_plan_v1& plan) {
    std::ostringstream out;
    out << "target=" << std::string_view(plan.target.triple.data, plan.target.triple.size)
        << " keep_temps=" << (plan.keep_temporary_artifacts ? 1 : 0) << '\n';
    for (const auto& file : plan.generated_files)
        out << "file " << file.logical_path << ' ' << file.contents.size() << '\n';
    for (const auto& data : plan.embedded_data)
        out << "data " << data.symbol << ' ' << data.bytes.size() << '\n';
    for (const auto& job : plan.compile_jobs)
        out << "compile " << job.source_path << " -> " << job.object_path << '\n';
    for (const auto& job : plan.link_jobs)
        out << "link " << job.output_path << ' ' << job.object_paths.size() << ' '
            << job.support_libraries.size() << '\n';
    for (const auto& map : plan.source_maps)
        out << "map " << map.generated_path << ':' << map.generated_line << " <- "
            << map.source_identity << ':' << map.source_offset << '\n';
    return out.str();
}

}  // namespace cellerator::compiler::backend::v1
