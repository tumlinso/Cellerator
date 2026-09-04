#include <Cellerator/compiler/pass/implement_pipeline_configuration_syntax_and_apis_v1.hh>

#include <algorithm>
#include <sstream>

namespace cellerator::compiler::pass::v1 {
namespace {
bool parse_phase(std::string_view name, pipeline_phase_v1* phase) {
    for (std::size_t index = 0; index < pipeline_phase_count_v1; ++index) {
        const auto candidate = static_cast<pipeline_phase_v1>(index);
        if (pipeline_phase_name_v1(candidate) == name) {
            *phase = candidate;
            return true;
        }
    }
    return false;
}
}  // namespace

pipeline_builder_v1& pipeline_builder_v1::profile(std::string value) {
    configuration_.profile = std::move(value);
    return *this;
}
pipeline_builder_v1& pipeline_builder_v1::add(
    pipeline_stage_v1 stage, std::string name) {
    configuration_.passes.push_back({stage, std::move(name)});
    return *this;
}
pipeline_configuration_v1 pipeline_builder_v1::build() const {
    auto result = configuration_;
    result.origin = pipeline_configuration_origin_v1::cpp_api;
    return result;
}

bool parse_pipeline_configuration_v1(std::string_view text,
    pipeline_configuration_origin_v1 origin, pipeline_configuration_v1* output,
    std::string* error) noexcept {
    if (output == nullptr) return false;
    pipeline_configuration_v1 result{};
    result.origin = origin;
    const auto separator = text.find(';');
    std::string_view passes = text;
    if (separator != std::string_view::npos) {
        const auto prefix = text.substr(0, separator);
        if (prefix.substr(0, 8) != "profile=") {
            if (error) *error = "expected profile prefix";
            return false;
        }
        result.profile = std::string(prefix.substr(8));
        passes = text.substr(separator + 1);
    }
    while (!passes.empty()) {
        const auto comma = passes.find(',');
        const auto item = passes.substr(0, comma);
        const auto dot = item.find('.');
        const auto colon = item.find(':');
        if (dot == std::string_view::npos || colon == std::string_view::npos
            || dot > colon || colon + 1 == item.size()) {
            if (error) *error = "invalid pass item";
            return false;
        }
        pipeline_phase_v1 phase{};
        if (!parse_phase(item.substr(0, dot), &phase)) {
            if (error) *error = "unknown phase";
            return false;
        }
        interception_side_v1 side{};
        const auto side_text = item.substr(dot + 1, colon - dot - 1);
        if (side_text == "before") side = interception_side_v1::before;
        else if (side_text == "after") side = interception_side_v1::after;
        else {
            if (error) *error = "unknown interception side";
            return false;
        }
        result.passes.push_back({{phase, side}, std::string(item.substr(colon + 1))});
        if (comma == std::string_view::npos) break;
        passes.remove_prefix(comma + 1);
    }
    if (result.passes.empty()) {
        if (error) *error = "empty pipeline";
        return false;
    }
    *output = std::move(result);
    return true;
}

std::string print_pipeline_configuration_v1(
    const pipeline_configuration_v1& configuration) {
    std::ostringstream output;
    if (!configuration.profile.empty()) output << "profile=" << configuration.profile << ';';
    for (std::size_t index = 0; index < configuration.passes.size(); ++index) {
        if (index != 0) output << ',';
        const auto& pass = configuration.passes[index];
        output << pipeline_phase_name_v1(pass.stage.phase) << '.'
               << (pass.stage.side == interception_side_v1::before ? "before" : "after")
               << ':' << pass.name;
    }
    return output.str();
}

}  // namespace cellerator::compiler::pass::v1
