#pragma once

#include <Cellerator/compiler/pass/freeze_the_pass_pipeline_stage_taxonomy_v1.hh>

#include <string>
#include <string_view>
#include <vector>

namespace cellerator::compiler::pass::v1 {

enum class pipeline_configuration_origin_v1 : std::uint8_t {
    source_directive = 1,
    inline_planning_ir,
    command_line,
    cpp_api,
    profile,
};

struct configured_pass_v1 {
    pipeline_stage_v1 stage{};
    std::string name;
    friend bool operator==(const configured_pass_v1& left,
        const configured_pass_v1& right) {
        return left.stage.phase == right.stage.phase
            && left.stage.side == right.stage.side && left.name == right.name;
    }
};

struct pipeline_configuration_v1 {
    std::string profile;
    std::vector<configured_pass_v1> passes;
    pipeline_configuration_origin_v1 origin =
        pipeline_configuration_origin_v1::command_line;
};

class pipeline_builder_v1 {
public:
    pipeline_builder_v1& profile(std::string value);
    pipeline_builder_v1& add(pipeline_stage_v1 stage, std::string name);
    [[nodiscard]] pipeline_configuration_v1 build() const;
private:
    pipeline_configuration_v1 configuration_{};
};

[[nodiscard]] bool parse_pipeline_configuration_v1(
    std::string_view text, pipeline_configuration_origin_v1 origin,
    pipeline_configuration_v1* output, std::string* error = nullptr) noexcept;

[[nodiscard]] std::string print_pipeline_configuration_v1(
    const pipeline_configuration_v1& configuration);

}  // namespace cellerator::compiler::pass::v1
