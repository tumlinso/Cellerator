#pragma once

#include <Cellerator/compiler/frontend/source/build_a_lossless_raw_token_stream_v1.hh>

#include <string>
#include <string_view>

namespace Cellerator::compiler::frontend::source {

struct source_dump_request_v1 {
    bool tokens = false;
    bool activation_map = false;
    bool shadow_source = false;
    bool source_map = false;
    std::string path_prefix;
    std::string remapped_prefix;
};
struct source_dump_inputs_v1 {
    std::string path;
    const raw_token_stream_v1* tokens = nullptr;
    std::string_view shadow_source;
    std::string_view source_map;
};

[[nodiscard]] std::string render_source_pipeline_dump_v1(const source_dump_request_v1& request,
                                                         const source_dump_inputs_v1& inputs);

} // namespace Cellerator::compiler::frontend::source
