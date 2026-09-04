#include <Cellerator/compiler/api/define_c_compiler_session_api_v1.hh>

#include <new>
#include <string>
#include <vector>

struct cellerator_compiler_session_v1 {
    cellerator_compiler_config_v1 config{};
    std::vector<std::string> sources;
    std::string object_identity;
};

extern "C" cellerator_compiler_session_v1* cellerator_compiler_session_create_v1(
    const cellerator_compiler_config_v1* config) {
    if (config == nullptr || config->struct_size < sizeof(*config)
        || config->target == nullptr || config->toolchain == nullptr) return nullptr;
    auto* session = new (std::nothrow) cellerator_compiler_session_v1;
    if (session != nullptr) session->config = *config;
    return session;
}

extern "C" void cellerator_compiler_session_destroy_v1(
    cellerator_compiler_session_v1* session) { delete session; }

extern "C" int cellerator_compiler_session_add_source_buffer_v1(
    cellerator_compiler_session_v1* session, const char* name, const char* data, size_t size) {
    if (session == nullptr || name == nullptr || data == nullptr) return 0;
    session->sources.emplace_back(std::string(name) + ":" + std::string(data, size));
    return 1;
}

extern "C" int cellerator_compiler_session_add_source_file_v1(
    cellerator_compiler_session_v1* session, const char* path) {
    if (session == nullptr || path == nullptr) return 0;
    session->sources.emplace_back(std::string("file:") + path);
    return 1;
}

extern "C" int cellerator_compiler_session_compile_v1(
    cellerator_compiler_session_v1* session, cellerator_compiler_output_v1* output) {
    if (session == nullptr || output == nullptr || output->struct_size < sizeof(*output)) return 0;
    if (session->config.cancelled != nullptr
        && session->config.cancelled(session->config.user_data)) return 0;
    if (session->sources.empty()) {
        if (session->config.diagnostic != nullptr) session->config.diagnostic(
            2, "compiler session has no source input", session->config.user_data);
        return 0;
    }
    session->object_identity = std::string(session->config.target) + ":"
        + session->config.toolchain + ":" + std::to_string(session->sources.size());
    output->object_identity = session->object_identity.c_str();
    output->source_count = session->sources.size();
    return 1;
}
