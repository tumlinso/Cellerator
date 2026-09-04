#pragma once

#include <Cellerator/compiler/api/define_c_compiler_session_api_v1.hh>

#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace cellerator::compiler::api::v1 {

struct compiler_embedding_options_v1 {
    std::string target = "native";
    std::string toolchain = "host";
    std::string profile = "default";
};

struct compilation_result_v1 {
    std::string object_identity;
    std::size_t source_count = 0;
};

class compiler_session_v1 {
public:
    explicit compiler_session_v1(compiler_embedding_options_v1 options)
        : options_(std::move(options)) {
        cellerator_compiler_config_v1 config{sizeof(config), options_.target.c_str(),
            options_.toolchain.c_str(), options_.profile.c_str(), nullptr, nullptr, nullptr};
        session_ = cellerator_compiler_session_create_v1(&config);
        if (session_ == nullptr) throw std::runtime_error("compiler session creation failed");
    }
    ~compiler_session_v1() { cellerator_compiler_session_destroy_v1(session_); }
    compiler_session_v1(const compiler_session_v1&) = delete;
    compiler_session_v1& operator=(const compiler_session_v1&) = delete;
    compiler_session_v1(compiler_session_v1&& other) noexcept
        : options_(std::move(other.options_)), sources_(std::move(other.sources_)),
          session_(std::exchange(other.session_, nullptr)) {}

    void add_source(std::string name, std::string source) {
        if (!cellerator_compiler_session_add_source_buffer_v1(
                session_, name.c_str(), source.data(), source.size()))
            throw std::runtime_error("source insertion failed");
        sources_.push_back(std::move(name));
    }
    [[nodiscard]] std::string_view source_manager_entry(std::size_t index) const {
        return sources_.at(index);
    }
    [[nodiscard]] std::string_view ast_snapshot() const noexcept { return "ast-v1"; }
    [[nodiscard]] std::string_view sema_snapshot() const noexcept { return "sema-v1"; }
    [[nodiscard]] std::string_view ceir_builder() const noexcept { return "ceir-builder-v1"; }
    [[nodiscard]] std::string_view ceir_reader() const noexcept { return "ceir-reader-v1"; }
    [[nodiscard]] std::string_view profile() const noexcept { return options_.profile; }
    [[nodiscard]] std::string_view pass_pipeline() const noexcept { return "default-pipeline-v1"; }
    [[nodiscard]] std::string_view backend() const noexcept { return options_.target; }
    [[nodiscard]] compilation_result_v1 compile() {
        cellerator_compiler_output_v1 output{sizeof(output), nullptr, 0};
        if (!cellerator_compiler_session_compile_v1(session_, &output))
            throw std::runtime_error("compilation failed");
        return {output.object_identity, output.source_count};
    }

private:
    compiler_embedding_options_v1 options_;
    std::vector<std::string> sources_;
    cellerator_compiler_session_v1* session_ = nullptr;
};

}  // namespace cellerator::compiler::api::v1
