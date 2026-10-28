#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace Cellerator::compiler::tooling {

enum class clangd_worker_status_v1 : std::uint8_t {
    stopped, running, missing, incompatible, launch_failed, crashed
};

struct clangd_worker_diagnostic_v1 {
    clangd_worker_status_v1 status = clangd_worker_status_v1::stopped;
    std::string executable;
    std::string version;
    std::string message;
    std::uint32_t restart_count = 0;
};

struct clangd_worker_hooks_v1 {
    std::function<std::string(const std::string &)> discover;
    std::function<std::string(const std::string &)> version;
    std::function<bool(const std::string &, const std::vector<std::string> &)> launch;
    std::function<void()> terminate;
};

class clangd_worker_lifecycle_v1 {
public:
    explicit clangd_worker_lifecycle_v1(clangd_worker_hooks_v1 hooks,
                                        std::uint32_t required_major = 18);
    [[nodiscard]] clangd_worker_diagnostic_v1 start(
        std::string override_command, std::vector<std::string> forwarded_arguments);
    [[nodiscard]] clangd_worker_diagnostic_v1 crashed();
    [[nodiscard]] clangd_worker_diagnostic_v1 restart();
    void stop();
    [[nodiscard]] const clangd_worker_diagnostic_v1 &diagnostic() const noexcept;

private:
    clangd_worker_hooks_v1 hooks_;
    std::uint32_t required_major_ = 18;
    std::string command_;
    std::vector<std::string> arguments_;
    clangd_worker_diagnostic_v1 diagnostic_;
};

} // namespace Cellerator::compiler::tooling
