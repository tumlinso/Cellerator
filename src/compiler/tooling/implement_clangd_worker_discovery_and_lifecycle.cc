#include <Cellerator/compiler/tooling/implement_clangd_worker_discovery_and_lifecycle_v1.hh>

#include <cctype>

namespace Cellerator::compiler::tooling {
namespace {
std::uint32_t major_version(const std::string &version) {
    std::size_t begin = 0;
    while (begin < version.size() && !std::isdigit(static_cast<unsigned char>(version[begin]))) ++begin;
    std::uint32_t value = 0;
    while (begin < version.size() && std::isdigit(static_cast<unsigned char>(version[begin])))
        value = value * 10u + static_cast<unsigned>(version[begin++] - '0');
    return value;
}
} // namespace

clangd_worker_lifecycle_v1::clangd_worker_lifecycle_v1(
    clangd_worker_hooks_v1 hooks, std::uint32_t required_major)
    : hooks_(std::move(hooks)), required_major_(required_major) {}

clangd_worker_diagnostic_v1 clangd_worker_lifecycle_v1::start(
    std::string override_command, std::vector<std::string> forwarded_arguments) {
    command_ = override_command.empty() ? "clangd" : std::move(override_command);
    arguments_ = std::move(forwarded_arguments);
    diagnostic_.executable = hooks_.discover ? hooks_.discover(command_) : std::string{};
    if (diagnostic_.executable.empty()) {
        diagnostic_.status = clangd_worker_status_v1::missing;
        diagnostic_.message = "compatible clangd executable not found";
        return diagnostic_;
    }
    diagnostic_.version = hooks_.version ? hooks_.version(diagnostic_.executable) : std::string{};
    if (major_version(diagnostic_.version) != required_major_) {
        diagnostic_.status = clangd_worker_status_v1::incompatible;
        diagnostic_.message = "clangd major version is incompatible";
        return diagnostic_;
    }
    if (!hooks_.launch || !hooks_.launch(diagnostic_.executable, arguments_)) {
        diagnostic_.status = clangd_worker_status_v1::launch_failed;
        diagnostic_.message = "clangd worker launch failed";
        return diagnostic_;
    }
    diagnostic_.status = clangd_worker_status_v1::running;
    diagnostic_.message = "clangd worker running";
    return diagnostic_;
}

clangd_worker_diagnostic_v1 clangd_worker_lifecycle_v1::crashed() {
    diagnostic_.status = clangd_worker_status_v1::crashed;
    diagnostic_.message = "clangd worker exited unexpectedly";
    return diagnostic_;
}

clangd_worker_diagnostic_v1 clangd_worker_lifecycle_v1::restart() {
    const auto restarts = diagnostic_.restart_count + 1;
    if (hooks_.terminate) hooks_.terminate();
    auto result = start(command_, arguments_);
    diagnostic_.restart_count = restarts;
    result.restart_count = restarts;
    return diagnostic_;
}

void clangd_worker_lifecycle_v1::stop() {
    if (diagnostic_.status == clangd_worker_status_v1::running && hooks_.terminate)
        hooks_.terminate();
    diagnostic_.status = clangd_worker_status_v1::stopped;
    diagnostic_.message = "clangd worker stopped";
}

const clangd_worker_diagnostic_v1 &clangd_worker_lifecycle_v1::diagnostic() const noexcept {
    return diagnostic_;
}

} // namespace Cellerator::compiler::tooling
