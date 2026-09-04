#include <Cellerator/compiler/tooling/implement_clangd_worker_discovery_and_lifecycle_v1.hh>

#include <cassert>

using namespace Cellerator::compiler::tooling;

int main() {
    int launches = 0;
    int stops = 0;
    clangd_worker_hooks_v1 hooks;
    hooks.discover = [](const std::string &name) { return name == "missing" ? "" : "/usr/bin/" + name; };
    hooks.version = [](const std::string &path) { return path.find("old") != std::string::npos ? "clangd 17.0" : "clangd 18.1"; };
    hooks.launch = [&](const std::string &, const std::vector<std::string> &args) {
        ++launches; return args.empty() || args.front() != "--fail";
    };
    hooks.terminate = [&] { ++stops; };

    clangd_worker_lifecycle_v1 worker(hooks);
    assert(worker.start("missing", {}).status == clangd_worker_status_v1::missing);
    assert(worker.start("old", {}).status == clangd_worker_status_v1::incompatible);
    assert(worker.start("clangd", {"--background-index"}).status == clangd_worker_status_v1::running);
    assert(worker.crashed().status == clangd_worker_status_v1::crashed);
    assert(worker.restart().status == clangd_worker_status_v1::running);
    assert(worker.diagnostic().restart_count == 1);
    worker.stop();
    assert(worker.diagnostic().status == clangd_worker_status_v1::stopped);
    assert(launches == 2);
    assert(stops == 2);
}
