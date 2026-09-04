#include <Cellerator/compiler/frontend/cxx/expose_reusable_frontend_sessions_v1.hh>

#include <future>
#include <iostream>
#include <set>
#include <string>
#include <vector>

namespace cxx = Cellerator::compiler::frontend::cxx;

int main() {
    cxx::cxx_compilation_invocation_request_v1 invocation_request;
    invocation_request.llvm_major = 18;
    invocation_request.language = cxx::cxx_language_mode_v1::cxx20;
    invocation_request.clang_driver_path = "/usr/bin/clang++-18";
    invocation_request.target_triple = "x86_64-pc-linux-gnu";
    invocation_request.sysroot = "/";
    cxx::cxx_compilation_invocation_v1 invocation;
    if (cxx::create_cxx_compilation_invocation_v1(invocation_request, &invocation) !=
        cxx::cxx_compilation_invocation_status_v1::success) {
        std::cerr << "invocation construction failed\n";
        return 1;
    }

    cxx::reusable_frontend_session_v1 session(18);
    constexpr std::size_t worker_count = 4;
    std::vector<std::future<std::pair<cxx::reusable_frontend_session_status_v1,
                                     cxx::immutable_frontend_snapshot_v1>>> futures;
    for (std::size_t index = 0; index < worker_count; ++index) {
        futures.push_back(std::async(std::launch::async, [&, index] {
            cxx::reusable_frontend_parse_request_v1 request;
            request.invocation = &invocation;
            request.virtual_filename = "concurrent_" + std::to_string(index) + ".cc";
            request.source = "template<class T> constexpr T twice(T x){return x+x;} "
                             "static_assert(twice(" + std::to_string(index) + ") == " +
                             std::to_string(index * 2) + ");";
            cxx::immutable_frontend_snapshot_v1 snapshot;
            const auto status = session.parse(request, &snapshot);
            return std::make_pair(status, std::move(snapshot));
        }));
    }

    std::set<std::uint64_t> sequences;
    std::set<std::string> filenames;
    std::vector<cxx::immutable_frontend_snapshot_v1> retained;
    for (auto& future : futures) {
        auto [status, snapshot] = future.get();
        if (status != cxx::reusable_frontend_session_status_v1::success ||
            !snapshot.valid() || !snapshot.errors().empty() ||
            cxx::validate_upstream_clang_adapter_v1(snapshot.adapter()) !=
                cxx::upstream_clang_adapter_status_v1::success) {
            std::cerr << "concurrent parse failed\n";
            return 1;
        }
        sequences.insert(snapshot.sequence());
        filenames.emplace(snapshot.virtual_filename());
        retained.push_back(snapshot);
    }
    if (sequences.size() != worker_count || filenames.size() != worker_count ||
        session.completed_parse_count() != worker_count) {
        std::cerr << "session did not isolate concurrent parses\n";
        return 1;
    }

    const auto immutable_name = std::string(retained.front().virtual_filename());
    cxx::immutable_frontend_snapshot_v1 copied = retained.front();
    retained.clear();
    if (!copied.valid() || copied.virtual_filename() != immutable_name) {
        std::cerr << "snapshot lifetime was not immutable and shared\n";
        return 1;
    }

    cxx::reusable_frontend_parse_request_v1 cancelled_request;
    cancelled_request.invocation = &invocation;
    cancelled_request.source = "int cancelled;";
    cancelled_request.cancellation.cancel();
    cxx::immutable_frontend_snapshot_v1 cancelled_snapshot;
    if (session.parse(cancelled_request, &cancelled_snapshot) !=
            cxx::reusable_frontend_session_status_v1::cancelled ||
        cancelled_snapshot.valid() || session.completed_parse_count() != worker_count) {
        std::cerr << "cancelled parse was published\n";
        return 1;
    }
    return 0;
}
