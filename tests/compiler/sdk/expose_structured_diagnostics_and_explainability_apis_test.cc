#include <Cellerator/sdk/expose_structured_diagnostics_and_explainability_apis_v1.hh>

#include <cassert>
#include <thread>
#include <vector>

namespace api = cellerator::compiler::api::v1;

int main() {
    api::diagnostic_stream_v1 diagnostics;
    std::vector<std::thread> producers;
    for (std::uint64_t session = 1; session <= 4; ++session) {
        producers.emplace_back([session, &diagnostics] {
            for (std::uint32_t line = 1; line <= 64; ++line) {
                diagnostics.emit({session, api::diagnostic_severity_v1::warning,
                                  "CE100", "candidate rejected", "model.cell", line});
            }
        });
    }
    for (auto& producer : producers) {
        producer.join();
    }
    for (std::uint64_t session = 1; session <= 4; ++session) {
        const auto records = diagnostics.snapshot_for_session(session);
        assert(records.size() == 64);
        for (const auto& record : records) {
            assert(record.session_id == session);
        }
    }

    api::explainability_report_v1 report;
    report.session_id = 2;
    report.planning_decisions = {"selected cpu/reference", "rejected cuda/no-device"};
    report.provenance = {"semantic-ir:sha256:abc", "profile:sha256:def"};
    report.reproducer = api::make_reproducer_v1("1.0", "abc", {"--target=cpu"});
    report.progress = 0.75;
    report.timings.push_back({"semantic-analysis", 500});
    assert(report.reproducer.find("--source-digest=abc") != std::string::npos);
    assert(report.progress == 0.75);

    api::cancellation_token_v1 cancellation;
    assert(!cancellation.requested());
    cancellation.request();
    assert(cancellation.requested());
}
