#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace cellerator::compiler::api::v1 {

enum class diagnostic_severity_v1 : std::uint8_t { note, warning, error };

struct diagnostic_record_v1 {
    std::uint64_t session_id = 0;
    diagnostic_severity_v1 severity = diagnostic_severity_v1::note;
    std::string code;
    std::string message;
    std::string source;
    std::uint32_t line = 0;
};

struct timing_record_v1 {
    std::string phase;
    std::uint64_t elapsed_ns = 0;
};

struct explainability_report_v1 {
    std::uint64_t session_id = 0;
    std::vector<std::string> planning_decisions;
    std::vector<std::string> provenance;
    std::string reproducer;
    double progress = 0.0;
    std::vector<timing_record_v1> timings;
};

class diagnostic_stream_v1 {
  public:
    void emit(diagnostic_record_v1 record);
    [[nodiscard]] std::vector<diagnostic_record_v1>
    snapshot_for_session(std::uint64_t session_id) const;

  private:
    mutable std::mutex mutex_;
    std::vector<diagnostic_record_v1> records_;
};

class cancellation_token_v1 {
  public:
    void request() noexcept { requested_.store(true, std::memory_order_release); }
    [[nodiscard]] bool requested() const noexcept {
        return requested_.load(std::memory_order_acquire);
    }

  private:
    std::atomic<bool> requested_{false};
};

[[nodiscard]] std::string make_reproducer_v1(
    const std::string& compiler_version,
    const std::string& source_digest,
    const std::vector<std::string>& options);

}  // namespace cellerator::compiler::api::v1
