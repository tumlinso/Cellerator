#include <Cellerator/compiler/api/expose_structured_diagnostics_and_explainability_apis_v1.hh>

#include <sstream>

namespace cellerator::compiler::api::v1 {

void diagnostic_stream_v1::emit(diagnostic_record_v1 record) {
    std::lock_guard<std::mutex> lock(mutex_);
    records_.push_back(std::move(record));
}

std::vector<diagnostic_record_v1>
diagnostic_stream_v1::snapshot_for_session(std::uint64_t session_id) const {
    std::vector<diagnostic_record_v1> result;
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& record : records_) {
        if (record.session_id == session_id) {
            result.push_back(record);
        }
    }
    return result;
}

std::string make_reproducer_v1(const std::string& compiler_version,
                               const std::string& source_digest,
                               const std::vector<std::string>& options) {
    std::ostringstream stream;
    stream << "cellerator-compile --compiler-version=" << compiler_version
           << " --source-digest=" << source_digest;
    for (const auto& option : options) {
        stream << " " << option;
    }
    return stream.str();
}

}  // namespace cellerator::compiler::api::v1
