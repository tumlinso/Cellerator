#include <Cellerator/compiler/frontend/cxx/expose_reusable_frontend_sessions_v1.hh>

#include <Cellerator/compiler/frontend/cxx/parse_shadow_translation_units_with_full_c_semantics_v1.hh>

#include <atomic>
#include <functional>
#include <thread>
#include <utility>

namespace Cellerator::compiler::frontend::cxx {

struct frontend_cancellation_token_v1::state {
    std::atomic<bool> cancelled{false};
};

struct immutable_frontend_snapshot_v1::implementation {
    shadow_translation_unit_v1 translation_unit;
    std::uint64_t parse_sequence = 0;
    std::uint64_t thread_id = 0;
};

struct reusable_frontend_session_v1::implementation {
    explicit implementation(const std::uint32_t major) noexcept : llvm_major(major) {}

    std::uint32_t llvm_major;
    std::atomic<std::uint64_t> next_sequence{1};
    std::atomic<std::uint64_t> completed_count{0};
};

frontend_cancellation_token_v1::frontend_cancellation_token_v1() noexcept
    : state_(std::make_shared<state>()) {}

void frontend_cancellation_token_v1::cancel() noexcept {
    state_->cancelled.store(true, std::memory_order_release);
}

bool frontend_cancellation_token_v1::is_cancelled() const noexcept {
    return state_->cancelled.load(std::memory_order_acquire);
}

immutable_frontend_snapshot_v1::immutable_frontend_snapshot_v1() noexcept = default;
immutable_frontend_snapshot_v1::~immutable_frontend_snapshot_v1() = default;
immutable_frontend_snapshot_v1::immutable_frontend_snapshot_v1(
    const immutable_frontend_snapshot_v1&) noexcept = default;
immutable_frontend_snapshot_v1& immutable_frontend_snapshot_v1::operator=(
    const immutable_frontend_snapshot_v1&) noexcept = default;
immutable_frontend_snapshot_v1::immutable_frontend_snapshot_v1(
    immutable_frontend_snapshot_v1&&) noexcept = default;
immutable_frontend_snapshot_v1& immutable_frontend_snapshot_v1::operator=(
    immutable_frontend_snapshot_v1&&) noexcept = default;

bool immutable_frontend_snapshot_v1::valid() const noexcept {
    return implementation_ != nullptr;
}

std::uint64_t immutable_frontend_snapshot_v1::sequence() const noexcept {
    return implementation_ == nullptr ? 0 : implementation_->parse_sequence;
}

std::uint64_t immutable_frontend_snapshot_v1::worker_thread_id() const noexcept {
    return implementation_ == nullptr ? 0 : implementation_->thread_id;
}

std::string_view immutable_frontend_snapshot_v1::virtual_filename() const noexcept {
    return implementation_ == nullptr ? std::string_view{}
                                      : implementation_->translation_unit.virtual_filename();
}

const std::vector<std::string>& immutable_frontend_snapshot_v1::errors() const noexcept {
    static const std::vector<std::string> empty;
    return implementation_ == nullptr ? empty : implementation_->translation_unit.errors();
}

const std::vector<std::string>& immutable_frontend_snapshot_v1::warnings() const noexcept {
    static const std::vector<std::string> empty;
    return implementation_ == nullptr ? empty : implementation_->translation_unit.warnings();
}

const upstream_clang_adapter_v1& immutable_frontend_snapshot_v1::adapter() const noexcept {
    static const upstream_clang_adapter_v1 empty;
    return implementation_ == nullptr ? empty : implementation_->translation_unit.adapter();
}

reusable_frontend_session_v1::reusable_frontend_session_v1(
    const std::uint32_t llvm_major) noexcept
    : implementation_(std::make_shared<implementation>(llvm_major)) {}

reusable_frontend_session_v1::~reusable_frontend_session_v1() = default;
reusable_frontend_session_v1::reusable_frontend_session_v1(
    const reusable_frontend_session_v1&) noexcept = default;
reusable_frontend_session_v1& reusable_frontend_session_v1::operator=(
    const reusable_frontend_session_v1&) noexcept = default;
reusable_frontend_session_v1::reusable_frontend_session_v1(
    reusable_frontend_session_v1&&) noexcept = default;
reusable_frontend_session_v1& reusable_frontend_session_v1::operator=(
    reusable_frontend_session_v1&&) noexcept = default;

std::uint32_t reusable_frontend_session_v1::llvm_major() const noexcept {
    return implementation_->llvm_major;
}

std::uint64_t reusable_frontend_session_v1::completed_parse_count() const noexcept {
    return implementation_->completed_count.load(std::memory_order_acquire);
}

reusable_frontend_session_status_v1 reusable_frontend_session_v1::parse(
    const reusable_frontend_parse_request_v1& request,
    immutable_frontend_snapshot_v1* snapshot) const noexcept {
    if (snapshot == nullptr) {
        return reusable_frontend_session_status_v1::null_output;
    }
    snapshot->implementation_.reset();
    if (request.schema_version != reusable_frontend_session_schema_version_v1) {
        return reusable_frontend_session_status_v1::schema_mismatch;
    }
    if (implementation_->llvm_major != 17 && implementation_->llvm_major != 18) {
        return reusable_frontend_session_status_v1::unsupported_llvm_major;
    }
    if (request.invocation == nullptr ||
        request.invocation->native_compiler_invocation() == nullptr) {
        return reusable_frontend_session_status_v1::missing_invocation;
    }
    if (request.source.empty()) {
        return reusable_frontend_session_status_v1::empty_source;
    }
    if (request.cancellation.is_cancelled()) {
        return reusable_frontend_session_status_v1::cancelled;
    }

    auto result = std::make_shared<immutable_frontend_snapshot_v1::implementation>();
    result->parse_sequence = implementation_->next_sequence.fetch_add(
        1, std::memory_order_relaxed);
    result->thread_id = static_cast<std::uint64_t>(
        std::hash<std::thread::id>{}(std::this_thread::get_id()));

    shadow_translation_unit_request_v1 shadow_request;
    shadow_request.llvm_major = implementation_->llvm_major;
    shadow_request.invocation = request.invocation;
    shadow_request.virtual_filename = request.virtual_filename;
    shadow_request.source = request.source;
    const auto status = parse_shadow_translation_unit_v1(
        shadow_request, &result->translation_unit);

    if (request.cancellation.is_cancelled()) {
        return reusable_frontend_session_status_v1::cancelled;
    }
    if (status == shadow_translation_unit_status_v1::clang_parse_failed) {
        return reusable_frontend_session_status_v1::clang_parse_failed;
    }
    if (status != shadow_translation_unit_status_v1::success &&
        status != shadow_translation_unit_status_v1::semantic_errors) {
        return reusable_frontend_session_status_v1::clang_parse_failed;
    }

    snapshot->implementation_ = std::move(result);
    implementation_->completed_count.fetch_add(1, std::memory_order_release);
    return status == shadow_translation_unit_status_v1::semantic_errors
        ? reusable_frontend_session_status_v1::semantic_errors
        : reusable_frontend_session_status_v1::success;
}

}  // namespace Cellerator::compiler::frontend::cxx
