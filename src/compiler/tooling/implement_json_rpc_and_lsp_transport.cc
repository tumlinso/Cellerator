#include <Cellerator/compiler/tooling/implement_json_rpc_and_lsp_transport_v1.hh>

#include <algorithm>
#include <charconv>

namespace Cellerator::compiler::tooling {
namespace {
std::string json_string(std::string_view json, std::string_view key) {
    const auto marker = std::string("\"") + std::string(key) + "\"";
    const auto key_at = json.find(marker);
    const auto colon = key_at == std::string_view::npos ? key_at : json.find(':', key_at + marker.size());
    const auto quote = colon == std::string_view::npos ? colon : json.find('"', colon + 1);
    if (quote == std::string_view::npos) return {};
    const auto end = json.find('"', quote + 1);
    return end == std::string_view::npos ? std::string{} : std::string(json.substr(quote + 1, end - quote - 1));
}
} // namespace

std::string frame_json_rpc_v1(std::string_view json) {
    return "Content-Length: " + std::to_string(json.size()) + "\r\n\r\n" + std::string(json);
}

lsp_frame_result_v1 parse_json_rpc_frame_v1(std::string_view bytes,
                                             std::size_t maximum_content_length) {
    lsp_frame_result_v1 result;
    const auto header_end = bytes.find("\r\n\r\n");
    if (header_end == std::string_view::npos) return result;
    const auto prefix = std::string_view("Content-Length:");
    if (bytes.substr(0, prefix.size()) != prefix) {
        result.status = lsp_transport_status_v1::malformed_header;
        return result;
    }
    auto length_text = bytes.substr(prefix.size(), header_end - prefix.size());
    while (!length_text.empty() && length_text.front() == ' ') length_text.remove_prefix(1);
    std::size_t length = 0;
    const auto converted = std::from_chars(length_text.data(), length_text.data() + length_text.size(), length);
    if (converted.ec != std::errc{} || converted.ptr != length_text.data() + length_text.size()) {
        result.status = lsp_transport_status_v1::invalid_content_length;
        return result;
    }
    if (length > maximum_content_length) {
        result.status = lsp_transport_status_v1::oversized_message;
        return result;
    }
    const auto body = header_end + 4;
    if (bytes.size() < body + length) return result;
    result.message.json = std::string(bytes.substr(body, length));
    if (result.message.json.size() < 2 || result.message.json.front() != '{'
        || result.message.json.back() != '}') {
        result.status = lsp_transport_status_v1::invalid_json;
        return result;
    }
    result.message.id = json_string(result.message.json, "id");
    result.message.method = json_string(result.message.json, "method");
    result.message.notification = result.message.id.empty();
    result.consumed = body + length;
    result.status = lsp_transport_status_v1::success;
    return result;
}

lsp_transport_status_v1 lsp_transport_session_v1::accept(const json_rpc_message_v1 &message) {
    auto status = lsp_transport_status_v1::success;
    if (state_ == lsp_server_state_v1::pre_initialize && message.method != "initialize")
        status = lsp_transport_status_v1::request_before_initialize;
    else if (state_ == lsp_server_state_v1::running && message.method == "initialize")
        status = lsp_transport_status_v1::duplicate_initialize;
    else if ((state_ == lsp_server_state_v1::shutdown || state_ == lsp_server_state_v1::exited)
             && message.method != "exit")
        status = lsp_transport_status_v1::request_after_shutdown;
    else if (message.method == "initialize")
        state_ = lsp_server_state_v1::running;
    else if (message.method == "shutdown")
        state_ = lsp_server_state_v1::shutdown;
    else if (message.method == "exit")
        state_ = lsp_server_state_v1::exited;
    else if (message.method == "$/cancelRequest" && !message.id.empty())
        cancelled_ids_.push_back(message.id);
    logs_.push_back({status == lsp_transport_status_v1::success ? "info" : "error",
                     message.method, message.id});
    return status;
}

bool lsp_transport_session_v1::cancelled(std::string_view request_id) const {
    return std::find(cancelled_ids_.begin(), cancelled_ids_.end(), request_id)
        != cancelled_ids_.end();
}

} // namespace Cellerator::compiler::tooling
