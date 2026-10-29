#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace Cellerator::compiler::tooling {

enum class lsp_transport_status_v1 : std::uint8_t {
    success, incomplete, malformed_header, invalid_content_length,
    oversized_message, invalid_json, request_before_initialize,
    duplicate_initialize, request_after_shutdown
};
enum class lsp_server_state_v1 : std::uint8_t { pre_initialize, running, shutdown, exited };

struct json_rpc_message_v1 {
    std::string id;
    std::string method;
    std::string json;
    bool notification = false;
};

struct lsp_frame_result_v1 {
    lsp_transport_status_v1 status = lsp_transport_status_v1::incomplete;
    std::size_t consumed = 0;
    json_rpc_message_v1 message;
};

struct lsp_log_record_v1 {
    std::string level;
    std::string event;
    std::string request_id;
};

[[nodiscard]] std::string frame_json_rpc_v1(std::string_view json);
[[nodiscard]] lsp_frame_result_v1 parse_json_rpc_frame_v1(
    std::string_view bytes, std::size_t maximum_content_length = 16u * 1024u * 1024u);

class lsp_transport_session_v1 {
public:
    [[nodiscard]] lsp_transport_status_v1 accept(const json_rpc_message_v1 &message);
    [[nodiscard]] lsp_server_state_v1 state() const noexcept { return state_; }
    [[nodiscard]] bool cancelled(std::string_view request_id) const;
    [[nodiscard]] const std::vector<lsp_log_record_v1> &logs() const noexcept { return logs_; }

private:
    lsp_server_state_v1 state_ = lsp_server_state_v1::pre_initialize;
    std::vector<std::string> cancelled_ids_;
    std::vector<lsp_log_record_v1> logs_;
};

} // namespace Cellerator::compiler::tooling
