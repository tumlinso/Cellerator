#include <Cellerator/compiler/tooling/implement_json_rpc_and_lsp_transport_v1.hh>

#include <cassert>

using namespace Cellerator::compiler::tooling;

int main() {
    const std::string initialize =
        R"({"jsonrpc":"2.0","id":"1","method":"initialize","params":{"capabilities":{}}})";
    const auto framed = frame_json_rpc_v1(initialize);
    const auto parsed = parse_json_rpc_frame_v1(framed);
    assert(parsed.status == lsp_transport_status_v1::success);
    assert(parsed.consumed == framed.size());
    assert(parsed.message.id == "1");
    assert(parsed.message.method == "initialize");

    assert(parse_json_rpc_frame_v1("Content-Length: x\r\n\r\n{}").status
           == lsp_transport_status_v1::invalid_content_length);
    assert(parse_json_rpc_frame_v1("Other: 2\r\n\r\n{}").status
           == lsp_transport_status_v1::malformed_header);
    assert(parse_json_rpc_frame_v1("Content-Length: 99\r\n\r\n{}").status
           == lsp_transport_status_v1::incomplete);
    assert(parse_json_rpc_frame_v1("Content-Length: 3\r\n\r\nxxx").status
           == lsp_transport_status_v1::invalid_json);

    lsp_transport_session_v1 session;
    assert(session.accept({"2", "textDocument/hover", "{}", false})
           == lsp_transport_status_v1::request_before_initialize);
    assert(session.accept(parsed.message) == lsp_transport_status_v1::success);
    assert(session.state() == lsp_server_state_v1::running);
    assert(session.accept({"7", "$/cancelRequest", "{}", true})
           == lsp_transport_status_v1::success);
    assert(session.cancelled("7"));
    assert(session.accept({"3", "shutdown", "{}", false})
           == lsp_transport_status_v1::success);
    assert(session.accept({{}, "exit", "{}", true}) == lsp_transport_status_v1::success);
    assert(session.state() == lsp_server_state_v1::exited);
    assert(session.logs().size() == 5);
}
