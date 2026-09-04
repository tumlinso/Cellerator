#include <Cellerator/compiler/tooling/implement_json_rpc_and_lsp_transport_v1.hh>

#include <iostream>
#include <iterator>

int main() {
    const std::string input{std::istreambuf_iterator<char>(std::cin),
                            std::istreambuf_iterator<char>()};
    const auto parsed = Cellerator::compiler::tooling::parse_json_rpc_frame_v1(input);
    return parsed.status == Cellerator::compiler::tooling::lsp_transport_status_v1::success ? 0 : 1;
}
