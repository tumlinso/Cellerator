#include <Cellerator/compiler/frontend/source/expose_source_pipeline_diagnostics_and_dumps_v1.hh>

#include <iostream>
#include <stdexcept>

using namespace Cellerator::compiler::frontend::source;

int main() {
    try {
        const auto tokens = build_raw_token_stream_v1(1, "x <[y]>\n", 2);
        source_dump_request_v1 request{true, true, true, true, "/secret/build", "$SRC"};
        const auto dump = render_source_pipeline_dump_v1(request,
            {"/secret/build/unit.cell", &tokens, "x (shadow())\n", "shadow:2 -> source:2"});
        const std::string expected =
            "source $SRC/unit.cell\n"
            "tokens\n0:1 ordinary x\n2:3 active <\n3:4 active [\n4:5 active y\n5:6 active ]\n6:7 active >\n"
            "activation 0 1 1 1 1 1\nshadow\nx (shadow())\n\nsource-map\nshadow:2 -> source:2\n";
        if (dump != expected || dump.find("/secret/build") != std::string::npos)
            throw std::runtime_error("dump snapshot or path remapping changed");
        if (!render_source_pipeline_dump_v1({}, {"/secret", &tokens, {}, {}}).empty())
            throw std::runtime_error("disabled cold dump produced a hot-path artifact");
        std::cout << "validated opt-in source diagnostics and path-remapped dumps\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
