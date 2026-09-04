#include <Cellerator/compiler/ir/common/expose_the_public_ceir_c_api_v1.hh>

#include <cassert>

int main() {
    namespace api = cellerator::compiler::ceir;
    namespace ir = cellerator::compiler::ir;
    api::context context;
    context.register_extension("x", [](std::string_view name) {
        return name == "x.custom";
    });
    api::module_builder builder(context);
    ir::common_operation core;
    core.namespace_name = "semantic";
    core.operation_name = "apply";
    assert(builder.append(core));
    ir::common_operation extension;
    extension.namespace_name = "x";
    extension.operation_name = "custom";
    assert(builder.append(extension));
    const auto snapshot = builder.freeze();
    assert(snapshot.size() == 2u);
    std::size_t count = 0u;
    for (const auto &operation : snapshot) {
        assert(!operation.operation_name.empty());
        ++count;
    }
    assert(count == 2u);
    api::writer output;
    assert(output.canonical(snapshot).find("semantic.apply") != std::string::npos);
    assert(output.pretty(snapshot).find("x.custom") != std::string::npos);
    api::diagnostic diagnostic;
    extension.operation_name = "unknown";
    assert(!builder.append(extension, &diagnostic) && !diagnostic.message.empty());
}
