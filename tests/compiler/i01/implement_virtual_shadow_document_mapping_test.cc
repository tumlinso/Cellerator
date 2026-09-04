#include <Cellerator/compiler/tooling/implement_virtual_shadow_document_mapping_v1.hh>

#include <cassert>

using namespace Cellerator::compiler::tooling;

int main() {
    const std::string source = "cell Gene { field count; }\nMACRO(Gene)\n";
    virtual_shadow_document_v1 document("file:///model.cell", source);
    document.append_generated("namespace __cell {\n");
    assert(document.replace_original({0, 4}, "stru"));
    assert(document.append_original({4, 26}));
    document.append_generated("}\n");
    assert(document.append_original({26, source.size()}));

    const auto field_shadow = document.to_shadow({12, 23});
    assert(field_shadow);
    assert(document.to_original(*field_shadow)->begin == 12);
    const auto macro_shadow = document.to_shadow({26, source.size()});
    assert(macro_shadow);
    assert(document.to_original(*macro_shadow)->end == source.size());

    const auto fix = document.map_edit_to_original({*field_shadow, "field value;"});
    assert(fix && fix->range.begin == 12 && fix->replacement == "field value;");
    assert(!document.to_original(document_span_v1{0, 5}));
    assert(!document.replace_original({0, 4}, "struct"));
}
