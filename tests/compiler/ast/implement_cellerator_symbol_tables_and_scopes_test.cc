#include <Cellerator/compiler/ast/implement_cellerator_symbol_tables_and_scopes_v1.hh>

#include <cassert>
#include <iostream>

using namespace Cellerator::compiler::ast;

int main() {
    symbol_scope_v1 root{0, invalid_symbol_scope_v1, "", {},
                         {{1, symbol_kind_v1::domain, "gene", 0, 10},
                          {2, symbol_kind_v1::profile, "training", 0, 10}}};
    symbol_scope_v1 imported_a{1, 0, "dataset_a", {},
                               {{10, symbol_kind_v1::relation, "edges", 0, 20},
                                {11, symbol_kind_v1::field, "propagate", 101, 20}}};
    symbol_scope_v1 imported_b{2, 0, "dataset_b", {},
                               {{20, symbol_kind_v1::relation, "edges", 0, 30},
                                {21, symbol_kind_v1::field, "propagate", 202, 30},
                                {22, symbol_kind_v1::ir_name, "lowered", 0, 30}}};
    symbol_scope_v1 function{3, 0, "analysis", {1, 2},
                             {{30, symbol_kind_v1::domain, "gene", 0, 40},
                              {31, symbol_kind_v1::axis, "cells", 0, 40},
                              {32, symbol_kind_v1::candidate, "kernel", 301, 40},
                              {33, symbol_kind_v1::candidate, "kernel", 302, 40},
                              {34, symbol_kind_v1::compiler_pass, "fuse", 0, 40}}};
    std::string error;
    auto table = freeze_symbol_table_v1({function, imported_b, root, imported_a}, &error);
    assert(table && error.empty());

    const auto shadowed = table->lookup({3, "gene", symbol_kind_v1::domain, {}, false});
    assert(shadowed.status == symbol_lookup_status_v1::resolved);
    assert(shadowed.candidates.front()->identity == 30);

    const auto overloads = table->lookup({3, "kernel", symbol_kind_v1::candidate, {}, false});
    assert(overloads.status == symbol_lookup_status_v1::overload_set);
    assert(overloads.candidates.size() == 2);
    const auto selected = table->lookup({3, "kernel", symbol_kind_v1::candidate, 302, false});
    assert(selected.status == symbol_lookup_status_v1::resolved);
    assert(selected.candidates.front()->identity == 33);

    const auto ambiguous_import = table->lookup({3, "edges", symbol_kind_v1::relation, {}, false});
    assert(ambiguous_import.status == symbol_lookup_status_v1::ambiguous);
    assert(ambiguous_import.candidates.size() == 2);
    const auto imported_overloads = table->lookup({3, "propagate", symbol_kind_v1::field, {}, false});
    assert(imported_overloads.status == symbol_lookup_status_v1::overload_set);
    const auto cross_file = table->lookup({3, "lowered", symbol_kind_v1::ir_name, {}, false});
    assert(cross_file.status == symbol_lookup_status_v1::resolved);
    assert(cross_file.candidates.front()->source_file_identity == 30);

    const auto qualified = table->lookup({2, "edges", symbol_kind_v1::relation, {}, true});
    assert(qualified.status == symbol_lookup_status_v1::resolved);
    assert(qualified.candidates.front()->identity == 20);
    const auto hidden = table->lookup({3, "training", symbol_kind_v1::profile, {}, true});
    assert(hidden.status == symbol_lookup_status_v1::not_found);

    auto duplicate = root;
    duplicate.declarations.push_back({99, symbol_kind_v1::domain, "gene", 0, 50});
    assert(!freeze_symbol_table_v1({duplicate}, &error));

    std::cout << "scopes=" << table->scope_count()
              << " imported_ambiguities=" << ambiguous_import.candidates.size() << '\n';
}
