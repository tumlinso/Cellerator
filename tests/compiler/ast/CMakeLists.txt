cmake_minimum_required(VERSION 3.20)
project(cellerator_compiler_ast_tests LANGUAGES CXX)

enable_testing()

get_filename_component(CELLERATOR_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)

add_executable(cellerator_ast_node_ownership_and_lifetime_test
    ast_node_ownership_and_lifetime_test.cc
    "${CELLERATOR_ROOT}/src/compiler/ast/ast_node_ownership_and_lifetime.cc")
target_include_directories(cellerator_ast_node_ownership_and_lifetime_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_ast_node_ownership_and_lifetime_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_ast_node_ownership_and_lifetime_test PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME cellerator_ast_node_ownership_and_lifetime_test COMMAND cellerator_ast_node_ownership_and_lifetime_test)

add_executable(cellerator_source_level_ast_node_families_test
    source_level_ast_node_families_test.cc
    "${CELLERATOR_ROOT}/src/compiler/ast/source_level_ast_node_families.cc"
    "${CELLERATOR_ROOT}/src/compiler/ast/ast_node_ownership_and_lifetime.cc")
target_include_directories(cellerator_source_level_ast_node_families_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_source_level_ast_node_families_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_source_level_ast_node_families_test PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME cellerator_source_level_ast_node_families_test COMMAND cellerator_source_level_ast_node_families_test)

add_executable(cellerator_bind_c_ast_references_safely_test
    bind_c_ast_references_safely_test.cc
    "${CELLERATOR_ROOT}/src/compiler/ast/bind_c_ast_references_safely.cc")
target_include_directories(cellerator_bind_c_ast_references_safely_test PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(cellerator_bind_c_ast_references_safely_test PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cellerator_bind_c_ast_references_safely_test PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME cellerator_bind_c_ast_references_safely_test COMMAND cellerator_bind_c_ast_references_safely_test)

add_executable(cellerator_cellerator_symbol_tables_and_scopes_test
    cellerator_symbol_tables_and_scopes_test.cc
    ../../../src/compiler/ast/cellerator_symbol_tables_and_scopes.cc)
target_include_directories(cellerator_cellerator_symbol_tables_and_scopes_test PRIVATE ../../../include)
target_compile_features(cellerator_cellerator_symbol_tables_and_scopes_test PRIVATE cxx_std_17)
add_test(NAME cellerator_cellerator_symbol_tables_and_scopes_test COMMAND cellerator_cellerator_symbol_tables_and_scopes_test)

add_executable(cellerator_assign_deterministic_source_identities_test
    assign_deterministic_source_identities_test.cc
    ../../../src/compiler/ast/assign_deterministic_source_identities.cc)
target_include_directories(cellerator_assign_deterministic_source_identities_test PRIVATE ../../../include)
target_compile_features(cellerator_assign_deterministic_source_identities_test PRIVATE cxx_std_17)
add_test(NAME cellerator_assign_deterministic_source_identities_test COMMAND cellerator_assign_deterministic_source_identities_test)

add_executable(cellerator_token_and_macro_provenance_test
    token_and_macro_provenance_test.cc
    ../../../src/compiler/ast/token_and_macro_provenance.cc)
target_include_directories(cellerator_token_and_macro_provenance_test PRIVATE ../../../include)
target_compile_features(cellerator_token_and_macro_provenance_test PRIVATE cxx_std_17)
add_test(NAME cellerator_token_and_macro_provenance_test COMMAND cellerator_token_and_macro_provenance_test)

add_executable(cellerator_ast_visitors_matchers_and_queries_test
    ast_visitors_matchers_and_queries_test.cc
    ../../../src/compiler/ast/ast_visitors_matchers_and_queries.cc)
target_include_directories(cellerator_ast_visitors_matchers_and_queries_test PRIVATE ../../../include)
target_compile_features(cellerator_ast_visitors_matchers_and_queries_test PRIVATE cxx_std_17)
add_test(NAME cellerator_ast_visitors_matchers_and_queries_test COMMAND cellerator_ast_visitors_matchers_and_queries_test)

add_executable(cellerator_structured_frontend_diagnostic_records_test
    structured_frontend_diagnostic_records_test.cc
    ../../../src/compiler/ast/structured_frontend_diagnostic_records.cc)
target_include_directories(cellerator_structured_frontend_diagnostic_records_test PRIVATE ../../../include)
target_compile_features(cellerator_structured_frontend_diagnostic_records_test PRIVATE cxx_std_17)
add_test(NAME cellerator_structured_frontend_diagnostic_records_test COMMAND cellerator_structured_frontend_diagnostic_records_test)

add_executable(cellerator_source_aware_fix_its_test
    source_aware_fix_its_test.cc
    ../../../src/compiler/ast/source_aware_fix_its.cc)
target_include_directories(cellerator_source_aware_fix_its_test PRIVATE ../../../include)
target_compile_features(cellerator_source_aware_fix_its_test PRIVATE cxx_std_17)
add_test(NAME cellerator_source_aware_fix_its_test COMMAND cellerator_source_aware_fix_its_test)

add_executable(cellerator_deterministic_ast_dump_and_snapshot_formats_test
    deterministic_ast_dump_and_snapshot_formats_test.cc
    ../../../src/compiler/ast/deterministic_ast_dump_and_snapshot_formats.cc)
target_include_directories(cellerator_deterministic_ast_dump_and_snapshot_formats_test PRIVATE ../../../include)
target_compile_features(cellerator_deterministic_ast_dump_and_snapshot_formats_test PRIVATE cxx_std_17)
add_test(NAME cellerator_deterministic_ast_dump_and_snapshot_formats_test COMMAND cellerator_deterministic_ast_dump_and_snapshot_formats_test)

add_executable(cellerator_support_incremental_ast_identity_reuse_test
    support_incremental_ast_identity_reuse_test.cc
    ../../../src/compiler/ast/support_incremental_ast_identity_reuse.cc)
target_include_directories(cellerator_support_incremental_ast_identity_reuse_test PRIVATE ../../../include)
target_compile_features(cellerator_support_incremental_ast_identity_reuse_test PRIVATE cxx_std_17)
add_test(NAME cellerator_support_incremental_ast_identity_reuse_test COMMAND cellerator_support_incremental_ast_identity_reuse_test)

# Compile the installed SDK consumer against an install-shaped copy, not the source
# include tree, so the gate catches missing public headers and relative leaks.
set(C02_012_INSTALLED_INCLUDE "${CMAKE_CURRENT_BINARY_DIR}/c02_012_installed/include")
file(REMOVE_RECURSE "${C02_012_INSTALLED_INCLUDE}")
file(MAKE_DIRECTORY "${C02_012_INSTALLED_INCLUDE}")
file(COPY ../../../include/Cellerator DESTINATION "${C02_012_INSTALLED_INCLUDE}")
add_executable(cellerator_ast_and_source_diagnostics_interface_test
    ast_and_source_diagnostics_interface_test.cc
    ../../../src/compiler/ast/ast_and_source_diagnostics_interface.cc
    ../../../src/compiler/ast/ast_node_ownership_and_lifetime.cc
    ../../../src/compiler/ast/source_level_ast_node_families.cc
    ../../../src/compiler/ast/bind_c_ast_references_safely.cc
    ../../../src/compiler/ast/ast_visitors_matchers_and_queries.cc
    ../../../src/compiler/ast/token_and_macro_provenance.cc
    ../../../src/compiler/ast/structured_frontend_diagnostic_records.cc)
target_include_directories(cellerator_ast_and_source_diagnostics_interface_test PRIVATE "${C02_012_INSTALLED_INCLUDE}")
target_compile_features(cellerator_ast_and_source_diagnostics_interface_test PRIVATE cxx_std_17)
add_test(NAME cellerator_ast_and_source_diagnostics_interface_test COMMAND cellerator_ast_and_source_diagnostics_interface_test)
