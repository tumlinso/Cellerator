cmake_minimum_required(VERSION 3.20)
project(cellerator_compiler_ast_tests LANGUAGES CXX)

enable_testing()

get_filename_component(CELLERATOR_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)

add_executable(ce_ccp1_c02_001
    freeze_ast_node_ownership_and_lifetime_test.cc
    "${CELLERATOR_ROOT}/src/compiler/ast/freeze_ast_node_ownership_and_lifetime.cc")
target_include_directories(ce_ccp1_c02_001 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_c02_001 PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ce_ccp1_c02_001 PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME ce_ccp1_c02_001 COMMAND ce_ccp1_c02_001)

add_executable(ce_ccp1_c02_002
    define_source_level_ast_node_families_test.cc
    "${CELLERATOR_ROOT}/src/compiler/ast/define_source_level_ast_node_families.cc"
    "${CELLERATOR_ROOT}/src/compiler/ast/freeze_ast_node_ownership_and_lifetime.cc")
target_include_directories(ce_ccp1_c02_002 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_c02_002 PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ce_ccp1_c02_002 PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME ce_ccp1_c02_002 COMMAND ce_ccp1_c02_002)

add_executable(ce_ccp1_c02_003
    bind_c_ast_references_safely_test.cc
    "${CELLERATOR_ROOT}/src/compiler/ast/bind_c_ast_references_safely.cc")
target_include_directories(ce_ccp1_c02_003 PRIVATE "${CELLERATOR_ROOT}/include")
target_compile_features(ce_ccp1_c02_003 PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ce_ccp1_c02_003 PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME ce_ccp1_c02_003 COMMAND ce_ccp1_c02_003)

add_executable(ce_ccp1_c02_004
    implement_cellerator_symbol_tables_and_scopes_test.cc
    ../../../src/compiler/ast/implement_cellerator_symbol_tables_and_scopes.cc)
target_include_directories(ce_ccp1_c02_004 PRIVATE ../../../include)
target_compile_features(ce_ccp1_c02_004 PRIVATE cxx_std_17)
add_test(NAME ce_ccp1_c02_004 COMMAND ce_ccp1_c02_004)

add_executable(ce_ccp1_c02_005
    assign_deterministic_source_identities_test.cc
    ../../../src/compiler/ast/assign_deterministic_source_identities.cc)
target_include_directories(ce_ccp1_c02_005 PRIVATE ../../../include)
target_compile_features(ce_ccp1_c02_005 PRIVATE cxx_std_17)
add_test(NAME ce_ccp1_c02_005 COMMAND ce_ccp1_c02_005)

add_executable(ce_ccp1_c02_006
    preserve_token_and_macro_provenance_test.cc
    ../../../src/compiler/ast/preserve_token_and_macro_provenance.cc)
target_include_directories(ce_ccp1_c02_006 PRIVATE ../../../include)
target_compile_features(ce_ccp1_c02_006 PRIVATE cxx_std_17)
add_test(NAME ce_ccp1_c02_006 COMMAND ce_ccp1_c02_006)

add_executable(ce_ccp1_c02_007
    implement_ast_visitors_matchers_and_queries_test.cc
    ../../../src/compiler/ast/implement_ast_visitors_matchers_and_queries.cc)
target_include_directories(ce_ccp1_c02_007 PRIVATE ../../../include)
target_compile_features(ce_ccp1_c02_007 PRIVATE cxx_std_17)
add_test(NAME ce_ccp1_c02_007 COMMAND ce_ccp1_c02_007)

add_executable(ce_ccp1_c02_008
    create_structured_frontend_diagnostic_records_test.cc
    ../../../src/compiler/ast/create_structured_frontend_diagnostic_records.cc)
target_include_directories(ce_ccp1_c02_008 PRIVATE ../../../include)
target_compile_features(ce_ccp1_c02_008 PRIVATE cxx_std_17)
add_test(NAME ce_ccp1_c02_008 COMMAND ce_ccp1_c02_008)

add_executable(ce_ccp1_c02_009
    implement_source_aware_fix_its_test.cc
    ../../../src/compiler/ast/implement_source_aware_fix_its.cc)
target_include_directories(ce_ccp1_c02_009 PRIVATE ../../../include)
target_compile_features(ce_ccp1_c02_009 PRIVATE cxx_std_17)
add_test(NAME ce_ccp1_c02_009 COMMAND ce_ccp1_c02_009)

add_executable(ce_ccp1_c02_010
    create_deterministic_ast_dump_and_snapshot_formats_test.cc
    ../../../src/compiler/ast/create_deterministic_ast_dump_and_snapshot_formats.cc)
target_include_directories(ce_ccp1_c02_010 PRIVATE ../../../include)
target_compile_features(ce_ccp1_c02_010 PRIVATE cxx_std_17)
add_test(NAME ce_ccp1_c02_010 COMMAND ce_ccp1_c02_010)
