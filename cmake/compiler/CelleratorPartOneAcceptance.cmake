function(cellerator_add_part_one_acceptance suffix source)
    set(target "ce_ccp1_j03_${suffix}")
    if(NOT TARGET ${target} AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${source}")
        add_executable(${target} "${source}")
        target_include_directories(${target} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
        target_compile_features(${target} PRIVATE cxx_std_17)
        add_test(NAME ${target} COMMAND ${target})
        set_tests_properties(${target} PROPERTIES LABELS "ce_ccp1_m90;part_one")
    endif()
endfunction()

cellerator_add_part_one_acceptance(001
    tests/compiler/j03/integrate_central_compiler_targets_and_registries_test.cc)
cellerator_add_part_one_acceptance(002
    tests/compiler/j03/integrate_jbc_migration_and_cellshard_compatibility_test.cc)
cellerator_add_part_one_acceptance(003
    tests/compiler/j03/reconcile_language_specification_with_implementation_test.cc)
cellerator_add_part_one_acceptance(004
    tests/compiler/j03/reconcile_ir_specification_with_implementation_test.cc)
cellerator_add_part_one_acceptance(005
    tests/compiler/j03/reconcile_programming_guides_and_examples_test.cc)
cellerator_add_part_one_acceptance(006
    tests/compiler/j03/publish_architecture_and_migration_completion_records_test.cc)
cellerator_add_part_one_acceptance(007
    tests/compiler/j03/run_clean_host_only_sdk_acceptance_test.cc)
cellerator_add_part_one_acceptance(008
    tests/compiler/j03/run_clean_nvidia_sdk_acceptance_test.cc)
cellerator_add_part_one_acceptance(009
    tests/compiler/j03/validate_all_final_part_one_capabilities_test.cc)
cellerator_add_part_one_acceptance(010
    tests/compiler/j03/audit_deferred_part_two_separation_test.cc)
cellerator_add_part_one_acceptance(011
    tests/compiler/j03/run_final_performance_and_regression_review_test.cc)
cellerator_add_part_one_acceptance(012
    tests/compiler/j03/create_release_and_bootstrap_reproducibility_bundle_test.cc)
cellerator_add_part_one_acceptance(013
    tests/compiler/j03/freeze_part_one_completion_checkpoint_test.cc)
