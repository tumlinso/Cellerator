function(cellerator_add_release_acceptance source)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${source}")
        get_filename_component(test_name "${source}" NAME_WE)
        set(target "cellerator_${test_name}")
        if(NOT TARGET ${target})
            add_executable(${target} "${source}")
            target_include_directories(${target} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
            target_compile_features(${target} PRIVATE cxx_std_17)
            add_test(NAME ${target} COMMAND ${target})
            set_tests_properties(${target} PROPERTIES LABELS "compiler_release_acceptance")
        endif()
    endif()
endfunction()

cellerator_add_release_acceptance(
    tests/compiler/release/integrate_central_compiler_targets_and_registries_test.cc)
cellerator_add_release_acceptance(
    tests/compiler/release/integrate_jbc_migration_and_cellshard_compatibility_test.cc)
cellerator_add_release_acceptance(
    tests/compiler/release/reconcile_language_specification_with_implementation_test.cc)
cellerator_add_release_acceptance(
    tests/compiler/release/reconcile_ir_specification_with_implementation_test.cc)
cellerator_add_release_acceptance(
    tests/compiler/release/reconcile_programming_guides_and_examples_test.cc)
cellerator_add_release_acceptance(
    tests/compiler/release/architecture_and_migration_records_test.cc)
cellerator_add_release_acceptance(
    tests/compiler/release/run_clean_host_only_sdk_acceptance_test.cc)
cellerator_add_release_acceptance(
    tests/compiler/release/run_clean_nvidia_sdk_acceptance_test.cc)
cellerator_add_release_acceptance(
    tests/compiler/release/validate_all_final_part_one_capabilities_test.cc)
cellerator_add_release_acceptance(
    tests/compiler/release/audit_deferred_part_two_separation_test.cc)
cellerator_add_release_acceptance(
    tests/compiler/release/run_final_performance_and_regression_review_test.cc)
cellerator_add_release_acceptance(
    tests/compiler/release/release_and_bootstrap_reproducibility_bundle_test.cc)
cellerator_add_release_acceptance(
    tests/compiler/release/part_one_completion_checkpoint_test.cc)
