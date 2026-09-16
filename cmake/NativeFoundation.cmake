# Linked numerical foundation used by external native consumers.  This preserves
# the prepared-program/session and relation owners; it does not create a runner.
add_subdirectory(src/compute/operation/native_numeric)

add_library(cellerator_native_foundation INTERFACE)
add_library(Cellerator::native_foundation ALIAS cellerator_native_foundation)
target_link_libraries(cellerator_native_foundation INTERFACE
    Cellerator::native_numeric
    Cellerator::executable_program)

if(TARGET cellerator_prepared_relation_cuda)
    target_link_libraries(cellerator_native_foundation INTERFACE
        Cellerator::prepared_relation_cuda)
    # The historical V07 facade also requires atom-plane owners.  Preserve the
    # real CUDA pair whenever available; publish that facade only when its
    # existing owners are present rather than introducing a duplicate owner.
    if(TARGET Cellerator::jbc_v1 AND TARGET cellerator_compute_sparse_project)
        add_subdirectory(src/execution/native_value_instance)
        target_link_libraries(cellerator_native_foundation INTERFACE
            Cellerator::native_value_instance)
    endif()
endif()

if(CELLERATOR_BUILD_NATIVE_FOUNDATION_TESTS)
    enable_testing()
    add_subdirectory(tests/native_foundation/reference)
    add_subdirectory(tests/native_foundation/numeric)
    add_subdirectory(tests/native_foundation/program)
    if(TARGET Cellerator::native_value_instance)
        add_subdirectory(tests/native_foundation/values)
    endif()
endif()
