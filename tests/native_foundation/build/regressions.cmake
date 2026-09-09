execute_process(COMMAND "${CMAKE_COMMAND}" -S "${REPO_ROOT}" -B "${TEST_ROOT}/minimal"
    -DCELLERATOR_ENABLE_CUDA=OFF -DCELLERATOR_NATIVE_FOUNDATION_ONLY=ON
    -DCELLERATOR_BUILD_NATIVE_FOUNDATION_TESTS=OFF
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "Minimal configuration failed: ${output}${error}")
endif()
execute_process(COMMAND "${CMAKE_COMMAND}" -S "${REPO_ROOT}" -B "${TEST_ROOT}/retained"
    -DCELLERATOR_ENABLE_CUDA=OFF -DCELLERATOR_BUILD_RELATION_UPDATE_SPINE_V1=ON
    -DCELLERATOR_BUILD_TESTS=ON RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "Retained regression configuration failed: ${output}${error}")
endif()
execute_process(COMMAND "${CMAKE_COMMAND}" --build "${TEST_ROOT}/retained"
    --target ceRU1HostTests ceRU1ReferenceTests -j 2
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "Retained regression build failed: ${output}${error}")
endif()
execute_process(COMMAND "${CMAKE_CTEST_COMMAND}" --test-dir "${TEST_ROOT}/retained"
    -R "^(ru1_calculus|ru1_reference)$" --output-on-failure --no-tests=error
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(NOT result EQUAL 0 OR NOT output MATCHES "2/2")
    message(FATAL_ERROR "Retained regression execution failed: ${output}${error}")
endif()
execute_process(COMMAND "${CMAKE_COMMAND}" -S "${REPO_ROOT}" -B "${TEST_ROOT}/conflict"
    -DCELLERATOR_ENABLE_CUDA=OFF -DCELLERATOR_NATIVE_FOUNDATION_ONLY=ON
    -DCELLERATOR_BUILD_RELATION_UPDATE_SPINE_V1=ON
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(result EQUAL 0 OR NOT "${output}${error}" MATCHES "not silently omitted")
    message(FATAL_ERROR "Requested regression suite was silently hidden")
endif()
