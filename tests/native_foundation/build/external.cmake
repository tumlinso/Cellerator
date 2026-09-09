# Configure and build the producer independently: no same-TU source inclusion.
execute_process(COMMAND "${CMAKE_COMMAND}" -S "${REPO_ROOT}" -B "${TEST_ROOT}/producer"
    -DCELLERATOR_ENABLE_CUDA=OFF -DCELLERATOR_NATIVE_FOUNDATION_ONLY=ON
    -DCELLERATOR_BUILD_NATIVE_FOUNDATION_TESTS=OFF
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "Producer configure failed: ${output}${error}")
endif()
execute_process(COMMAND "${CMAKE_COMMAND}" --build "${TEST_ROOT}/producer" -j 2
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "Producer libraries failed: ${output}${error}")
endif()
file(MAKE_DIRECTORY "${TEST_ROOT}/elsewhere/consumer")
file(WRITE "${TEST_ROOT}/elsewhere/consumer/CMakeLists.txt"
    "cmake_minimum_required(VERSION 3.20)\nproject(external LANGUAGES CXX)\nfind_package(CelleratorNativeFoundation CONFIG REQUIRED)\nadd_executable(consumer \"${REPO_ROOT}/src/compute/candidate/segment/reduce_v2_test.cc\")\ntarget_link_libraries(consumer PRIVATE Cellerator::native_foundation)\nif(NOT EXISTS \"\${CELLERATOR_NATIVE_FOUNDATION_DEPENDENCY_MANIFEST}\")\nmessage(FATAL_ERROR \"Missing dependency manifest\")\nendif()\n")
execute_process(COMMAND "${CMAKE_COMMAND}" -S "${TEST_ROOT}/elsewhere/consumer"
    -B "${TEST_ROOT}/elsewhere/build" "-DCelleratorNativeFoundation_DIR=${TEST_ROOT}/producer"
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "External package configure failed: ${output}${error}")
endif()
execute_process(COMMAND "${CMAKE_COMMAND}" --build "${TEST_ROOT}/elsewhere/build" -j 2
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "External link failed: ${output}${error}")
endif()
execute_process(COMMAND "${TEST_ROOT}/elsewhere/build/consumer" RESULT_VARIABLE result)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "External linked execution failed")
endif()
file(READ "${TEST_ROOT}/producer/CelleratorNativeFoundationDependencies.json" manifest)
string(JSON kind GET "${manifest}" kind)
string(JSON count LENGTH "${manifest}" files)
if(NOT kind STREQUAL "nf1-build-dependencies-v1" OR count LESS 10)
    message(FATAL_ERROR "Dependency fingerprint manifest incomplete")
endif()
