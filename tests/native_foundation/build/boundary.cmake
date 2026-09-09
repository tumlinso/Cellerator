file(MAKE_DIRECTORY "${TEST_ROOT}/source")
file(WRITE "${TEST_ROOT}/source/main.cc" "#include \"private.cu\"\nint main() { return 0; }\n")
file(WRITE "${TEST_ROOT}/source/CMakeLists.txt" "cmake_minimum_required(VERSION 3.20)\nproject(boundary LANGUAGES CXX)\nset(CELLERATOR_ENABLE_CUDA OFF CACHE STRING \"\")\nset(CELLERATOR_NATIVE_FOUNDATION_ONLY ON CACHE BOOL \"\")\nadd_subdirectory(\"${REPO_ROOT}\" cellerator)\nadd_executable(consumer main.cc)\ncellerator_link_native_foundation(consumer)\n")
execute_process(COMMAND "${CMAKE_COMMAND}" -S "${TEST_ROOT}/source" -B "${TEST_ROOT}/build"
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(result EQUAL 0 OR NOT "${output}${error}" MATCHES "includes a private implementation")
    message(FATAL_ERROR "Private .cu inclusion was not rejected: ${output}${error}")
endif()
