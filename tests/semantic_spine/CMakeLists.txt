# All suites execute the original independent test programs with assertions on.
function(spine_test target)
    add_executable(${target} ${ARGN})
    target_compile_features(${target} PRIVATE cxx_std_20)
    target_compile_options(${target} PRIVATE -UNDEBUG)
    target_include_directories(${target} PRIVATE "${PROJECT_SOURCE_DIR}/include")
    set_target_properties(${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
endfunction()

# Keep each test's real main and isolate CUDA implementation translation units.
# The generated runner propagates failed exits and signals from every child.
function(spine_suite target gpu)
    set(children "")
    foreach(child IN LISTS ARGN)
        string(APPEND children "\"$<TARGET_FILE:${child}>\",\n")
    endforeach()
    set(source "${CMAKE_CURRENT_BINARY_DIR}/${target}_runner.cc")
    file(GENERATE OUTPUT "${source}" CONTENT "
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <spawn.h>
#include <sys/wait.h>
extern char **environ;
int main(int argc, char **argv) {
    if (argc != 1 && !(${gpu} && argc == 2 && std::strcmp(argv[1], \"--require-sm70\") == 0)) {
        std::fprintf(stderr, \"unexpected suite argument\\n\"); return 2;
    }
    const char *programs[] = {${children}};
    for (const char *program : programs) {
        std::printf(\"RUN %s\\n\", program); std::fflush(stdout);
        char *args[] = {const_cast<char *>(program), nullptr};
        pid_t pid;
        int rc = posix_spawn(&pid, program, nullptr, nullptr, args, environ);
        if (rc != 0) { std::fprintf(stderr, \"spawn: %s\\n\", std::strerror(rc)); return 1; }
        int status;
        do { rc = waitpid(pid, &status, 0); } while (rc == -1 && errno == EINTR);
        if (rc == -1 || !WIFEXITED(status) || WEXITSTATUS(status) != 0) return 1;
    }
    return 0;
}
")
    spine_test(${target} "${source}")
    add_dependencies(${target} ${ARGN})
endfunction()

add_library(cellerator_spine_algebra STATIC
    src/compute/decomposition/support_embedding_v1.cc
    src/compute/operation/operation_core_v2/schema.cc)
target_include_directories(cellerator_spine_algebra PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_compile_features(cellerator_spine_algebra PUBLIC cxx_std_20)
spine_test(ceSpineAlgebra_support_product tests/semantic_spine/algebra/support_product_test.cc)
target_link_libraries(ceSpineAlgebra_support_product PRIVATE cellerator_spine_algebra)
spine_test(ceSpineAlgebra_legacy_support tests/jbc/decomposition/test_support_embedding_v1.cc)
target_link_libraries(ceSpineAlgebra_legacy_support PRIVATE cellerator_spine_algebra)
spine_suite(ceSpineAlgebraTest 0 ceSpineAlgebra_support_product ceSpineAlgebra_legacy_support)

set(native_tests)
foreach(name adapter preparation generation forward transpose lifecycle overhead)
    spine_test(ceSpineNative_${name} tests/semantic_spine/native/${name}_test.cu)
    target_link_libraries(ceSpineNative_${name} PRIVATE cellerator_relation_update_support
        Cellerator::feature_major_small_n_candidate Cellerator::transpose_backward_candidate CUDA::cudart)
    list(APPEND native_tests ceSpineNative_${name})
endforeach()
spine_suite(ceSpineNativeTest 1 ${native_tests})

spine_test(ceSpineOracleTest tests/semantic_spine/verification/oracle_test.cc)
spine_test(ceSpineContractProbes tests/semantic_spine/verification/contract_probes.cc)
target_link_libraries(ceSpineContractProbes PRIVATE Cellerator::relation_semantics)
