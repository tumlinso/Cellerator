include_guard(GLOBAL)

function(cellerator_generate_backend_resource_manifest output)
    find_program(CELLERATOR_NVCC nvcc)
    find_program(CELLERATOR_CLANG_CUDA NAMES clang++-18 clang++)
    find_program(CELLERATOR_LLVM_CONFIG NAMES llvm-config-18 llvm-config)
    find_program(CELLERATOR_LLC NAMES llc-18 llc)
    find_program(CELLERATOR_PTXAS ptxas)
    find_program(CELLERATOR_LINKER NAMES ld.lld-18 ld.lld ld)
    set(cellerator_manifest
        "host_cxx=${CMAKE_CXX_COMPILER}\n"
        "nvcc=${CELLERATOR_NVCC}\n"
        "clang_cuda=${CELLERATOR_CLANG_CUDA}\n"
        "llvm_config=${CELLERATOR_LLVM_CONFIG}\n"
        "nvptx=${CELLERATOR_LLC}\n"
        "ptxas=${CELLERATOR_PTXAS}\n"
        "linker=${CELLERATOR_LINKER}\n"
        "resource_dir=${CMAKE_INSTALL_PREFIX}/lib/cellerator\n")
    file(GENERATE OUTPUT "${output}" CONTENT "${cellerator_manifest}")
endfunction()
