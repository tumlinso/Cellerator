function(cellerator_escape_config_string out_var value)
    string(REPLACE "\\" "\\\\" cellerator_escaped "${value}")
    string(REPLACE "\"" "\\\"" cellerator_escaped "${cellerator_escaped}")
    set(${out_var} "${cellerator_escaped}" PARENT_SCOPE)
endfunction()

function(cellerator_cpu_flag_bool out_var flags token)
    if("${flags}" MATCHES "(^|[ \t:])${token}([ \t\n]|$)")
        set(${out_var} 1 PARENT_SCOPE)
    else()
        set(${out_var} 0 PARENT_SCOPE)
    endif()
endfunction()

function(cellerator_probe_value out_var probe_output key default_value)
    string(REGEX MATCH "(^|\n)${key}=([^\n]*)" cellerator_probe_match "${probe_output}")
    if(cellerator_probe_match)
        set(${out_var} "${CMAKE_MATCH_2}" PARENT_SCOPE)
    else()
        set(${out_var} "${default_value}" PARENT_SCOPE)
    endif()
endfunction()

function(cellerator_detect_host_vector_ops)
    set(cellerator_cpuinfo "")
    set(CELLERATOR_HOST_CPU_ARCH "${CMAKE_SYSTEM_PROCESSOR}" PARENT_SCOPE)
    set(CELLERATOR_HOST_CPU_MODEL "unknown" PARENT_SCOPE)

    if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND EXISTS "/proc/cpuinfo")
        file(READ "/proc/cpuinfo" cellerator_cpuinfo)
    endif()
    string(TOLOWER "${cellerator_cpuinfo}" cellerator_cpuinfo_lower)

    set(cellerator_flags "")
    if(cellerator_cpuinfo_lower)
        string(REGEX MATCH "(^|\n)flags[ \t]*:[^\n]*" cellerator_flags_line "${cellerator_cpuinfo_lower}")
        if(cellerator_flags_line)
            set(cellerator_flags " ${cellerator_flags_line} ")
        else()
            string(REGEX MATCH "(^|\n)features[ \t]*:[^\n]*" cellerator_features_line "${cellerator_cpuinfo_lower}")
            set(cellerator_flags " ${cellerator_features_line} ")
        endif()

        string(REGEX MATCH "(^|\n)model name[ \t]*:[ \t]*([^\n]+)" cellerator_model_match "${cellerator_cpuinfo}")
        if(cellerator_model_match)
            string(STRIP "${CMAKE_MATCH_2}" cellerator_cpu_model)
            cellerator_escape_config_string(cellerator_cpu_model_escaped "${cellerator_cpu_model}")
            set(CELLERATOR_HOST_CPU_MODEL "${cellerator_cpu_model_escaped}" PARENT_SCOPE)
        endif()
    endif()

    cellerator_cpu_flag_bool(CELLERATOR_HOST_HAS_SSE2 "${cellerator_flags}" "sse2")
    cellerator_cpu_flag_bool(CELLERATOR_HOST_HAS_SSE3 "${cellerator_flags}" "pni")
    cellerator_cpu_flag_bool(CELLERATOR_HOST_HAS_SSSE3 "${cellerator_flags}" "ssse3")
    cellerator_cpu_flag_bool(CELLERATOR_HOST_HAS_SSE41 "${cellerator_flags}" "sse4_1")
    cellerator_cpu_flag_bool(CELLERATOR_HOST_HAS_SSE42 "${cellerator_flags}" "sse4_2")
    cellerator_cpu_flag_bool(CELLERATOR_HOST_HAS_AVX "${cellerator_flags}" "avx")
    cellerator_cpu_flag_bool(CELLERATOR_HOST_HAS_AVX2 "${cellerator_flags}" "avx2")
    cellerator_cpu_flag_bool(CELLERATOR_HOST_HAS_AVX512F "${cellerator_flags}" "avx512f")
    cellerator_cpu_flag_bool(CELLERATOR_HOST_HAS_AVX512BW "${cellerator_flags}" "avx512bw")
    cellerator_cpu_flag_bool(CELLERATOR_HOST_HAS_AVX512VNNI "${cellerator_flags}" "avx512_vnni")
    cellerator_cpu_flag_bool(CELLERATOR_HOST_HAS_F16C "${cellerator_flags}" "f16c")
    cellerator_cpu_flag_bool(CELLERATOR_HOST_HAS_FMA "${cellerator_flags}" "fma")
    cellerator_cpu_flag_bool(CELLERATOR_HOST_HAS_NEON "${cellerator_flags}" "neon")
    cellerator_cpu_flag_bool(CELLERATOR_HOST_HAS_ASIMD "${cellerator_flags}" "asimd")
    cellerator_cpu_flag_bool(CELLERATOR_HOST_HAS_SVE "${cellerator_flags}" "sve")

    set(cellerator_vector_level "scalar")
    if(CELLERATOR_HOST_HAS_AVX512F)
        set(cellerator_vector_level "avx512")
    elseif(CELLERATOR_HOST_HAS_AVX2)
        set(cellerator_vector_level "avx2")
    elseif(CELLERATOR_HOST_HAS_AVX)
        set(cellerator_vector_level "avx")
    elseif(CELLERATOR_HOST_HAS_SSE42)
        set(cellerator_vector_level "sse4_2")
    elseif(CELLERATOR_HOST_HAS_SSE41)
        set(cellerator_vector_level "sse4_1")
    elseif(CELLERATOR_HOST_HAS_SSE2)
        set(cellerator_vector_level "sse2")
    elseif(CELLERATOR_HOST_HAS_SVE)
        set(cellerator_vector_level "sve")
    elseif(CELLERATOR_HOST_HAS_NEON OR CELLERATOR_HOST_HAS_ASIMD)
        set(cellerator_vector_level "neon")
    endif()
    foreach(cellerator_host_feature
            IN ITEMS
                CELLERATOR_HOST_HAS_SSE2
                CELLERATOR_HOST_HAS_SSE3
                CELLERATOR_HOST_HAS_SSSE3
                CELLERATOR_HOST_HAS_SSE41
                CELLERATOR_HOST_HAS_SSE42
                CELLERATOR_HOST_HAS_AVX
                CELLERATOR_HOST_HAS_AVX2
                CELLERATOR_HOST_HAS_AVX512F
                CELLERATOR_HOST_HAS_AVX512BW
                CELLERATOR_HOST_HAS_AVX512VNNI
                CELLERATOR_HOST_HAS_F16C
                CELLERATOR_HOST_HAS_FMA
                CELLERATOR_HOST_HAS_NEON
                CELLERATOR_HOST_HAS_ASIMD
                CELLERATOR_HOST_HAS_SVE)
        set(${cellerator_host_feature} "${${cellerator_host_feature}}" PARENT_SCOPE)
    endforeach()
    set(CELLERATOR_HOST_VECTOR_LEVEL_ENUM "${cellerator_vector_level}" PARENT_SCOPE)
    message(STATUS "Cellerator host vector level: ${cellerator_vector_level}")
endfunction()

function(cellerator_cuda_arch_family out_var sm)
    if(sm GREATER_EQUAL 100)
        set(${out_var} "blackwell" PARENT_SCOPE)
    elseif(sm GREATER_EQUAL 90)
        set(${out_var} "hopper" PARENT_SCOPE)
    elseif(sm EQUAL 89)
        set(${out_var} "ada" PARENT_SCOPE)
    elseif(sm GREATER_EQUAL 80)
        set(${out_var} "ampere" PARENT_SCOPE)
    elseif(sm EQUAL 75)
        set(${out_var} "turing" PARENT_SCOPE)
    elseif(sm GREATER_EQUAL 70)
        set(${out_var} "volta" PARENT_SCOPE)
    else()
        set(${out_var} "unknown" PARENT_SCOPE)
    endif()
endfunction()

function(cellerator_detect_cuda_hardware)
    set(CELLERATOR_HARDWARE_PROBE_RAN 0 PARENT_SCOPE)
    set(CELLERATOR_HARDWARE_PROBE_OK 0 PARENT_SCOPE)
    set(CELLERATOR_DETECTED_CUDA_DEVICE_COUNT 0 PARENT_SCOPE)
    set(CELLERATOR_DETECTED_CUDA_DEVICE_INDEX -1 PARENT_SCOPE)
    set(CELLERATOR_DETECTED_CUDA_COMPUTE_MAJOR 0 PARENT_SCOPE)
    set(CELLERATOR_DETECTED_CUDA_COMPUTE_MINOR 0 PARENT_SCOPE)
    set(CELLERATOR_DETECTED_CUDA_SM 0 PARENT_SCOPE)
    set(CELLERATOR_DETECTED_CUDA_DEVICE_NAME "unknown" PARENT_SCOPE)
    set(CELLERATOR_DETECTED_CUDA_TOTAL_GLOBAL_MEM_MIB 0 PARENT_SCOPE)
    set(CELLERATOR_DETECTED_CUDA_MULTIPROCESSORS 0 PARENT_SCOPE)
    set(CELLERATOR_DETECTED_CUDA_WARP_SIZE 0 PARENT_SCOPE)
    set(CELLERATOR_DETECTED_CUDA_ARCH_FAMILY_ENUM "unknown" PARENT_SCOPE)
    set(CELLERATOR_DETECTED_CUDA_HAS_FP16_TENSOR_CORES 0 PARENT_SCOPE)
    set(CELLERATOR_DETECTED_CUDA_HAS_BF16_TENSOR_CORES 0 PARENT_SCOPE)
    set(CELLERATOR_DETECTED_CUDA_HAS_TF32_TENSOR_CORES 0 PARENT_SCOPE)
    set(CELLERATOR_DETECTED_CUDA_HAS_FP8_TENSOR_CORES 0 PARENT_SCOPE)

    if(NOT CELLERATOR_ENABLE_HARDWARE_PROBE)
        message(STATUS "Cellerator hardware probe disabled")
        return()
    endif()
    if(CMAKE_CROSSCOMPILING)
        message(STATUS "Cellerator CUDA hardware probe skipped while cross-compiling")
        return()
    endif()

    try_run(
        cellerator_cuda_probe_run_result
        cellerator_cuda_probe_compile_result
        "${CMAKE_CURRENT_BINARY_DIR}/hardware_probe"
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/probe_cuda_device.cc"
        LINK_LIBRARIES CUDA::cudart
        RUN_OUTPUT_VARIABLE cellerator_cuda_probe_output
        COMPILE_OUTPUT_VARIABLE cellerator_cuda_probe_compile_output
    )

    if(NOT cellerator_cuda_probe_compile_result)
        message(WARNING "Cellerator CUDA hardware probe failed to compile: ${cellerator_cuda_probe_compile_output}")
        return()
    endif()

    set(CELLERATOR_HARDWARE_PROBE_RAN 1 PARENT_SCOPE)
    cellerator_probe_value(cellerator_probe_ok "${cellerator_cuda_probe_output}" "cuda_probe_ok" "0")
    cellerator_probe_value(cellerator_gpu_count "${cellerator_cuda_probe_output}" "gpu_count" "0")
    cellerator_probe_value(cellerator_gpu_index "${cellerator_cuda_probe_output}" "selected_device" "-1")
    cellerator_probe_value(cellerator_compute_major "${cellerator_cuda_probe_output}" "compute_major" "0")
    cellerator_probe_value(cellerator_compute_minor "${cellerator_cuda_probe_output}" "compute_minor" "0")
    cellerator_probe_value(cellerator_sm "${cellerator_cuda_probe_output}" "sm" "0")
    cellerator_probe_value(cellerator_gpu_name "${cellerator_cuda_probe_output}" "name" "unknown")
    cellerator_probe_value(cellerator_gpu_mem "${cellerator_cuda_probe_output}" "total_global_mem_mib" "0")
    cellerator_probe_value(cellerator_gpu_sms "${cellerator_cuda_probe_output}" "multiprocessors" "0")
    cellerator_probe_value(cellerator_gpu_warp "${cellerator_cuda_probe_output}" "warp_size" "0")
    cellerator_probe_value(cellerator_fp16_tc "${cellerator_cuda_probe_output}" "has_fp16_tensor_cores" "0")
    cellerator_probe_value(cellerator_bf16_tc "${cellerator_cuda_probe_output}" "has_bf16_tensor_cores" "0")
    cellerator_probe_value(cellerator_tf32_tc "${cellerator_cuda_probe_output}" "has_tf32_tensor_cores" "0")
    cellerator_probe_value(cellerator_fp8_tc "${cellerator_cuda_probe_output}" "has_fp8_tensor_cores" "0")
    cellerator_escape_config_string(cellerator_gpu_name_escaped "${cellerator_gpu_name}")
    cellerator_cuda_arch_family(cellerator_arch_family "${cellerator_sm}")

    set(CELLERATOR_HARDWARE_PROBE_OK "${cellerator_probe_ok}" PARENT_SCOPE)
    set(CELLERATOR_DETECTED_CUDA_DEVICE_COUNT "${cellerator_gpu_count}" PARENT_SCOPE)
    set(CELLERATOR_DETECTED_CUDA_DEVICE_INDEX "${cellerator_gpu_index}" PARENT_SCOPE)
    set(CELLERATOR_DETECTED_CUDA_COMPUTE_MAJOR "${cellerator_compute_major}" PARENT_SCOPE)
    set(CELLERATOR_DETECTED_CUDA_COMPUTE_MINOR "${cellerator_compute_minor}" PARENT_SCOPE)
    set(CELLERATOR_DETECTED_CUDA_SM "${cellerator_sm}" PARENT_SCOPE)
    set(CELLERATOR_DETECTED_CUDA_DEVICE_NAME "${cellerator_gpu_name_escaped}" PARENT_SCOPE)
    set(CELLERATOR_DETECTED_CUDA_TOTAL_GLOBAL_MEM_MIB "${cellerator_gpu_mem}" PARENT_SCOPE)
    set(CELLERATOR_DETECTED_CUDA_MULTIPROCESSORS "${cellerator_gpu_sms}" PARENT_SCOPE)
    set(CELLERATOR_DETECTED_CUDA_WARP_SIZE "${cellerator_gpu_warp}" PARENT_SCOPE)
    set(CELLERATOR_DETECTED_CUDA_ARCH_FAMILY_ENUM "${cellerator_arch_family}" PARENT_SCOPE)
    set(CELLERATOR_DETECTED_CUDA_HAS_FP16_TENSOR_CORES "${cellerator_fp16_tc}" PARENT_SCOPE)
    set(CELLERATOR_DETECTED_CUDA_HAS_BF16_TENSOR_CORES "${cellerator_bf16_tc}" PARENT_SCOPE)
    set(CELLERATOR_DETECTED_CUDA_HAS_TF32_TENSOR_CORES "${cellerator_tf32_tc}" PARENT_SCOPE)
    set(CELLERATOR_DETECTED_CUDA_HAS_FP8_TENSOR_CORES "${cellerator_fp8_tc}" PARENT_SCOPE)

    if(cellerator_probe_ok)
        message(STATUS
            "Cellerator CUDA probe: ${cellerator_gpu_name} sm_${cellerator_sm} "
            "(${cellerator_arch_family}), ${cellerator_gpu_mem} MiB")
    else()
        message(STATUS "Cellerator CUDA probe: no usable CUDA device visible")
    endif()
endfunction()
