if(TARGET Cellerator::rds AND TARGET Cellerator::execution_streaming_cuda
   AND TARGET cellerator_compute_sparse_ops)
    find_package(Threads REQUIRED)
    add_executable(cellerator_rds_sell_fetch_bench EXCLUDE_FROM_ALL rds_sell_fetch_bench.cc)
    set_target_properties(cellerator_rds_sell_fetch_bench PROPERTIES
        CXX_EXTENSIONS OFF CXX_SCAN_FOR_MODULES ON)
    target_link_libraries(cellerator_rds_sell_fetch_bench PRIVATE
        Cellerator::rds Cellerator::execution_streaming_cuda cellerator_compute_sparse_ops Threads::Threads)
endif()
