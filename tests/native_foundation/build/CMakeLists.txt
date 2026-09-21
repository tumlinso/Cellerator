# Configure a separate consumer only from the installed Cellerator package.
add_test(NAME ce_nf1_installed_consumer
    COMMAND ${CMAKE_COMMAND}
        -DCE_NF1_CONSUMER_SOURCE=${CMAKE_CURRENT_SOURCE_DIR}/consumer
        -DCE_NF1_CONSUMER_BINARY=${CMAKE_CURRENT_BINARY_DIR}/installed-consumer
        -DCE_NF1_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
        -DCE_NF1_CUDA_COMPILER=${CMAKE_CUDA_COMPILER}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/run_installed_consumer.cmake)
set_tests_properties(ce_nf1_installed_consumer PROPERTIES
    LABELS "nf1;installed;cuda;consumer;width;sm70" TIMEOUT 180)
