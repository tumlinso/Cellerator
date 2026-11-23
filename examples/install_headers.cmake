install(DIRECTORY
    "${CMAKE_CURRENT_LIST_DIR}/models"
    "${CMAKE_CURRENT_LIST_DIR}/trajectory"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/Cellerator/examples"
    FILES_MATCHING PATTERN "*.hh" PATTERN "*.cuh")
