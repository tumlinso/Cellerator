include(GNUInstallDirs)
configure_file("${CMAKE_CURRENT_LIST_DIR}/cellerator.pc.in" "${CMAKE_CURRENT_BINARY_DIR}/cellerator.pc" @ONLY)
set(CELLERATOR_PKGCONFIG_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/pkgconfig")
