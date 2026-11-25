cmake_minimum_required(VERSION 3.28)
project(CelleratorRdsConsumer LANGUAGES CXX)

find_package(Cellerator CONFIG REQUIRED COMPONENTS rds)
add_executable(rds_consumer main.cc)
set_target_properties(rds_consumer PROPERTIES CXX_EXTENSIONS OFF)
target_link_libraries(rds_consumer PRIVATE Cellerator::rds)
