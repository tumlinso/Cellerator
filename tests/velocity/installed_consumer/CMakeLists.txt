cmake_minimum_required(VERSION 3.28)
project(CelleratorVelocityInstalledConsumer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Cellerator CONFIG REQUIRED COMPONENTS core_module velocity)
find_package(anndatapp CONFIG REQUIRED)

add_executable(cellerator_velocity_installed_consumer main.cpp)
target_link_libraries(cellerator_velocity_installed_consumer PRIVATE
    Cellerator::core_module
    Cellerator::velocity
    anndatapp::anndatapp)

enable_testing()
add_test(NAME cellerator_velocity_installed_consumer
    COMMAND cellerator_velocity_installed_consumer)
