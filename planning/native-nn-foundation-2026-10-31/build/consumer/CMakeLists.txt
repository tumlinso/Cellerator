cmake_minimum_required(VERSION 3.20)
project(CelleratorNNInstalledConsumer LANGUAGES CXX)

find_package(Cellerator REQUIRED COMPONENTS nn)

add_executable(nn_installed_consumer main.cc)
target_link_libraries(nn_installed_consumer PRIVATE Cellerator::nn)
target_compile_features(nn_installed_consumer PRIVATE cxx_std_20)
