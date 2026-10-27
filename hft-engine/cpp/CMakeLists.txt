cmake_minimum_required(VERSION 3.15)
project(hft_cpp VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find pybind11 (installed via pip)
find_package(pybind11 REQUIRED)

# Collect all C++ sources
file(GLOB SOURCES "*.cpp")

# Build the shared library
pybind11_add_module(hft_cpp ${SOURCES})

# Link libraries (none beyond pybind11)
target_link_libraries(hft_cpp PRIVATE pybind11::headers)

# Set output directory (per-config for MSVC)
set_target_properties(hft_cpp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/../src/hft_engine"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/../src/hft_engine"
    LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_SOURCE_DIR}/../src/hft_engine"
)

# Concurrency stress and microbenchmark targets. Off by default; CI builds the
# stress target with ThreadSanitizer.
option(HFT_BUILD_STRESS "Build the concurrency stress test and benchmark" OFF)
if(HFT_BUILD_STRESS)
    find_package(Threads REQUIRED)
    add_executable(hft_stress tests/stress.cpp)
    add_executable(hft_bench tests/bench_lockfree.cpp)
    foreach(the_target hft_stress hft_bench)
        target_include_directories(${the_target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
        target_link_libraries(${the_target} PRIVATE Threads::Threads)
        set_target_properties(${the_target} PROPERTIES
            CXX_STANDARD 17
            CXX_STANDARD_REQUIRED ON
        )
    endforeach()
endif()
