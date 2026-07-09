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
