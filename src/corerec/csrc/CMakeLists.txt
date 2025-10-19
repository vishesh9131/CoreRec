cmake_minimum_required(VERSION 3.10)
project(corerec_csrc)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python and NumPy
find_package(PythonLibs REQUIRED)
execute_process(
    COMMAND python -c "import numpy; print(numpy.get_include())"
    OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Get Python executable path and version information
execute_process(
    COMMAND python -c "import sys; print(sys.executable)"
    OUTPUT_VARIABLE PYTHON_EXECUTABLE
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
    OUTPUT_VARIABLE PYTHON_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

message(STATUS "Using Python: ${PYTHON_EXECUTABLE}")
message(STATUS "Python version: ${PYTHON_VERSION}")
message(STATUS "Python include: ${PYTHON_INCLUDE_DIRS}")
message(STATUS "Python library: ${PYTHON_LIBRARIES}")
message(STATUS "NumPy include: ${NUMPY_INCLUDE_DIR}")

# Include directories
include_directories(
    ${PYTHON_INCLUDE_DIRS}
    ${NUMPY_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Add library
add_library(tensor_ops MODULE
    tensor/tensor.cpp
    ops/embedding_ops.cpp
    python/module.cpp
)

# Set output name
set_target_properties(tensor_ops PROPERTIES
    PREFIX ""
    OUTPUT_NAME "tensor_ops"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/csrc"
)

# Link against Python
target_link_libraries(tensor_ops ${PYTHON_LIBRARIES})