cmake_minimum_required(VERSION 3.10)

project(ALPR_System
    VERSION 1.0.0
    LANGUAGES CXX
)

# =====================
# C++ standard
# =====================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# =====================
# Default build type
# =====================
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "No build type specified. Defaulting to Release.")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# =====================
# Compiler flags
# =====================
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")

# =====================
# Find OpenCV
# =====================
set(OpenCV_DIR "/usr/lib/x86_64-linux-gnu/cmake/opencv4")
find_package(OpenCV REQUIRED)

# =====================
# Add executable
# =====================
add_executable(${PROJECT_NAME} main.cc)

# =====================
# Link headers
# =====================
target_include_directories(${PROJECT_NAME}
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)

# =====================
# Link OpenCV
# =====================
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

# =====================
# Install binary
# =====================
install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION bin)

# =====================
# Info
# =====================
message(STATUS "System: ${CMAKE_SYSTEM_NAME}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "Arch: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
