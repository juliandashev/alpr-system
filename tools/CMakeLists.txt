cmake_minimum_required(VERSION 3.10)

project(KNN_Trainer
    VERSION 1.0.0
    LANGUAGES CXX
)

# =====================
# C++ standard (C++17 required for std::filesystem)
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
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

# =====================
# Find OpenCV
# =====================
find_package(OpenCV REQUIRED)

# =====================
# Add executable
# =====================
add_executable(${PROJECT_NAME} train_knn.cpp)

# =====================
# Link headers
# =====================
target_include_directories(${PROJECT_NAME}
    PRIVATE ${OpenCV_INCLUDE_DIRS}
)

# =====================
# Link OpenCV
# =====================
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

# =====================
# Info
# =====================
message(STATUS "System: ${CMAKE_SYSTEM_NAME}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
