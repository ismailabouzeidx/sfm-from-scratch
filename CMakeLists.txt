cmake_minimum_required(VERSION 3.10)

# Set the project name
project(sfm_from_scratch)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(OpenCV REQUIRED)

find_package(X11 REQUIRED)

# Add the executable
add_executable(sfm_from_scratch src/main.cpp)

# Include directories
include_directories(include ${OpenCV_INCLUDE_DIRS} ${X11_INCLUDE_DIR})

target_link_libraries(sfm_from_scratch ${OpenCV_LIBS} ${X11_LIBRARIES} stdc++fs)
