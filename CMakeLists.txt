cmake_minimum_required(VERSION 3.9)
project(sft)
find_package(OpenCV REQUIRED)
set(CMAKE_CXX_STANDARD 17)

add_executable(sft main.cpp)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(sft ${OpenCV_LIBS})