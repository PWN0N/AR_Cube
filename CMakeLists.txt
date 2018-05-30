cmake_minimum_required(VERSION 3.9)
project(AR_Cube)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV)

include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( "/usr/include/eigen3")
set(SOURCE_FILES main.cpp)

add_executable(AR_Cube main.cpp)
target_link_libraries( AR_Cube ${OpenCV_LIBS} )