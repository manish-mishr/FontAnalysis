cmake_minimum_required(VERSION 3.9)
project(ClionProjects)

set(CMAKE_CXX_STANDARD 11)
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(ClionProjects main.cpp fontHeader.cpp)
target_link_libraries( ClionProjects ${OpenCV_LIBS} )