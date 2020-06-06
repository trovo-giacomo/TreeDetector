cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(treeDetector)

find_package(OpenCV REQUIRED)

include_directories(include ${OpenCV_INCLUDE_DIRS})

add_executable(${PROJECT_NAME} src/main.cpp)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

add_executable(erosion src/erosion.cpp)
target_link_libraries(erosion ${OpenCV_LIBS})