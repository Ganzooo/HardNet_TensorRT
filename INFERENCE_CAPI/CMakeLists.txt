cmake_minimum_required(VERSION 2.8)
project(demo)
set(CMAKE_CXX_STANDARD 11)
find_package(PkgConfig REQUIRED)
find_package(OpenCV REQUIRED)

#set(CMAKE_BUILD_TYPE Release) if uncomment this line, eth cam don't work
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors")

#add wrapper
set(TRT_WRAPPER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tensorRTWrapper/code)
add_subdirectory(${TRT_WRAPPER_DIR})
include_directories(${TRT_WRAPPER_DIR}/include)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

#add opencv
FIND_PACKAGE(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARIES_DIRS})

#build runFCHardNet
add_executable(runFCHardNet main.cpp eval.cpp dataReader.cpp)

#set(TENSORRT_LIBRARY /TensorRT-7.1.3.4/lib/)
#target_link_libraries(TrtNet /TensorRT-7.1.3.4/lib/libnvinfer.so)
#target_link_libraries(runFCHardNet ${OpenCV_LIBS})
#target_link_libraries(runFCHardNet TrtNet /TensorRT-7.1.3.4/lib/libnvinfer.so)

target_link_libraries(runFCHardNet ${OpenCV_LIBS} TrtNet ${OpenCV_LIBS})

install(TARGETS runFCHardNet RUNTIME DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/)
