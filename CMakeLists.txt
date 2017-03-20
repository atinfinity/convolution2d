cmake_minimum_required(VERSION 2.8)
project(convolution2d)

#set(OpenCV_DIR /usr/local/share/OpenCV)
find_package(OpenCV REQUIRED)
file(GLOB SOURCES "*.cpp" "*.c" "*.h" "*.hpp")

if(OpenCV_FOUND)
	add_executable(convolution2d ${SOURCES})
	include_directories(${OpenCV_INCLUDE_DIRS})
	target_link_libraries(convolution2d ${OpenCV_LIBS})
endif(OpenCV_FOUND)
