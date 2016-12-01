cmake_minimum_required(VERSION 2.8)
project(imageAligen)

#===============Compiler Configure===================
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

set(CMAKE_BUILD_TYPE Release)
#==============OpenCV Library=========================
set(OpenCV_DIR "B://opencv-3.1.0//bulid//")

find_package(OpenCV 3.1.0 EXACT REQUIRED)
if(OpenCV_FOUND)
	message("${Red}-- Find OpenCV library 3.1.0${ColourReset}")
	message("-- OpenCV Dir:${OpenCV_INCLUDE_DIRS}")
	include_directories(${OpenCV_INCLUDE_DIRS})
	link_directories(${OpenCV_LIBRARY_DIRS})
endif()
#==============Create executable file=================
include_directories(${PROJECT_SOURCE_DIR}/include)
add_executable(binfinal src/main.cpp
						src/visionkit.cpp
						src/invcomp.cpp)

#==============Link libraries to executable file======
target_link_libraries(binfinal ${OpenCV_LIBS})
