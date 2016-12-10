cmake_minimum_required(VERSION 2.8)
project(imageAligen)

OPTION(DEFINE_DBUG_INFOUT "Output the debug information" ON)

#===============Compiler Configure====================
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_BUILD_TYPE Release)

#==============Compiler Definitions===================
message("DEFINE_DBUG_INFOUT:${DEFINE_DBUG_INFOUT}")
if(DEFINE_DBUG_INFOUT)
	add_definitions(-DDEBUG_INF_OUT)
endif(DEFINE_DBUG_INFOUT)

#==============OpenCV Library=========================
set(OpenCV_DIR "B:/opencv/build")

find_package(OpenCV 3.1.0 EXACT REQUIRED)
if(OpenCV_FOUND)
    message("${Red}-- Find OpenCV library 3.1.0${ColourReset}")
    message("-- OpenCV Dir:${OpenCV_INCLUDE_DIRS}")
    include_directories(${OpenCV_INCLUDE_DIRS})
    link_directories(${OpenCV_LIBRARY_DIRS})
endif()
#==============Create executable file==================
include_directories(${PROJECT_SOURCE_DIR}/include)
add_executable(test_invcomp  test/test_invcomp.cpp
                        src/invcomposit.cpp
                        src/visionkit.cpp)

add_executable(test_additive test/test_additive.cpp
                        src/additive.cpp
                        src/visionkit.cpp)

add_executable(test_composit test/test_composit.cpp
                        src/composit.cpp
                        src/visionkit.cpp)

add_executable(test_invaddt test/test_invaddt.cpp
                        src/invadditive.cpp
                        src/visionkit.cpp)

#==============Link libraries to executable file=======
target_link_libraries(test_invcomp ${OpenCV_LIBS})
target_link_libraries(test_additive ${OpenCV_LIBS})
target_link_libraries(test_composit ${OpenCV_LIBS})
target_link_libraries(test_invaddt ${OpenCV_LIBS})