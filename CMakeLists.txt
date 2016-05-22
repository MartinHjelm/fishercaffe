CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(FisherCaffe)

IF(APPLE)
  SET(HOME /Users/martinhjelm)
  SET(CMAKE_CXX_FLAGS "-std=c++11 -O4 -stdlib=libc++ -Wno-unused-variable")
  SET(EXECUTABLE_FLAG MACOSX_BUNDLE)
ELSE(APPLE)
  SET(HOME /home/martin)
  SET(CMAKE_CXX_FLAGS "-std=c++0x -O4 -fPIC -Wno-unused-variable")
ENDIF(APPLE)

FIND_PACKAGE(OpenCV REQUIRED )
FIND_PACKAGE(BOOST REQUIRED )

SET(CMAKE_MODULE_PATH "/usr/local/Cellar/eigen/3.2.8/share/cmake/Modules;${CMAKE_MODULE_PATH}")
FIND_PACKAGE(Eigen3 REQUIRED )


SET(VLFEAT_INCLUDE_DIRS "PATH-TO/vlfeat")
SET(VLFEAT_LIBS "PATH-TO/vlfeat")

FIND_PACKAGE(Boost COMPONENTS random system filesystem thread date_time iostreams serialization chrono REQUIRED)
# Might be needed if using homebrew
# LINK_DIRECTORIES(/usr/local/Cellar/tbb/4.4-20160128/lib)

INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${VLFEAT_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})

ADD_DEFINITIONS("-DVL_DISABLE_AVX")
FILE(GLOB SOURCES "PATH-TO"vlfeat/vl/*.c)
MESSAGE(STATUS "    sources path: ${SOURCES}")
ADD_LIBRARY(vlfeats ${SOURCES} )

ADD_EXECUTABLE(fisher fisher.cpp)
TARGET_LINK_LIBRARIES(fisher ${OpenCV_LIBS} vlfeats ${Boost_LIBRARIES} ${OpenCV_LIBRARIES} )