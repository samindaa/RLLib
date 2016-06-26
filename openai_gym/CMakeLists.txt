cmake_minimum_required(VERSION 2.8.7)

project(openai_gym)

SET(CMAKE_C_COMPILER ${CMAKE_CXX_COMPILER})

SET(Boost_USE_STATIC_LIBS OFF)
SET(Boost_USE_MULTITHREAD ON)
FIND_PACKAGE(Boost 1.59.0 REQUIRED COMPONENTS system thread regex)
IF(Boost_FOUND)
  INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
  LINK_DIRECTORIES(${Boost_LIBRARY_DIRS})
ENDIF(Boost_FOUND)

SET(USED_LIBS ${Boost_SYSTEM_LIBRARY} ${Boost_THREAD_LIBRARY} ${Boost_REGEX_LIBRARY})


list( APPEND CMAKE_CXX_FLAGS "-mmmx -msse -msse2 -msse3 -mssse3 -Wno-deprecated -ggdb -O3 -std=c++0x -fPIC -fno-rtti -U_FORTIFY_SOURCE")

file(GLOB FWX_SOURCES1 "*.cpp")
set( FWX_SOURCES "")
list(APPEND FWX_SOURCES ${FWX_SOURCES1})

set (FWX_INCLUDE_DIRS ".")
list (APPEND FWX_INCLUDE_DIRS "../include")
list(REMOVE_DUPLICATES FWX_INCLUDE_DIRS)
include_directories(${FWX_INCLUDE_DIRS})


add_executable(openai_gym ${FWX_SOURCES})
TARGET_LINK_LIBRARIES(openai_gym ${USED_LIBS})
