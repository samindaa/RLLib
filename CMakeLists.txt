cmake_minimum_required(VERSION 2.8.7)

project(RLLib)

SET(CMAKE_C_COMPILER ${CMAKE_CXX_COMPILER})

list( APPEND CMAKE_CXX_FLAGS "-mmmx -msse -msse2 -msse3 -mssse3 -Wno-deprecated -ggdb -O3 -std=c++0x -fPIC -fno-rtti -U_FORTIFY_SOURCE")

file(GLOB FWX_SOURCES1 "test/*.cpp")
file(GLOB FWX_SOURCES2 "util/cma/*.c")
file(GLOB FWX_SOURCES3 "util/TreeFitted/*.cpp")
set( FWX_SOURCES "")
list(APPEND FWX_SOURCES ${FWX_SOURCES1})
list(APPEND FWX_SOURCES ${FWX_SOURCES2})
list(APPEND FWX_SOURCES ${FWX_SOURCES3})

set (FWX_INCLUDE_DIRS ".")
list (APPEND FWX_INCLUDE_DIRS "include")
list (APPEND FWX_INCLUDE_DIRS "test")
list (APPEND FWX_INCLUDE_DIRS "simulation")
list (APPEND FWX_INCLUDE_DIRS "util")
list(REMOVE_DUPLICATES FWX_INCLUDE_DIRS)
include_directories(${FWX_INCLUDE_DIRS})


file(COPY test/test.cfg DESTINATION test)

find_package(Threads)
add_executable(RLLib ${FWX_SOURCES})
