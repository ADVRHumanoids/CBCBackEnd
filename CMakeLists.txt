# Copyright: (C) 2018 INAIL Consortium
# Authors: Enrico Mingo Hoffman
# CopyPolicy: Released under the terms of the GNU GPL v2.0.

cmake_minimum_required(VERSION 2.8.12)
if(COMMAND cmake_policy)
    cmake_policy(SET CMP0003 OLD)
    cmake_policy(SET CMP0005 NEW)
    cmake_policy(SET CMP0017 NEW)
endif(COMMAND cmake_policy)
include(ExternalProject)
PROJECT(CBCBackEnd)

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-std=c++11" COMPILER_SUPPORTS_CXX11)
check_cxx_compiler_flag("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wmaybe-uninitialized -Wuninitialized")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -Wmaybe-uninitialized -Wuninitialized")
else()
    message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

option(SET_SSE4_FLAG "set -msse4 flag to gcc" OFF)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror=return-type -Werror=address -Werror=parentheses " )
endif(CMAKE_BUILD_TYPE STREQUAL "Debug")

if(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -std=c++11 -Wmaybe-uninitialized -Wuninitialized" )
endif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")

IF(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
   SET(CMAKE_BUILD_TYPE RelWithDebInfo)
ENDIF(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)

find_package(YCM REQUIRED)
find_package(OpenSoT REQUIRED)
FIND_PACKAGE(Eigen3 REQUIRED)
FIND_PACKAGE(PkgConfig REQUIRED)

include_directories(include ${OpenSoT_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR})

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
include(MacroInstallLib)

pkg_check_modules(CBC REQUIRED cbc)
pkg_check_modules(OSICBC REQUIRED osi-cbc)

add_library(OpenSotBackEndCBC SHARED src/CBCBackEnd.cpp)
target_link_libraries(OpenSotBackEndCBC ${OpenSoT_LIBRARIES} ${CBC_LIBRARIES} ${OSICBC_LIBRARIES})
library_install(OpenSotBackEndCBC 1 0 0)

if(CBCBACKEND_COMPILE_TESTS)
  enable_testing()
    add_subdirectory(tests)
endif()
