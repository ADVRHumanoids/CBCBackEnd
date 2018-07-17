cmake_minimum_required(VERSION 2.8.11)
include(ExternalProject)

set(PROJECTNAME tests)
project(${PROJECTNAME})

# compite GTest
find_package(GTest)
if (NOT GTEST_FOUND)
    set(gTestSource "/usr/src/gtest")
    ExternalProject_Add(GTest-ext SOURCE_DIR ${gTestSource}
                                  PREFIX "${CMAKE_CURRENT_BINARY_DIR}/external"
                                  INSTALL_COMMAND "")
    set(GTEST_LIB_DIRS "${CMAKE_CURRENT_BINARY_DIR}/external/src/GTest-ext-build/")
    set(GTEST_INCLUDE_DIRS ${gTestSource})
    set(GTEST_BOTH_LIBRARIES gtest gtest_main)
endif()

include_directories(${GTEST_INCLUDE_DIRS})
link_directories(${GTEST_LIB_DIRS})

set(CBCBackEnd_TESTS testCBCSolver)

set(TestLibs ${GTEST_BOTH_LIBRARIES} ${OpenSoT_LIBRARIES})

ADD_EXECUTABLE(testCBCSolver TestCBC.cpp)
TARGET_LINK_LIBRARIES(testCBCSolver ${TestLibs})
add_dependencies(testCBCSolver GTest-ext OpenSoT)
add_test(NAME OpenSoT_solvers_cbc COMMAND testCBCSolver)
