cmake_minimum_required (VERSION 2.8.3)
project (RamanAssistedControl)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

#include_directories(${PROJECT_SOURCE_DIR})
#link_directories(${PROJECT_SOURCE_DIR})

# add_library(RamanAssistedControl_utils SHARED RamanAssistedControl_utils.c)

add_library(RamanAssistedControl SHARED RamanAssistedControl.c)
target_link_libraries(RamanAssistedControl nlopt m)
